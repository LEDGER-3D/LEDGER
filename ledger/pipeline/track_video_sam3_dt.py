"""Detect-then-track variant of the open-world SAM3 pipeline.

Instead of letting Sam3VideoModel re-detect on every frame (which mints new object
IDs mid-video and fragments tracks), this:

  Phase 1 (detect)  -- run Sam3VideoModel on the first `detect_frames` frames only,
                       to create masks from the text prompts. The detected instances
                       are de-duplicated with a one-time IoU NMS so overlapping
                       concepts (bicycle / wheel / tire / bicycle wheel) collapse to a
                       distinct set of seed objects.
  Phase 2 (track)   -- seed those masks into Sam3TrackerVideoModel (a pure mask
                       propagator with NO detector and NO keep-alive removal) and
                       propagate across the whole video. No new objects are created;
                       each seed is tracked to the end.

This gives stable, distinct object IDs for cross-frame tracking. Use the per-frame
re-detecting pipeline (track_video_sam3_concept.py) instead when objects enter the
scene mid-video and you want them picked up automatically.
"""
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

import object_memory as omem

from transformers import (
    Sam3VideoModel, Sam3VideoProcessor,
    Sam3TrackerVideoModel, Sam3TrackerVideoProcessor,
)

# Reuse frame IO + rendering from the concept module.
from track_video_sam3_concept import _color, render_objects, MASK_ALPHA
from track_video_sam3 import _load_video_frames

HF_NAME = "facebook/sam3"
DTYPE = torch.bfloat16
STRIDE = 1
TRACK_STRIDE = 1
TRACK_OFFSET = 0
GIF_FPS = 6
SCORE_THRESH = 0.5
MAX_AREA_FRAC = 0.6
MAX_FRAMES = None
DETECT_FRAMES = 1       # number of initial (tracked) frames to run detection on
SEED_NMS_IOU = 0.7      # merge seeds overlapping above this IoU (distinct object set)
OVERLAY_STYLE = "both"  # "filled" | "outline" | "both" (fill + bold border)
DRAW_LABELS = True      # draw "<id>:<tag>" near each object in the overlay


def load_models(device, hf_name=HF_NAME, dtype=DTYPE):
    """Load the concept detector (Sam3Video) and the pure tracker (Sam3TrackerVideo)."""
    vmodel = Sam3VideoModel.from_pretrained(hf_name).to(device, dtype=dtype).eval()
    vproc = Sam3VideoProcessor.from_pretrained(hf_name)
    tmodel = Sam3TrackerVideoModel.from_pretrained(hf_name).to(device, dtype=dtype).eval()
    tproc = Sam3TrackerVideoProcessor.from_pretrained(hf_name)
    return vmodel, vproc, tmodel, tproc


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def _nms_seeds(seeds, iou_thresh):
    """Greedy IoU NMS over seed masks (cross-tag) -> distinct object set."""
    order = sorted(seeds, key=lambda s: s["score"], reverse=True)
    kept = []
    for s in order:
        if all(_iou(s["mask"], k["mask"]) < iou_thresh for k in kept):
            kept.append(s)
    return kept


def process_video(video_path, tags, out_dir, label, vmodel, vproc, tmodel, tproc, device,
                  dtype=DTYPE, detect_frames=DETECT_FRAMES, seed_nms_iou=SEED_NMS_IOU,
                  stride=STRIDE, track_stride=TRACK_STRIDE, track_offset=TRACK_OFFSET,
                  gif_fps=GIF_FPS, mask_alpha=MASK_ALPHA, score_thresh=SCORE_THRESH,
                  max_area_frac=MAX_AREA_FRAC, max_frames=MAX_FRAMES, style=OVERLAY_STYLE,
                  draw_labels=DRAW_LABELS, vq3d_json=None):
    if not tags:
        print(f"[{label}] no tags to prompt with, skipping.")
        return
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    frames_dir = os.path.join(out_dir, "frames")
    masks_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlay")
    for d in (frames_dir, masks_dir, overlay_dir):
        os.makedirs(d, exist_ok=True)

    frames = _load_video_frames(video_path)
    if not frames:
        print(f"[{label}] could not read frames, skipping.")
        return
    total = len(frames)
    orig_idx = list(range(max(0, track_offset), total, max(1, track_stride)))
    if max_frames is not None:
        orig_idx = orig_idx[:max_frames]
    if not orig_idx:
        print(f"[{label}] no frames to track (offset {track_offset}).")
        return
    frames = [frames[i] for i in orig_idx]
    n = len(frames)
    H, W = frames[0].shape[:2]
    pil_seq = [Image.fromarray(f) for f in frames]
    det_n = min(max(1, detect_frames), n)
    print(f"[{label}] {n}/{total} tracked (offset {track_offset}, stride {track_stride}) "
          f"@ {W}x{H}; detect on first {det_n} frame(s), {len(tags)} tag(s).")

    # --- Phase 1: detection on the first det_n frames -----------------------------
    vsession = vproc.init_video_session(
        video=pil_seq[:det_n], inference_device=device, inference_state_device="cpu",
        processing_device="cpu", video_storage_device="cpu", dtype=dtype,
    )
    vproc.add_text_prompt(vsession, list(tags))
    # Detection runs over the window so SAM3 can stabilize the set; we take the LAST
    # detect frame's object set as the seeds (a single common conditioning frame -- the
    # tracker requires all objects to be seeded on one frame before a propagation pass).
    last = {"frame": -1, "ids": [], "masks": None, "scores": [], "tag": {}}
    with torch.no_grad():
        for out in vmodel.propagate_in_video_iterator(vsession, start_frame_idx=0,
                                                      max_frame_num_to_track=det_n - 1):
            res = vproc.postprocess_outputs(vsession, out)
            if out.frame_idx < last["frame"]:
                continue
            tagmap = {oid: tag for tag, oids in res["prompt_to_obj_ids"].items() for oid in oids}
            last = {"frame": out.frame_idx, "ids": res["object_ids"].tolist(),
                    "masks": res["masks"].cpu().numpy().astype(bool),
                    "scores": res["scores"].tolist(), "tag": tagmap}

    seed_frame = max(last["frame"], 0)
    seeds = []
    for k, oid in enumerate(last["ids"]):
        m = last["masks"][k]
        area = int(m.sum())
        if area == 0:
            continue
        seeds.append({"mask": m, "area": area, "score": float(last["scores"][k]),
                      "tag": last["tag"].get(oid, "?")})

    # filter weak / scene-spanning seeds, then NMS to a distinct set
    seeds = [s for s in seeds
             if s["score"] >= score_thresh
             and (max_area_frac is None or s["area"] / (H * W) <= max_area_frac)]
    seeds = _nms_seeds(seeds, seed_nms_iou)
    if not seeds:
        print(f"[{label}] no seed objects after filtering, nothing to track.")
        return
    print(f"[{label}] seeded {len(seeds)} distinct object(s) at frame {orig_idx[seed_frame]}: "
          f"{[(i, s['tag']) for i, s in enumerate(seeds)]}")

    # --- Phase 2: pure mask propagation over the whole clip -----------------------
    tsession = tproc.init_video_session(
        video=pil_seq, inference_device=device, inference_state_device="cpu",
        processing_device="cpu", video_storage_device="cpu", dtype=dtype,
    )
    # All objects seeded on one frame (batched), then propagate forward AND backward
    # from it so frames before the seed frame are covered too.
    tproc.add_inputs_to_inference_session(
        inference_session=tsession, frame_idx=seed_frame,
        obj_ids=list(range(len(seeds))),
        input_masks=[s["mask"].astype(np.uint8) for s in seeds],
    )

    obj_masks = {i: np.zeros((n, H, W), dtype=bool) for i in range(len(seeds))}

    def _collect(reverse):
        for out in tmodel.propagate_in_video_iterator(tsession, start_frame_idx=seed_frame,
                                                      reverse=reverse):
            res_masks = tproc.post_process_masks(
                [out.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
            for i, oid in enumerate(tsession.obj_ids):
                obj_masks[oid][out.frame_idx] = res_masks[i, 0].float().cpu().numpy() > 0.0

    with torch.no_grad():
        _collect(reverse=False)
        if seed_frame > 0:
            _collect(reverse=True)

    # --- Save (same layout as the concept pipeline) -------------------------------
    np.savez_compressed(os.path.join(out_dir, "masks_orig_space.npz"),
                        **{f"obj_{i}": obj_masks[i] for i in obj_masks})

    # Self-contained VQ3D object memory: per-object 2D trajectory (frame-indexed boxes /
    # areas / timestamps), temporal extent, static/moving flag, and an appearance crop --
    # everything a query needs without re-decoding the video. 3D boxes (WildDet3D) and
    # camera poses are attached later into the box_3d_cam / pose hooks in the trajectory.
    mem_root = os.path.join(out_dir, "object_memory")
    objects = {}
    for i in obj_masks:
        entry = {"tag": seeds[i]["tag"], "score": round(seeds[i]["score"], 4),
                 "seed_frame": int(orig_idx[seed_frame]),
                 "frames_present": int(obj_masks[i].reshape(n, -1).any(-1).sum()),
                 "color": _color(i)}
        traj = omem.object_trajectory(obj_masks[i], orig_idx, fps, score=seeds[i]["score"])
        if traj is not None:
            entry.update(traj)
        crop = omem.save_object_crop(mem_root, i, frames, obj_masks[i], orig_idx)
        if crop is not None:
            entry["crop"] = crop
        objects[str(i)] = entry

    header = omem.build_memory_header(
        label, video_path, fps, H, W,
        sampling={"track_offset": track_offset, "track_stride": track_stride,
                  "max_frames": max_frames, "detect_frames": det_n},
        vq3d_json_path=vq3d_json)
    with open(os.path.join(out_dir, "objects.json"), "w") as fh:
        json.dump({"label": label, "mode": "detect_then_track", "tags_prompted": list(tags),
                   "detect_frames": det_n, "seed_nms_iou": seed_nms_iou,
                   "track_stride": track_stride, "track_offset": track_offset,
                   "tracked_frame_indices": orig_idx, **header, "objects": objects}, fh, indent=2)

    gif_frames = []
    for t in range(0, n, stride):
        fid = orig_idx[t]
        Image.fromarray(frames[t]).save(os.path.join(frames_dir, f"frame_{fid:05d}.jpg"))
        items = []
        for i in obj_masks:
            m = obj_masks[i][t]
            if not m.any():
                continue
            items.append((m, _color(i), f"{i}:{seeds[i]['tag']}"))
            d = os.path.join(masks_dir, f"obj_{i}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray((m.astype(np.uint8) * 255)).save(os.path.join(d, f"frame_{fid:05d}.png"))
        overlay = render_objects(frames[t], items, alpha=mask_alpha, style=style,
                                 draw_labels=draw_labels)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"frame_{fid:05d}.png"))
        gif_frames.append(Image.fromarray(overlay))

    gif_path = os.path.join(out_dir, f"{label}_tracks.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(1000 / gif_fps), loop=0, optimize=False)
    print(f"[{label}] done -> {gif_path} ({len(seeds)} tracked object(s))")
