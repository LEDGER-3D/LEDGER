"""Stage 2 of the open-world SAM3 pipeline: detect + segment + track from text.

Concept-mode analogue of track_video_sam3.py. Instead of propagating a single
hand-authored seed mask with Sam3TrackerVideoModel, this feeds a list of
discovered noun-phrase tags (from discover_tags_ram.py) to Sam3VideoModel, which
detects every instance of every concept, assigns stable object IDs, segments, and
tracks them across the whole video in one pass -- including re-detecting objects
that enter mid-video.

Reuses _load_video_frames / _overlay_mask from track_video_sam3.py so the output
layout matches the existing seed-mask pipeline (frames/, masks/, overlay/, a GIF),
extended to multiple auto-discovered objects.

Requires the patched transformers fork that ships the SAM3 family:
    pip install -e <a transformers checkout that provides Sam3VideoModel>
"""
import colorsys
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import Sam3VideoModel, Sam3VideoProcessor

# Reuse the exact frame reader from the seed-mask pipeline.
from track_video_sam3 import _load_video_frames

# --- Defaults (override from launchers) -----------------------------------------
HF_NAME = "facebook/sam3"
DTYPE = torch.bfloat16
STRIDE = 1                  # output cadence (every STRIDE-th *tracked* frame is written)
TRACK_STRIDE = 1            # propagation subsample: track every TRACK_STRIDE-th source
                            # frame. >1 cuts time + GPU memory ~linearly on long videos.
TRACK_OFFSET = 0            # skip this many initial source frames before tracking begins
GIF_FPS = 6
MASK_ALPHA = 0.55
SCORE_THRESH = 0.5          # drop objects whose best detection score is below this
                            # (0.5 trims weak/duplicate re-detections; lower to recall more)
MAX_AREA_FRAC = 0.6         # drop objects whose typical mask covers > this frac of the
                            # frame (scene/place tags like "garage"/"room"); None disables
MAX_FRAMES = None           # cap frames tracked per video (None = all)
OVERLAY_STYLE = "both"      # "filled" | "outline" | "both" (fill + bold border)
DRAW_LABELS = True          # draw "<id>:<tag>" near each object in the overlay


def load_model(device, hf_name=HF_NAME, dtype=DTYPE):
    model = Sam3VideoModel.from_pretrained(hf_name).to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained(hf_name)
    model.eval()
    return model, processor


def _color(obj_id):
    """Distinct RGB per object via golden-ratio hue spacing (no collisions for any
    realistic object count, unlike a small fixed palette)."""
    h = (obj_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.78, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_objects(frame_rgb, items, alpha=MASK_ALPHA, style=OVERLAY_STYLE, draw_labels=DRAW_LABELS):
    """Composite per-object masks onto a frame.

    items: list of (mask_bool, color, label). Masks are drawn LARGEST-FIRST so small
    objects end up on top and stay visible instead of being painted over by big ones.
    style:
      "filled"  -> alpha-blended fill + thin border
      "outline" -> colored contour only (no fill); overlapping objects stay visible
      "both"    -> filled fill + a bold border (default)
    """
    do_fill = style in ("filled", "both")
    border = 2 if style in ("outline", "both") else 1
    out = frame_rgb.copy()
    for mask, color, label in sorted(items, key=lambda it: int(it[0].sum()), reverse=True):
        sel = np.asarray(mask, dtype=bool)
        if not sel.any():
            continue
        cnts, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if do_fill:
            tint = np.empty_like(out)
            tint[:] = color
            out[sel] = (alpha * tint[sel] + (1 - alpha) * out[sel]).astype(np.uint8)
        cv2.drawContours(out, cnts, -1, color, border, lineType=cv2.LINE_AA)
        if draw_labels and label and cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def process_video(video_path, tags, out_dir, label, model, processor, device,
                  dtype=DTYPE, stride=STRIDE, track_stride=TRACK_STRIDE, track_offset=TRACK_OFFSET,
                  gif_fps=GIF_FPS, mask_alpha=MASK_ALPHA, score_thresh=SCORE_THRESH,
                  max_area_frac=MAX_AREA_FRAC, max_frames=MAX_FRAMES, style=OVERLAY_STYLE,
                  draw_labels=DRAW_LABELS):
    if not tags:
        print(f"[{label}] no tags to prompt with, skipping.")
        return
    frames_dir = os.path.join(out_dir, "frames")
    masks_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlay")
    for d in (frames_dir, masks_dir, overlay_dir):
        os.makedirs(d, exist_ok=True)

    frames = _load_video_frames(video_path)
    if not frames:
        print(f"[{label}] could not read frames, skipping.")
        return
    # Subsample frames for propagation: skip the first `track_offset`, then take every
    # `track_stride`-th frame. The tracked subset maps back to original source indices.
    total = len(frames)
    orig_idx = list(range(max(0, track_offset), total, max(1, track_stride)))
    if not orig_idx:
        print(f"[{label}] track_offset {track_offset} >= {total} frames, nothing to track.")
        return
    frames = [frames[i] for i in orig_idx]
    n = len(frames)
    H, W = frames[0].shape[:2]
    print(f"[{label}] video {n}/{total} tracked (offset {track_offset}, stride {track_stride}) "
          f"@ {W}x{H}, prompting {len(tags)} tag(s): {tags}")

    pil_seq = [Image.fromarray(f) for f in frames]
    session = processor.init_video_session(
        video=pil_seq,
        inference_device=device,
        inference_state_device="cpu",   # offload growing per-frame state off the GPU
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    processor.add_text_prompt(session, list(tags))

    # obj_id -> (n, H, W) bool track; obj_id -> best score; obj_id -> tag text.
    obj_masks = {}
    obj_score = {}
    obj_tag = {}
    with torch.no_grad():
        for out in model.propagate_in_video_iterator(
            session, start_frame_idx=0, max_frame_num_to_track=max_frames
        ):
            res = processor.postprocess_outputs(session, out)
            fi = out.frame_idx
            ids = res["object_ids"].tolist()
            if not ids:
                continue
            masks = res["masks"].cpu().numpy().astype(bool)   # (N, H, W)
            scores = res["scores"].tolist()
            for k, oid in enumerate(ids):
                if oid not in obj_masks:
                    obj_masks[oid] = np.zeros((n, H, W), dtype=bool)
                obj_masks[oid][fi] = masks[k]
                obj_score[oid] = max(obj_score.get(oid, 0.0), float(scores[k]))
            for tag, oids in res["prompt_to_obj_ids"].items():
                for oid in oids:
                    obj_tag[oid] = tag

    # Drop low-confidence / empty / scene-spanning objects. A scene/place tag (e.g.
    # "garage") tends to segment most of the frame, so cap by typical mask coverage.
    def _too_big(m):
        if max_area_frac is None:
            return False
        present = m.reshape(n, -1).sum(-1)
        present = present[present > 0]
        return present.size > 0 and float(np.median(present)) / (H * W) > max_area_frac

    kept = [oid for oid in sorted(obj_masks)
            if obj_score.get(oid, 0.0) >= score_thresh
            and obj_masks[oid].any()
            and not _too_big(obj_masks[oid])]
    if not kept:
        print(f"[{label}] no objects detected above threshold {score_thresh}, nothing to write.")
        return

    # Persist raw tracks (per object) + metadata.
    np.savez_compressed(
        os.path.join(out_dir, "masks_orig_space.npz"),
        **{f"obj_{oid}": obj_masks[oid] for oid in kept},
    )
    objects = {
        str(oid): {
            "tag": obj_tag.get(oid, "?"),
            "score": round(obj_score.get(oid, 0.0), 4),
            "frames_present": int(obj_masks[oid].reshape(n, -1).any(-1).sum()),
            "color": _color(oid),
        }
        for oid in kept
    }
    with open(os.path.join(out_dir, "objects.json"), "w") as fh:
        json.dump({"label": label, "tags_prompted": list(tags), "track_stride": track_stride,
                   "track_offset": track_offset, "tracked_frame_indices": orig_idx,
                   "objects": objects}, fh, indent=2)

    # Write per-frame frames, combined overlay (all objects), per-object masks, GIF.
    # File names use the ORIGINAL source frame index for traceability.
    gif_frames = []
    for t in range(0, n, stride):
        fid = orig_idx[t]
        Image.fromarray(frames[t]).save(os.path.join(frames_dir, f"frame_{fid:05d}.jpg"))
        items = []
        for oid in kept:
            m = obj_masks[oid][t]
            if not m.any():
                continue
            items.append((m, _color(oid), f"{oid}:{obj_tag.get(oid, '?')}"))
            obj_mask_dir = os.path.join(masks_dir, f"obj_{oid}")
            os.makedirs(obj_mask_dir, exist_ok=True)
            Image.fromarray((m.astype(np.uint8) * 255)).save(
                os.path.join(obj_mask_dir, f"frame_{fid:05d}.png"))
        overlay = render_objects(frames[t], items, alpha=mask_alpha, style=style,
                                 draw_labels=draw_labels)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"frame_{fid:05d}.png"))
        gif_frames.append(Image.fromarray(overlay))

    gif_path = os.path.join(out_dir, f"{label}_tracks.gif")
    gif_frames[0].save(
        gif_path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / gif_fps), loop=0, optimize=False,
    )
    print(f"[{label}] done -> {gif_path} ({len(kept)} object(s): "
          f"{[(oid, objects[str(oid)]['tag']) for oid in kept]})")
