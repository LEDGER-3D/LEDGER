"""Detect+track FUSION variant of the open-world SAM3 pipeline (VQ3D-oriented).

Motivation
----------
`track_video_sam3_dt.py` seeds objects from a SINGLE frame (the last detect frame),
so any object not present in that one frame is never tracked. For VQ3D that misses
almost every queried object (they appear only in short windows scattered through the
clip). `track_video_sam3_concept.py` re-detects every frame but fragments IDs.

This module fuses the two, as requested:

  Phase 1 (detect on EVERY tracked frame) -- run Sam3VideoModel on every frame in the
      (explicit) tracked-frame set and collect per-frame detections. With a sparse grid
      the tracked set is small, so "detect every frame" is cheap.
  Phase 2 (cluster -> distinct seeds) -- greedy containment clustering per tag collapses
      repeated detections of the same object into a small set of seeds, each anchored at
      its highest-score frame (capped at MAX_SEEDS).
  Phase 3 (track) -- seed ALL objects into ONE Sam3TrackerVideoModel session (each at its
      own frame), then propagate forward+backward from every distinct seed frame so each
      object is covered over the whole clip. Stable IDs, no detector drift.
  Phase 4 (merge) -- fuse tracks that are the same physical object using the overlap
      heuristic: containment (intersection / smaller-area). Keep-rule is CONTINUITY-FIRST
      (the longer / higher-scoring track wins), with SMALLER-MASK as the per-frame
      tiebreaker. This is the detection<->track reconciliation applied at track level.

Everything downstream (object_memory objects.json, masks, overlay, gif) matches the dt
pipeline so the WildDet3D lift + VQ3D eval consume it unchanged.
"""
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

import object_memory as omem
from track_video_sam3_concept import _color, render_objects, MASK_ALPHA
from track_video_sam3 import _load_video_frames
# Reuse the exact model loaders + primitives proven in the dt pipeline.
from track_video_sam3_dt import load_models, _iou, DTYPE, HF_NAME

SCORE_THRESH = 0.5
SEED_NMS_IOU = 0.7      # within-frame dedup of overlapping concepts (bike/wheel/tire)
CLUSTER_CONTAIN = 0.5   # cross-frame: attach a detection to a cluster if containment >= this
MERGE_CONTAIN = 0.6     # track<->track: merge if median containment over co-present frames >= this
MAX_AREA_FRAC = 0.6     # drop scene-spanning masks (a place tag segmenting the whole frame)
MAX_SEEDS = 40          # safety cap on number of tracks before merge (bounds time+memory)
OVERLAY_STYLE = "both"
DRAW_LABELS = True


def _contain(a, b):
    """Fraction of the SMALLER mask covered by the other = intersection / min(area)."""
    aa, ab = int(a.sum()), int(b.sum())
    if aa == 0 or ab == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return inter / float(min(aa, ab))


def _nms_frame(dets, iou_thresh):
    """Greedy IoU NMS within one frame -> distinct detections (highest score first)."""
    kept = []
    for d in sorted(dets, key=lambda s: s["score"], reverse=True):
        if all(_iou(d["mask"], k["mask"]) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def _detect_all_frames(pil_seq, tags, vmodel, vproc, device, dtype,
                       score_thresh, seed_nms_iou, max_area_frac, H, W):
    """Phase 1: Sam3VideoModel detection on every frame of pil_seq.
    Returns det_by_t: list (len n) of per-frame detection lists {mask,area,score,tag}."""
    n = len(pil_seq)
    vsession = vproc.init_video_session(
        video=pil_seq, inference_device=device, inference_state_device="cpu",
        processing_device="cpu", video_storage_device="cpu", dtype=dtype,
    )
    vproc.add_text_prompt(vsession, list(tags))
    det_by_t = [[] for _ in range(n)]
    with torch.no_grad():
        for out in vmodel.propagate_in_video_iterator(vsession, start_frame_idx=0):
            res = vproc.postprocess_outputs(vsession, out)
            t = out.frame_idx
            if t < 0 or t >= n:
                continue
            ids = res["object_ids"].tolist()
            if not ids:
                continue
            masks = res["masks"].cpu().numpy().astype(bool)
            scores = res["scores"].tolist()
            tagmap = {oid: tag for tag, oids in res["prompt_to_obj_ids"].items() for oid in oids}
            dets = []
            for k, oid in enumerate(ids):
                m = masks[k]
                area = int(m.sum())
                s = float(scores[k])
                if s < score_thresh or area == 0:
                    continue
                if max_area_frac is not None and area / (H * W) > max_area_frac:
                    continue
                dets.append({"mask": m, "area": area, "score": s, "tag": tagmap.get(oid, "?")})
            det_by_t[t] = _nms_frame(dets, seed_nms_iou)
    return det_by_t


def _cluster_seeds(det_by_t, contain_thresh, max_seeds):
    """Phase 2: greedy per-tag containment clustering across frames -> distinct seeds.

    A detection joins an existing cluster of the SAME tag if it overlaps that cluster's
    running representative mask (containment >= thresh); otherwise it starts a new cluster.
    Each cluster is anchored (seeded) at its highest-score detection frame.
    Returns list of seeds: {tag, t (seed local idx), mask, score, support (#dets)}.
    """
    clusters = []  # {tag, rep_mask, best_t, best_mask, best_score, support}
    for t, dets in enumerate(det_by_t):
        for d in dets:
            cand = [c for c in clusters if c["tag"] == d["tag"]]
            best, best_ov = None, 0.0
            for c in cand:
                ov = _contain(d["mask"], c["rep_mask"])
                if ov >= contain_thresh and ov > best_ov:
                    best, best_ov = c, ov
            if best is None:
                clusters.append({"tag": d["tag"], "rep_mask": d["mask"],
                                 "best_t": t, "best_mask": d["mask"],
                                 "best_score": d["score"], "support": 1})
            else:
                best["rep_mask"] = d["mask"]          # follow the object (last-seen mask)
                best["support"] += 1
                if d["score"] > best["best_score"]:
                    best["best_t"], best["best_mask"], best["best_score"] = t, d["mask"], d["score"]
    seeds = [{"tag": c["tag"], "t": c["best_t"], "mask": c["best_mask"],
              "score": c["best_score"], "support": c["support"]} for c in clusters]
    seeds.sort(key=lambda s: (s["support"], s["score"]), reverse=True)
    return seeds[:max_seeds]


def _track_seeds(pil_seq, seeds, tmodel, tproc, device, dtype, H, W, window=None):
    """Phase 3: track seeds with the pure mask propagator.

    SAM3's tracker requires every object in a session to share ONE initial conditioning
    frame (it errors otherwise), so we run ONE session per distinct seed frame -- objects
    that appear at the same frame propagate together, exactly like the proven dt pipeline.
    Each group is propagated forward to the end and backward to the start (or only within
    +/-`window` local frames of the seed frame when `window` is set, to bound cost).
    Returns obj_masks {global_seed_index: (n,H,W) bool}."""
    from collections import defaultdict
    n = len(pil_seq)
    groups = defaultdict(list)
    for i, s in enumerate(seeds):
        groups[int(s["t"])].append(i)
    obj_masks = {i: np.zeros((n, H, W), dtype=bool) for i in range(len(seeds))}

    with torch.no_grad():
        for sf, members in sorted(groups.items()):
            tsession = tproc.init_video_session(
                video=pil_seq, inference_device=device, inference_state_device="cpu",
        processing_device="cpu", video_storage_device="cpu", dtype=dtype,
            )
            tproc.add_inputs_to_inference_session(
                inference_session=tsession, frame_idx=sf, obj_ids=list(members),
                input_masks=[seeds[i]["mask"].astype(np.uint8) for i in members],
            )
            lo = 0 if window is None else max(0, sf - window)
            hi = n - 1 if window is None else min(n - 1, sf + window)

            def _collect(reverse):
                # propagate_in_video_iterator yields frames monotonically from sf, so once
                # we pass the window edge we can STOP (break) -- this bounds the model
                # compute to ~2*window frames per session instead of the whole clip.
                for out in tmodel.propagate_in_video_iterator(
                        tsession, start_frame_idx=sf, reverse=reverse):
                    fi = out.frame_idx
                    if reverse and fi < lo:
                        break
                    if (not reverse) and fi > hi:
                        break
                    res_masks = tproc.post_process_masks(
                        [out.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
                    for j, oid in enumerate(tsession.obj_ids):
                        m = res_masks[j, 0].float().cpu().numpy() > 0.0
                        if m.any():
                            obj_masks[oid][fi] |= m

            _collect(reverse=False)
            if sf > 0:
                _collect(reverse=True)
    return obj_masks


def _merge_tracks(obj_masks, seeds, contain_thresh):
    """Phase 4: fuse tracks that are the same physical object.

    Same-object test: over frames where BOTH tracks are present, median containment
    (intersection / smaller-area) >= contain_thresh. Keep-rule = CONTINUITY-FIRST: the
    track present on more frames (tie -> higher score) is kept; the other is absorbed by
    OR-ing its masks in. Per-frame, where both are present we keep the SMALLER mask
    (tiebreak) to avoid background-bleed inflation. Returns (merged_masks, merged_seeds).
    """
    ids = sorted(obj_masks.keys())
    n = next(iter(obj_masks.values())).shape[0]
    present = {i: obj_masks[i].reshape(n, -1).any(-1) for i in ids}
    count = {i: int(present[i].sum()) for i in ids}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = sorted(ids, key=lambda i: (count[i], seeds[i]["score"]), reverse=True)
    for a_i in range(len(order)):
        for b_i in range(a_i + 1, len(order)):
            a, b = order[a_i], order[b_i]
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            co = present[ra] & present[rb]
            if co.sum() == 0:
                continue
            cons = [_contain(obj_masks[ra][t], obj_masks[rb][t]) for t in np.where(co)[0]]
            if float(np.median(cons)) >= contain_thresh:
                # absorb rb into ra (ra is earlier in `order` => continuity-first winner)
                for t in range(n):
                    mb = obj_masks[rb][t]
                    if not mb.any():
                        continue
                    ma = obj_masks[ra][t]
                    if not ma.any():
                        obj_masks[ra][t] = mb
                    else:
                        # both present -> keep smaller-area mask (tiebreak)
                        obj_masks[ra][t] = mb if int(mb.sum()) < int(ma.sum()) else ma
                parent[rb] = ra
                present[ra] = obj_masks[ra].reshape(n, -1).any(-1)
                count[ra] = int(present[ra].sum())

    roots = [i for i in ids if find(i) == i]
    merged_masks, merged_seeds = {}, []
    for new_i, r in enumerate(sorted(roots, key=lambda i: (count[i], seeds[i]["score"]),
                                     reverse=True)):
        merged_masks[new_i] = obj_masks[r]
        merged_seeds.append(seeds[r])
    return merged_masks, merged_seeds


def process_video_fuse(video_path, tags, out_dir, label, vmodel, vproc, tmodel, tproc,
                       device, dtype=DTYPE, frame_indices=None, track_offset=0,
                       track_stride=1, max_frames=None, score_thresh=SCORE_THRESH,
                       seed_nms_iou=SEED_NMS_IOU, cluster_contain=CLUSTER_CONTAIN,
                       merge_contain=MERGE_CONTAIN, max_area_frac=MAX_AREA_FRAC,
                       max_seeds=MAX_SEEDS, track_window=8, gif_fps=6, mask_alpha=MASK_ALPHA,
                       style=OVERLAY_STYLE, draw_labels=DRAW_LABELS, vq3d_json=None,
                       intrinsics_K=None, intrinsics_source=None, stride=1):
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

    all_frames = _load_video_frames(video_path)
    if not all_frames:
        print(f"[{label}] could not read frames, skipping.")
        return
    total = len(all_frames)
    if frame_indices is not None:
        orig_idx = sorted({int(i) for i in frame_indices if 0 <= int(i) < total})
    else:
        orig_idx = list(range(max(0, track_offset), total, max(1, track_stride)))
        if max_frames is not None:
            orig_idx = orig_idx[:max_frames]
    if not orig_idx:
        print(f"[{label}] no frames to track.")
        return
    frames = [all_frames[i] for i in orig_idx]
    n = len(frames)
    H, W = frames[0].shape[:2]
    pil_seq = [Image.fromarray(f) for f in frames]
    print(f"[{label}] {n} tracked frame(s) @ {W}x{H}; {len(tags)} tag(s); detect on ALL frames.")

    det_by_t = _detect_all_frames(pil_seq, tags, vmodel, vproc, device, dtype,
                                  score_thresh, seed_nms_iou, max_area_frac, H, W)
    n_det = sum(len(d) for d in det_by_t)
    seeds = _cluster_seeds(det_by_t, cluster_contain, max_seeds)
    print(f"[{label}] {n_det} detection(s) -> {len(seeds)} seed cluster(s): "
          f"{[(s['tag'], orig_idx[s['t']], round(s['score'], 2)) for s in seeds]}")
    if not seeds:
        print(f"[{label}] no seeds after clustering, nothing to track.")
        return

    obj_masks = _track_seeds(pil_seq, seeds, tmodel, tproc, device, dtype, H, W,
                             window=track_window)
    obj_masks, seeds = _merge_tracks(obj_masks, seeds, merge_contain)
    # drop tracks that ended up empty after merge
    keep = [i for i in sorted(obj_masks) if obj_masks[i].any()]
    obj_masks = {new: obj_masks[old] for new, old in enumerate(keep)}
    seeds = [seeds[old] for old in keep]
    print(f"[{label}] {len(seeds)} object(s) after merge: "
          f"{[(s['tag']) for s in seeds]}")

    np.savez_compressed(os.path.join(out_dir, "masks_orig_space.npz"),
                        **{f"obj_{i}": obj_masks[i] for i in obj_masks})

    mem_root = os.path.join(out_dir, "object_memory")
    objects = {}
    for i in obj_masks:
        entry = {"tag": seeds[i]["tag"], "score": round(float(seeds[i]["score"]), 4),
                 "seed_frame": int(orig_idx[seeds[i]["t"]]),
                 "support": int(seeds[i].get("support", 1)),
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
                  "max_frames": max_frames, "n_tracked_frames": n, "detect_every_frame": True},
        vq3d_json_path=vq3d_json, intrinsics_K=intrinsics_K, intrinsics_source=intrinsics_source)
    with open(os.path.join(out_dir, "objects.json"), "w") as fh:
        json.dump({"label": label, "mode": "detect_track_fuse", "tags_prompted": list(tags),
                   "cluster_contain": cluster_contain, "merge_contain": merge_contain,
                   "seed_nms_iou": seed_nms_iou, "score_thresh": score_thresh,
                   "track_offset": track_offset, "track_stride": track_stride,
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

    if gif_frames:
        gif_path = os.path.join(out_dir, f"{label}_tracks.gif")
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / gif_fps), loop=0, optimize=False)
        print(f"[{label}] done -> {gif_path} ({len(seeds)} object(s))")
