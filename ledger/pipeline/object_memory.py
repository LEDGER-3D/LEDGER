"""Self-contained per-object memory for VQ3D-style querying.

Turns a set of tracked masks (from the SAM3 detect_track / concept pipelines) into a
STANDALONE scene-graph-style object memory that can answer temporal 3D queries
("where was object X last seen before frame N?") WITHOUT re-decoding the video --
everything a query needs is materialized here during the single exploration pass.

What this module fills in (2D + temporal + identity, computable from exploration):
  - per-object trajectory: frame-indexed 2D boxes / areas / scores / timestamps,
    stored at the exact (sparse, offset+stride) observed frames so a query resolves by
    nearest-earlier observed frame;
  - temporal extent (first/last seen, observed-frame set) and a static/moving flag;
  - a representative appearance crop path (so query-object matching never needs frames);
  - scene linkage: video_uid, fps, dims, intrinsics used, and -- if a VQ3D annotation
    file is passed -- scan_uid + the object_titles VQ3D will ask about for this video.

What it deliberately leaves as HOOKS (needs data not available at exploration time):
  - box_3d per frame            -> filled downstream by WildDet3D (lifted3d.json), CAMERA frame;
  - pose_cam2ref / world coords -> needs the camera-pose stage; see POSE_HOOK below.
"""
import json
import os

import numpy as np
from PIL import Image

MEMORY_SCHEMA_VERSION = "vq3d-1.0"
STATIC_DISP_FRAC = 0.15   # center wander < this * mean box-diagonal -> treated as static


def default_intrinsics_K(h, w):
    """Placeholder pinhole K (focal=max(H,W), centered) -- MUST match whatever WildDet3D
    is given so its box_3d and these 2D boxes share a camera model. Replace with the real
    (448-scaled) Ego4D intrinsics when available."""
    f = float(max(h, w))
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def mask_to_bbox(mask):
    """Boolean (H,W) mask -> tight [x1,y1,x2,y2] (floats), or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def object_trajectory(mask_stack, orig_idx, fps, score=None, clip_offsets=None):
    """Build the self-contained trajectory + temporal/motion summary for one object.

    mask_stack : (n_tracked, H, W) bool
    orig_idx   : list[int] absolute source-video frame index per tracked slot
    fps        : video fps (for timestamps)
    score      : per-object detection score (dt mode has no per-frame score -> constant/None)
    clip_offsets: optional list of (clip_uid, video_start_frame) to also emit clip-local frames
    Returns dict with trajectory + first/last_seen + observed_frames + motion/is_static.
    """
    n = mask_stack.shape[0]
    traj, centers, diags, observed = [], [], [], []
    for t in range(n):
        bb = mask_to_bbox(mask_stack[t])
        if bb is None:
            continue
        fv = int(orig_idx[t])
        area = float(mask_stack[t].sum())
        cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
        entry = {
            "frame_video": fv,
            "time_s": round(fv / fps, 4) if fps else None,
            "bbox_xyxy": [round(v, 2) for v in bb],
            "center_xy": [round(cx, 2), round(cy, 2)],
            "area_px": area,
            "score": round(float(score), 4) if score is not None else None,
            # HOOKS filled later (kept present so the schema is stable):
            "box_3d_cam": None,     # <- WildDet3D lifted3d.json
            "pose_cam2ref": None,   # <- POSE_HOOK: camera->reference/scan transform for this frame
            "position_ref": None,   # <- box_3d_cam transported by pose_cam2ref
        }
        if clip_offsets:
            entry["frame_clip"] = {cu: fv - s for cu, s in clip_offsets if s is not None and fv >= s}
        traj.append(entry)
        centers.append([cx, cy]); diags.append(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
        observed.append(fv)

    if not traj:
        return None
    centers = np.array(centers)
    center_std = centers.std(0)
    mean_diag = float(np.mean(diags)) or 1.0
    disp = float(np.hypot(*(centers.max(0) - centers.min(0)))) if len(centers) > 1 else 0.0
    is_static = bool(disp <= STATIC_DISP_FRAC * mean_diag)
    return {
        "trajectory": traj,
        "first_seen": observed[0],
        "last_seen": observed[-1],
        "observed_frames": observed,
        "n_observed": len(observed),
        "motion": {"center_std_px": [round(float(center_std[0]), 2), round(float(center_std[1]), 2)],
                   "center_disp_px": round(disp, 2), "mean_box_diag_px": round(mean_diag, 2)},
        "is_static": is_static,
    }


def save_object_crop(mem_root, oid, frames, mask_stack, orig_idx, pad=0.12):
    """Save one representative appearance crop (largest-area observed frame) so query-object
    matching can be done from memory alone. Returns the relative crop path or None."""
    areas = mask_stack.reshape(mask_stack.shape[0], -1).sum(-1)
    if areas.max() == 0:
        return None
    t = int(np.argmax(areas))
    bb = mask_to_bbox(mask_stack[t])
    H, W = frames[t].shape[:2]
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    x1 = max(0, int(bb[0] - pad * bw)); y1 = max(0, int(bb[1] - pad * bh))
    x2 = min(W, int(bb[2] + pad * bw)); y2 = min(H, int(bb[3] + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    odir = os.path.join(mem_root, f"obj_{oid}")
    os.makedirs(odir, exist_ok=True)
    Image.fromarray(frames[t][y1:y2, x1:x2]).save(os.path.join(odir, "thumbnail.png"))
    return os.path.join("object_memory", f"obj_{oid}", "thumbnail.png")


def load_vq3d_scene(vq3d_json_path, video_uid):
    """Best-effort scene linkage from a VQ3D annotation file: returns
    {scan_uid, clips:[{clip_uid, object_titles:[...]}]} for this video_uid, or None.
    (Clip video-frame offsets/poses are NOT in this file -> left as hooks.)"""
    if not vq3d_json_path or not os.path.isfile(vq3d_json_path):
        return None
    try:
        data = json.load(open(vq3d_json_path))
    except Exception:
        return None
    for v in data.get("videos", []):
        if v.get("video_uid") != video_uid:
            continue
        clips = []
        for c in v.get("clips", []):
            titles = []
            for a in c.get("annotations", []):
                for qs in (a.get("query_sets") or {}).values():
                    if qs.get("object_title"):
                        titles.append(qs["object_title"])
            clips.append({"clip_uid": c.get("clip_uid"),
                          "object_titles": sorted(set(titles))})
        return {"scan_uid": v.get("scan_uid"), "clips": clips}
    return None


def build_memory_header(label, video_path, fps, H, W, sampling, vq3d_json_path=None,
                        intrinsics_K=None, intrinsics_source=None):
    """Top-level self-describing block for objects.json (everything a query needs about
    the scene, so nothing is re-read from the video).

    intrinsics_K / intrinsics_source: pass the REAL (e.g. VQ3D letterboxed-448) camera
    matrix + a provenance string to bake them in. When omitted, falls back to the
    placeholder pinhole K (focal=max(H,W), centered) -- which is NOT metrically valid."""
    video_uid = os.path.splitext(os.path.basename(video_path))[0]
    if intrinsics_K is not None:
        K = intrinsics_K
        K_src = intrinsics_source or "provided"
    else:
        K = default_intrinsics_K(H, W)
        K_src = "placeholder(focal=max(H,W),centered) -- replace with real 448 K"
    header = {
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "video": {"video_uid": video_uid, "path": os.path.realpath(video_path),
                  "fps": round(float(fps), 3) if fps else None, "width": W, "height": H},
        "intrinsics_K": K,
        "intrinsics_source": K_src,
        "sampling": sampling,
        # POSE_HOOK: per-frame camera->reference poses go here once the pose stage runs.
        "poses_available": False,
    }
    scene = load_vq3d_scene(vq3d_json_path, video_uid)
    if scene is not None:
        header["vq3d"] = scene
    return header
