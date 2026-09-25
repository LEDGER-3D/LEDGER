"""UCS-Bench adapter for the VQ3D object-memory pipeline (selected by --dataset ucs).

Everything UCS-Bench-specific lives here; the shared stages consume the files this module emits.
Ego4D and HD-EPIC are untouched -- every shared script keeps its old default and only branches on
--dataset ucs.

WHAT UCS-BENCH SHIPS, AND WHAT IT DOES NOT
------------------------------------------
UCS-Bench (huggingface cocowy1/UCS-Bench) is 540 mp4 videos + 532 MCQ files. It ships NO camera
poses, NO intrinsics and NO 3D ground truth. Its videos are re-encodes pooled from six sources
(Ego4D, EPIC-Kitchens, EgoLife/Aria, ScanNet renders, and two numeric-id sets). That has three
consequences the rest of this file exists to handle:

  1. POSES must be estimated.        -> FastVGGT, chunked with overlap + Sim3 chaining (`poses`).
  2. INTRINSICS must be estimated.   -> VGGT's own camera head, mapped back to native pixels.
  3. SCALE is not metric.            -> VGGT is up to scale, WildDet3D's lift IS metric, and
                                        world = inv(pose) @ cam mixes the two. One scalar per video
                                        makes them consistent; `scale` solves it in closed form from
                                        the lift's own cam_3d (see solve_metric_scale).

FRAME-INDEX CONVENTION (the #1 source of silent corruption)
-------------------------------------------------------------------------------------------------
Every one of the 540 videos is re-encoded at exactly 10.00 fps (verified: r_frame_rate == 10/1 on a
40-video sample covering all six sources and all six resolutions). So, unlike Ego4D/HD-EPIC where a
30 fps video was subsampled to a 5 fps clip grid (CLIP_RATIO 6), there is nothing to subsample:

    CLIP_RATIO = 1     ->   video_frame == clip_frame,  offset 0
    time_s = video_frame / 10.0
    grid_stride = 40 clip-frames = 4.0 s  (the same 4 s sampling density as VQ3D and HD-EPIC)

Grid JPEGs are still named <clip_frame:06d>.jpg, so the shared stages are unchanged. Because
CLIP_RATIO is 1 the mapping is the identity and cannot be off by one -- deliberately the simplest
convention available.

Note that `fv = 6*cf + off` and `time_s = fv/30.0` were HARDCODED in build_yolo_detect.py /
build_mem2_detect.py. They now read `clip_ratio` and `fps` from the grid spec this module writes,
defaulting to 6 / 30.0 when absent, so Ego4D and HD-EPIC behave exactly as before.

GATE 1 (the projection test) CANNOT BE RUN ON UCS-BENCH.
--------------------------------------------------------
It needs GT 3D points and GT 2D boxes; UCS-Bench has neither. `check` runs the strongest available
substitute instead -- a pose/lift SELF-consistency test (see cmd_check). Read its output as
"the geometry is internally coherent", never as "the geometry is correct".
"""
import os, sys, json, glob, re, subprocess
import numpy as np

import ledger_paths as LP
M    = LP.WORK
UCS  = LP.UCS
QADIR = UCS + "/QAs"
FASTVGGT = LP.FASTVGGT

FPS = 10.0
CLIP_RATIO = 1            # video frame = CLIP_RATIO * clip frame  (10 fps video, 10 fps clip grid)

# ---------------------------------------------------------------- pipeline config
CONFIG = {
    "grid_stride":     40,    # CLIP frames; 10 fps video so 40 = 4.0 s, matching VQ3D / HD-EPIC
    "max_frames":      1200,  # longest video is 5863 s -> 1466 grid frames; 1200 thins only those
    # Floor on grid frames for SHORT videos, the mirror of max_frames at the other end. 0 = off,
    # which is what the first pilot ran with; set it deliberately, and never mid-experiment.
    #
    # Why it exists: a fixed 4 s stride spans a 240x range of memory richness here, because
    # UCS-Bench videos run 20 s to 98 min. A 20 s ScanNet clip yields FIVE grid frames, and that
    # starves everything downstream -- tagging sees 5 images, and only ~20% of trajectory clusters
    # reach the 3 observations triangulation needs. Measured on scene0720: clusters with >=3 rays
    # have 0.65 m camera spread and ~11 deg effective ray spread, which is perfectly good geometry;
    # they fail the triangulation conditioning guard purely on RAY COUNT (cond = sum_i sin^2(theta_i),
    # so 3 rays at 13.7 deg give 0.168 against a 0.25 threshold, while 5 rays clear it).
    # So ScanNet's weakness is a grid-density choice, not a property of the source. 60 of 540 videos
    # are ScanNet, whose median is ~5 grid frames, so this is systematic.
    "min_frames":      0,
    "per_frame_lift":  12,    # YOLO-World emits <=12 dets/frame (--topk 12)
    "min_scene_budget": 400,
    "min_per_tag":      25,
    "seg_thresh_m":     0.3,
    "merge_max_dist":   1.0,
    # VGGT pose estimation
    "vggt_chunk":       96,   # grid frames per FastVGGT forward
    "vggt_overlap":     16,   # frames shared with the previous chunk, for the chaining Sim3
    "vggt_target":     518,   # square preprocess size (keeps FastVGGT's token-merge assertion valid)

    # ---- v3 additions. Every default below reproduces the v2 pipeline; each is switched on
    # ---- explicitly (driver env vars / agent flags), never implicitly.
    # Pose source. "fastvggt" = v2 (up to scale, needs the scale stage). "da3" = Depth Anything 3
    # nested model, whose poses and depth are already metric (any-view model for pose + monocular
    # metric model for scale), so there is nothing for the scale stage to solve.
    "pose_backend":    "fastvggt",
    # used when the driver runs with POSE_BACKEND=auto: pick per video from its pixels
    "pinhole_pose_backend": "da3",
    "fisheye_pose_backend": "fastvggt",
    # DA3 quality gate. Chunked DA3 is excellent on some videos (P04_119: chunk scales 0.96-1.06, seams
    # <= 5 cm) and badly wrong on others (D2-P3: scales 1.32-1.58, a 7.1 m seam, a 14 m jump in 4 s),
    # because the nested model's metric scale depends on which frames share a batch. A video keeps its
    # DA3 poses only if its chunks agree; otherwise it falls back and the reason is recorded.
    "da3_gate_max_seam_rmse_m": 0.5,
    "da3_gate_scale_min":       0.8,
    "da3_gate_scale_max":       1.25,
    "da3_gate_max_step_m":      6.0,   # per grid step (4 s); a fast walk is ~5.6 m
    "da3_gate_fallback":        "fastvggt",
    "da3_model":       "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "da3_chunk":       48,    # grid frames per DA3 forward (24 GB cards)
    "da3_overlap":     12,    # frames shared with the previous chunk, for the chaining Sim3
    "da3_process_res": 504,   # DA3's own default input size
    # Movement segmentation. "threshold" = v2 (new segment whenever a point is > thresh from the
    # running centroid). "persist": a move counts only if the
    # object STAYS at the new place, so per-frame lift jitter stops being recorded as movement.
    "seg_mode":        "threshold",
    "seg_thresh_unscaled": 0.3,   # SCENE UNITS, used only for a video whose poses are not metric
    "seg_persist_s":   8.0,       # seconds a move must be sustained (k=3 observations at 4 s)
    # Object-object relations, from objects seen in the SAME frame (camera coordinates, metric
    # because WildDet3D's cam_3d is metric regardless of pose scale).
    "rel_lat_m":       0.15,  # min left/right separation
    "rel_depth_m":     0.25,  # min in-front/behind separation
    "rel_vert_m":      0.15,  # min above/below separation
    "rel_axis_margin": 1.5,   # a horizontal direction must dominate the other axis by this factor
    "rel_near_m":      1.0,
    "rel_on_overlap":  0.5,   # image-box horizontal overlap (fraction of the smaller box)
    "rel_on_xy_m":     0.6,   # max horizontal centre offset for "on"
    "rel_min_frames":  2,     # an episode must be seen in at least this many frames
    "rel_gap_s":       8.0,   # frames closer than this belong to one episode
    "rel_max_objects_per_frame": 20,
    # Place nodes from per-frame scene tags.
    "place_smooth_frames": 1, # majority filter half-width over grid frames (1 -> window of 3)
}

METRIC_SOURCES = ("wilddet3d_pair_lsq", "da3_metric")

def is_metric(pose_entry):
    """True when a video's world positions are in metres (solved scale, or metric by construction)."""
    return bool(pose_entry) and pose_entry.get("scale_source") in METRIC_SOURCES

def seg_params(pose_entry, stride_clip=None):
    """(threshold, k observations, units) for persistence segmentation of one video.
    Threshold is 0.3 m for metric videos and CONFIG["seg_thresh_unscaled"] scene units otherwise;
    persistence is configured in SECONDS and converted with the video's own grid spacing."""
    stride_clip = stride_clip or CONFIG["grid_stride"]
    stride_s = stride_clip * CLIP_RATIO / FPS
    k = max(1, int(round(CONFIG["seg_persist_s"] / stride_s)) + 1)
    if is_metric(pose_entry):
        return CONFIG["seg_thresh_m"], k, "m"
    return CONFIG["seg_thresh_unscaled"], k, "scene_units"

# Answering-time profile for `hdepic_vqa_agent.py --dataset ucs --profile ucs`. Explicit CLI flags
# override any entry. Without --profile the agent keeps its v2 behaviour.
RETRIEVAL = {
    "rank":            "text",
    "causal":          True,     # snapshot of the memory as it stood at the question time
    "ego_labels":      "both",   # numeric | sector8 | both
    "relations":       True,
    "places":          True,
    "ego_elev_deg":    15.0,
    "relations_max":   10,
    "places_max_objects": 12,
}
# v4 profile (`--profile ucs2`), 2026-09-17: same as above plus pose-free geometry. Every candidate
# and history line carries the object's position relative to the CAMERA at the moment it was last
# seen (WildDet3D's metric cam_3d), comparison questions ("X or Y") retrieve both named objects,
# "how many" questions get a counting aid from per-frame co-visibility, and the wearer's path is
# withheld on videos whose poses could not be scaled. See ucs_sidecars.py (v4 helpers).
RETRIEVAL_V4 = {**RETRIEVAL, "direct_obs": True, "pair_retrieval": True, "count_context": False,
                "ego_path_unscaled": False}   # count_context: tried 2026-09-17, made counting WORSE (15 vs 22.5)

def lift_budgets(n_grid_frames):
    """Per-FRAME lift budget, identical in spirit to hdepic_adapter.lift_budgets.

    UCS-Bench videos run 10 s to 98 min (5 to 1200 grid frames after the cap), an even wider spread
    than HD-EPIC's, so a global top-N budget would starve the long ones and bias the survivors toward
    large high-scoring fixtures.
    """
    return {"scene_lift_budget": max(CONFIG["min_scene_budget"],
                                     CONFIG["per_frame_lift"] * int(n_grid_frames)),
            "max_per_tag":       max(CONFIG["min_per_tag"], int(n_grid_frames) // 4)}

# ---------------------------------------------------------------- enumeration
# Stitched streams (multi-recording robustness experiment, ucs_full/stitch/): an id "stitch__<cond>__<key>" maps to a
# concatenated mp4 and to the questions of its TARGET (last) part. The registry is a JSON {sid: {"mp4":..., "target":...}}
# named by STITCH_REGISTRY, default ucs_full/stitch/true/registry.json. Normal ids are untouched.
_STITCH = None
def _stitch(vid):
    global _STITCH
    if not vid.startswith(("stitch__", "bigstitch__")): return None
    if _STITCH is None:
        _STITCH = {}
        for rp in (f"{M}/ucs_full/stitch/true/registry.json", f"{M}/ucs_full/stitch/big/registry.json", os.environ.get("STITCH_REGISTRY")):
            if rp and os.path.isfile(rp): _STITCH.update(json.load(open(rp)))
    return _STITCH.get(vid)

def video_path(vid):
    st = _stitch(vid)
    return st["mp4"] if st else f"{UCS}/{vid}.mp4"

def qa_path(vid):
    st = _stitch(vid)
    return f"{QADIR}/{st['target']}-mcq.json" if st else f"{QADIR}/{vid}-mcq.json"

SOURCES = ("ego4d", "epic", "egolife", "egolife_raw", "scannet", "numeric5", "other")

def source_of(vid):
    """Which upstream dataset a UCS-Bench video came from (drives the stratified subset)."""
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(-\d+)?", vid): return "ego4d"
    if re.fullmatch(r"P\d\d_\d+", vid):        return "epic"
    if vid.startswith("egolife_"):             return "egolife"
    if re.match(r"D\d-P\d", vid):              return "egolife_raw"
    if vid.startswith("scene0"):               return "scannet"
    if re.fullmatch(r"\d{5}", vid):            return "numeric5"
    return "other"

def list_videos(require_qa=True):
    """All UCS-Bench videos; by default only the 532 that actually carry questions.

    8 videos ship without a QA file (egolife_A1_D4_033/034/035/037/040, D5_023/029/030); they can
    never be evaluated, so they are excluded unless require_qa=False.
    """
    vids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{UCS}/*.mp4"))
    if require_qa:
        vids = [v for v in vids if os.path.isfile(qa_path(v))]
    return vids

# ---------------------------------------------------------------- video metadata
_META_CACHE = f"{M}/ucs_logs/video_meta.json"
_meta = None

def video_meta(vid):
    """(width, height, n_video_frames). Cached on disk -- resolution varies per video (six distinct
    sizes across the six sources) and must never be assumed."""
    global _meta
    if _meta is None:
        _meta = json.load(open(_META_CACHE)) if os.path.isfile(_META_CACHE) else {}
    if vid not in _meta:
        import cv2
        cap = cv2.VideoCapture(video_path(vid))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        cap.release()
        if abs(fps - FPS) > 0.05:
            print(f"  WARNING {vid}: fps {fps:.3f} != adapter FPS {FPS}; time_s will be wrong", flush=True)
        _meta[vid] = [W, H, n]
        os.makedirs(os.path.dirname(_META_CACHE), exist_ok=True)
        json.dump(_meta, open(_META_CACHE, "w"))
    return tuple(_meta[vid])

def n_video_frames(vid):
    return video_meta(vid)[2]

def grid_clip_frames(vid, stride=None, max_frames=None, n_frames=None):
    """Clip-frame grid. stride is in CLIP frames and CLIP_RATIO is 1, so it is also video frames."""
    stride = CONFIG["grid_stride"] if stride is None else stride
    max_frames = CONFIG["max_frames"] if max_frames is None else max_frames
    n = n_frames if n_frames is not None else n_video_frames(vid)
    mf = CONFIG.get("min_frames") or 0
    if mf and n // CLIP_RATIO > 1:
        # shrink the stride (never below 1) until the video yields at least mf frames
        stride = max(1, min(stride, (n // CLIP_RATIO) // mf))
    grid = list(range(0, n // CLIP_RATIO, stride))
    if len(grid) > max_frames:                     # uniform subsample (same policy as HD-EPIC)
        grid = [grid[i] for i in np.linspace(0, len(grid) - 1, max_frames).astype(int)]
    return grid

# ---------------------------------------------------------------- questions
def _hms(s):
    """'00:04:47.9' or '4:47' -> seconds. UCS-Bench timestamps are HH:MM:SS(.f)."""
    if s is None: return None
    parts = str(s).strip().split(":")
    try: parts = [float(p) for p in parts]
    except ValueError: return None
    while len(parts) < 3: parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def questions(vid):
    """UCS-Bench MCQ -> the record shape hdepic_vqa_agent already consumes.

    Differences from HD-EPIC that the agent has to know about:
      * NO <BBOX ...> anchor  -> there is no question box to lift or to IoU against, so the box-based
        rankers (iou / d3 / area1m / uni / b3d) have nothing to key on. `_anchor_box` is always None.
      * NO <TIME ...> token   -> the time anchor is the separate `question_timestamps` field, carried
        here as `_anchor_t` (seconds).
      * 2 OR 5 options        -> 1571 of the 8114 questions are BINARY (chance 50%, not 20%).
    """
    p = qa_path(vid)
    if not os.path.isfile(p): return []
    out = []
    for q in json.load(open(p)):
        opts = q.get("options") or {}
        if not opts: continue
        keys = sorted(opts)                                   # 'A'..'E'
        choices = [opts[k] for k in keys]
        lab = q.get("answer_label")
        if lab not in keys: continue                          # ungradable; drop rather than guess
        out.append({
            "inputs": {"video 1": {"id": vid}},
            "question": q["question"],
            "choices": choices,
            "correct_idx": keys.index(lab),
            "_video": vid,
            "_type": q.get("subcategory") or q.get("category") or "unknown",
            "_category": q.get("category"),
            "_subcategory": q.get("subcategory"),
            "_qid": q.get("qid") or q.get("q_id"),
            "_qtype": q.get("qtype"),
            "_difficulty": q.get("task_difficulty") or None,
            "_n_options": len(choices),
            "_anchor_t": _hms(q.get("question_timestamps")),
            "_anchor_box": None,
            "_evidence": [{"description": e.get("description"),
                           "start_s": _hms((e.get("timestamps") or {}).get("start")),
                           "end_s":   _hms((e.get("timestamps") or {}).get("end"))}
                          for e in (q.get("evidence") or [])],
        })
    return out

def all_questions(vids):
    qs = []
    for v in vids: qs += questions(v)
    return qs

# ---------------------------------------------------------------- GT (none)
def gt_objects(vid):
    """UCS-Bench ships no 3D ground truth. Returns [] so shared code that asks stays silent."""
    return []

def gt_names(vid):
    return []

# ---------------------------------------------------------------- grid spec
def write_grid_spec(vid, out_root, stride=None, max_frames=None, titles=None):
    """Write the detections.json-SHAPED grid spec that build_yolo_detect.py consumes via --sam3-root.

    Carries `clip_ratio` and `fps` so the detector stops assuming 6 / 30.0 (HD-EPIC and Ego4D specs
    have neither key and keep the old defaults).
    """
    n = n_video_frames(vid)
    grid = grid_clip_frames(vid, stride, max_frames, n_frames=n)
    W, H, _ = video_meta(vid)
    od = os.path.join(out_root, vid); os.makedirs(od, exist_ok=True)
    spec = {"clip_uid": vid, "video_uid": vid, "offset": 0, "N": n // CLIP_RATIO,
            "res": [W, H], "grid_clip_frames": grid, "titles": titles or [],
            "dataset": "ucs", "n_video_frames": n, "fps": FPS, "clip_ratio": CLIP_RATIO,
            "source": source_of(vid), "detections": []}
    json.dump(spec, open(os.path.join(od, "detections.json"), "w"), indent=2)
    return spec

# ---------------------------------------------------------------- poses (FastVGGT)
def _load_vggt(merging=6, device="cuda"):
    if FASTVGGT not in sys.path: sys.path.insert(0, FASTVGGT)
    from vggt.models.vggt import VGGT
    return VGGT.from_pretrained("facebook/VGGT-1B", merging=merging, merge_ratio=0.9).to(device).eval()

def _vggt_forward(model, paths, target=None, device="cuda"):
    """One FastVGGT pass -> (extrinsics (N,3,4) world->cam in this chunk's own frame,
    intrinsics (N,3,3) in SQUARE-preprocessed pixels, original_coords (N,5) [x1,y1,x2,y2,W,H]).

    Square (centre-pad) preprocessing keeps every image on an identical patch grid, which FastVGGT's
    token-merging asserts on. Unlike vggt_poses_egoloc.py we DO keep the intrinsics: UCS-Bench ships
    none, so VGGT's camera head is the only source of a K for the WildDet3D lift.
    """
    import torch
    target = target or CONFIG["vggt_target"]
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    images, coords = load_and_preprocess_images_square(paths, target_size=target)
    images = images.to(device)
    pw = ph = images.shape[-1] // 14           # FastVGGT reads static patch_width/height; set them
    for blk in list(model.aggregator.global_blocks) + list(model.aggregator.frame_blocks):
        if hasattr(blk, "attn"): blk.attn.patch_width = pw; blk.attn.patch_height = ph
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dt):
        agg, ps = model.aggregator(images[None])
        pose_enc = model.camera_head(agg)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    return (extr.squeeze(0).float().cpu().numpy(),
            intr.squeeze(0).float().cpu().numpy(),
            np.asarray(coords, float))

def _K_to_native(K_sq, coord, target):
    """VGGT K is in square-preprocessed pixels; map it back to the video's native resolution.

    load_and_preprocess_images_square centre-pads to max(W,H) then resizes to `target`, recording
    [x1,y1,x2,y2,W,H] -- the original image's box inside the square. So scale = target/max(W,H) and
    the native K is K_sq with f/scale and the padding offset removed from the principal point.
    """
    x1, y1, x2, y2, W, H = coord
    scale = target / max(W, H)
    return np.array([[K_sq[0, 0] / scale, 0.0, (K_sq[0, 2] - x1) / scale],
                     [0.0, K_sq[1, 1] / scale, (K_sq[1, 2] - y1) / scale],
                     [0.0, 0.0, 1.0]], float)

def _center_w2c(m):
    m = np.asarray(m, float); return -m[:3, :3].T @ m[:3, 3]

def _sim3(model_src, model_dst):
    """Robust Sim3 mapping src camera centres onto dst camera centres (reuses vq3d's RANSAC umeyama)."""
    if M not in sys.path: sys.path.insert(0, LP.CODE)
    from vggt_poses_egoloc import ransac_sim3
    return ransac_sim3(np.asarray(model_src, float), np.asarray(model_dst, float))

def poses_vggt_shape(vids, grid_root, frames_root, merging=6, chunk=None, overlap=None,
                     target=None, device="cuda", model=None, verbose=True):
    """{video: {good_poses, camera_poses, K, ...}} indexed by CLIP frame -- the same shape
    build_mem2_lift already expects from EgoLoc, so the lift needs no new file format.

    Long videos exceed one FastVGGT forward, so the grid is processed in overlapping chunks and each
    chunk is mapped into chunk 0's frame by a robust Sim3 over the shared cameras. `chain_rmse_m`
    per chunk is the residual of that fit -- watch it in `check`; a chunk that does not agree with
    its predecessor is a broken reconstruction, not a pose.

    Scale here is VGGT's arbitrary per-video unit. `solve_metric_scale` fixes it afterwards.
    """
    chunk   = chunk   or CONFIG["vggt_chunk"]
    overlap = overlap or CONFIG["vggt_overlap"]
    target  = target  or CONFIG["vggt_target"]
    if model is None: model = _load_vggt(merging, device)
    out = {}
    for vid in vids:
        spec_p = f"{grid_root}/{vid}/detections.json"
        grid = (json.load(open(spec_p))["grid_clip_frames"] if os.path.isfile(spec_p)
                else grid_clip_frames(vid))
        paths = [f"{frames_root}/{vid}/{cf:06d}.jpg" for cf in grid]
        keep  = [(cf, p) for cf, p in zip(grid, paths) if os.path.isfile(p)]
        if not keep:
            print(f"  {vid}: no grid frames; skip", flush=True); continue
        n_clip = n_video_frames(vid) // CLIP_RATIO
        step = max(1, chunk - overlap)
        starts = list(range(0, max(1, len(keep) - overlap), step))
        acc_extr = {}                     # clip_frame -> 3x4 world->cam in chunk-0 frame
        prev_centers = None               # {clip_frame: centre in chunk-0 frame}
        Ks = []; chain = []; reanchored = []
        for ci, st in enumerate(starts):
            sub = keep[st:st + chunk]
            if len(sub) < 2: continue
            extr, intr, coords = _vggt_forward(model, [p for _, p in sub], target, device)
            Ks += [_K_to_native(intr[i], coords[i], target) for i in range(len(sub))]
            cen = {cf: _center_w2c(extr[i]) for i, (cf, _) in enumerate(sub)}
            if ci == 0:
                for i, (cf, _) in enumerate(sub): acc_extr[cf] = np.asarray(extr[i], float)
                prev_centers = dict(cen); chain.append(0.0); continue
            shared = [cf for cf in cen if cf in prev_centers]
            if len(shared) < 3:
                print(f"  {vid}: chunk {ci} shares only {len(shared)} frames; dropped", flush=True)
                chain.append(None); continue
            src = np.array([cen[cf] for cf in shared]); dst = np.array([prev_centers[cf] for cf in shared])
            try:
                s, R, t, inl = _sim3(src, dst)
            except (np.linalg.LinAlgError, ValueError) as ex:
                # The chunk cannot be aligned to the previous one (seen on stitched streams at the cut between two
                # unrelated recordings: "SVD did not converge"). Crashing loses the whole video; instead RE-ANCHOR:
                # keep this chunk in its own frame and chain the following chunks to it. Positions before and after
                # the break then live in different frames, which is recorded in `chain_reanchored`.
                print(f"  {vid}: chunk {ci} could not be chained ({type(ex).__name__}); re-anchored", flush=True)
                for i, (cf, _) in enumerate(sub): acc_extr[cf] = np.asarray(extr[i], float)
                prev_centers = dict(cen); chain.append(None); reanchored.append(ci); continue
            rmse = float(np.sqrt(((s * (R @ src.T).T + t - dst) ** 2).sum(1).mean()))
            chain.append(round(rmse, 4))
            for i, (cf, _) in enumerate(sub):
                E = np.asarray(extr[i], float); Rv = E[:3, :3]
                Cg = s * (R @ cen[cf]) + t                      # centre in chunk-0 frame
                Rg = Rv @ R.T                                   # world->cam rotation in chunk-0 frame
                acc_extr[cf] = np.hstack([Rg, (-Rg @ Cg).reshape(3, 1)])
            prev_centers = {cf: s * (R @ cen[cf]) + t for cf in cen}
        good = [False] * n_clip
        poses = [np.zeros((3, 4)).tolist() for _ in range(n_clip)]
        for cf, E in acc_extr.items():
            if 0 <= cf < n_clip:
                good[cf] = True
                poses[cf] = [[round(float(v), 6) for v in row] for row in E]
        K = np.median(np.stack(Ks), 0) if Ks else None
        W, H, _ = video_meta(vid)
        out[vid] = {"good_poses": good, "camera_poses": poses,
                    "K": (K.tolist() if K is not None else None), "res": [W, H],
                    "n_grid": len(keep), "n_posed": int(sum(good)), "n_chunks": len(starts),
                    "chain_rmse": chain, "chain_reanchored": reanchored, "scale": 1.0, "scale_source": "vggt_arbitrary",
                    "frame": "VGGT chunk-0 frame; world = inv(pose) @ cam; UP TO SCALE until `scale` runs"}
        if verbose:
            ok = [c for c in chain if c is not None]
            print(f"  {vid}: {sum(good)}/{len(keep)} grid frames posed over {len(starts)} chunk(s), "
                  f"chain rmse med {np.median(ok) if ok else float('nan'):.3f} (vggt units)", flush=True)
    return out

# ---------------------------------------------------------------- pose backend per video
def is_circular_fisheye(vid, frames_root=None):
    """True for circular-fisheye footage (EgoLife / Aria RGB): a disc on a black square frame.
    Decided from pixels rather than the source name -- square frame, all four corners black (median,
    so burned-in overlay text in a corner does not flip it), centre not black."""
    import cv2
    W, H, _ = video_meta(vid)
    if W != H: return False
    fs = sorted(glob.glob(f"{frames_root or M + '/frames_grid_ucs'}/{vid}/*.jpg"))
    if not fs: return True                       # square with no frames yet: Aria is the only square source
    img = cv2.imread(fs[len(fs) // 2], cv2.IMREAD_GRAYSCALE)
    h, w = img.shape; k = max(4, int(0.04 * min(h, w)))
    corners = [img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]]
    centre = img[h // 2 - k:h // 2 + k, w // 2 - k:w // 2 + k]
    return all(float(np.median(c)) < 16 for c in corners) and float(centre.mean()) > 20

def pose_backend_for(vid, frames_root=None, mode=None):
    """Which pose estimator to use for one video. mode 'auto' routes circular fisheye footage to
    CONFIG['fisheye_pose_backend'] and everything else to CONFIG['pinhole_pose_backend']; any other
    mode is returned as-is."""
    mode = mode or CONFIG["pose_backend"]
    if mode != "auto": return mode
    return CONFIG["fisheye_pose_backend"] if is_circular_fisheye(vid, frames_root) else CONFIG["pinhole_pose_backend"]

def da3_gate(entry):
    """(ok, reasons, stats) for one video's DA3 poses: do its chunks agree with each other?"""
    rm = [x for x in (entry.get("chain_rmse") or []) if x is not None]
    sc = [x for x in (entry.get("chain_scale") or [])[1:] if x is not None]
    gp, cp = entry["good_poses"], entry["camera_poses"]
    C = np.array([-np.array(cp[i])[:3, :3].T @ np.array(cp[i])[:3, 3] for i in range(len(gp)) if gp[i]])
    st = np.linalg.norm(np.diff(C, axis=0), axis=1) if len(C) > 1 else np.zeros(1)
    reasons = []
    if rm and max(rm) > CONFIG["da3_gate_max_seam_rmse_m"]:
        reasons.append(f"seam rmse {max(rm):.2f} m > {CONFIG['da3_gate_max_seam_rmse_m']}")
    if sc and (min(sc) < CONFIG["da3_gate_scale_min"] or max(sc) > CONFIG["da3_gate_scale_max"]):
        reasons.append(f"chunk scale {min(sc):.2f}-{max(sc):.2f} outside "
                       f"[{CONFIG['da3_gate_scale_min']}, {CONFIG['da3_gate_scale_max']}]")
    if float(st.max()) > CONFIG["da3_gate_max_step_m"]:
        reasons.append(f"camera jump {st.max():.1f} m in one grid step > {CONFIG['da3_gate_max_step_m']}")
    stats = {"max_seam_rmse_m": round(max(rm), 3) if rm else 0.0,
             "chunk_scale_range": [round(min(sc), 3), round(max(sc), 3)] if sc else [1.0, 1.0],
             "max_step_m": round(float(st.max()), 2), "median_step_m": round(float(np.median(st)), 3)}
    return (not reasons), reasons, stats

# ---------------------------------------------------------------- poses (Depth Anything 3)
def _load_da3(model_id=None, device="cuda"):
    import torch
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(model_id or CONFIG["da3_model"]).to(device=torch.device(device)).eval()

def _as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _da3_forward(model, paths, process_res=None, want_depth=False):
    """One DA3 pass -> (extrinsics (N,3,4) world->cam OpenCV, intrinsics (N,3,3) in PROCESSED
    pixels, processed (w, h), depth (N,h,w) metres or None)."""
    import torch
    with torch.no_grad():
        pred = model.inference(paths, process_res=process_res or CONFIG["da3_process_res"])
    extr = _as_np(pred.extrinsics).astype(float)
    intr = _as_np(pred.intrinsics).astype(float)
    hp, wp = _as_np(pred.processed_images).shape[1:3]
    depth = _as_np(pred.depth).astype(np.float32) if want_depth else None
    return extr, intr, (int(wp), int(hp)), depth

def poses_da3_shape(vids, grid_root, frames_root, chunk=None, overlap=None, process_res=None,
                    device="cuda", model=None, verbose=True):
    """Same output shape as poses_vggt_shape, from DA3's NESTED model: poses and intrinsics are
    metric, so `scale_source` is "da3_metric" and the scale stage has nothing to solve.

    Long videos are processed in overlapping chunks joined by a robust Sim3 over shared cameras,
    exactly as for FastVGGT. Because every chunk is independently metric, the Sim3 scale between
    chunks should come out near 1.0; `chain_scale` records it per chunk as a built-in check that the
    metric claim holds on this footage.

    DA3 returns intrinsics for its resized input. They are mapped back to native pixels only when
    the resize preserved the aspect ratio; anything else (crop/pad) raises rather than silently
    producing a wrong K.
    """
    chunk = chunk or CONFIG["da3_chunk"]; overlap = overlap or CONFIG["da3_overlap"]
    process_res = process_res or CONFIG["da3_process_res"]
    if model is None: model = _load_da3(device=device)
    out = {}
    for vid in vids:
        spec_p = f"{grid_root}/{vid}/detections.json"
        grid = (json.load(open(spec_p))["grid_clip_frames"] if os.path.isfile(spec_p) else grid_clip_frames(vid))
        keep = [(cf, f"{frames_root}/{vid}/{cf:06d}.jpg") for cf in grid]
        keep = [(cf, p) for cf, p in keep if os.path.isfile(p)]
        if not keep:
            print(f"  {vid}: no grid frames; skip", flush=True); continue
        W, H, _ = video_meta(vid)
        n_clip = n_video_frames(vid) // CLIP_RATIO
        step = max(1, chunk - overlap)
        starts = list(range(0, max(1, len(keep) - overlap), step))
        acc, prev, Ks, chain, chain_s, diag = {}, None, [], [], [], []
        prev_depth = {}                              # overlap-frame depth, already in chunk-0 units
        for ci, st in enumerate(starts):
            sub = keep[st:st + chunk]
            if not sub: continue
            extr, intr, (wp, hp), depth = _da3_forward(model, [p for _, p in sub], process_res, want_depth=True)
            if abs(wp / hp - W / H) > 0.02 * (W / H):
                raise RuntimeError(f"{vid}: DA3 processed {wp}x{hp} is not an aspect-preserving resize "
                                   f"of {W}x{H}; intrinsics mapping for crop/pad is not implemented")
            sx, sy = W / wp, H / hp
            for K in intr:
                Ks.append(np.array([[K[0, 0] * sx, 0, K[0, 2] * sx], [0, K[1, 1] * sy, K[1, 2] * sy], [0, 0, 1]]))
            cen = {cf: _center_w2c(extr[i]) for i, (cf, _) in enumerate(sub)}
            dep = {cf: depth[i] for i, (cf, _) in enumerate(sub)}
            if prev is None:
                for i, (cf, _) in enumerate(sub): acc[cf] = np.asarray(extr[i], float)
                prev = dict(cen); prev_depth = {cf: dep[cf] for cf in list(dep)[-overlap:]}
                chain.append(0.0); chain_s.append(1.0); diag.append(None); continue
            shared = [cf for cf in cen if cf in prev]
            if len(shared) < 3:
                print(f"  {vid}: chunk {ci} shares only {len(shared)} frames; dropped", flush=True)
                chain.append(None); chain_s.append(None); diag.append(None); continue
            # SCALE from dense depth on the shared frames (thousands of pixels per frame) -- far better
            # conditioned than a Sim3 over a dozen nearly-coincident camera centres, which is what the
            # first version used and which drifted to 0.26 on a 5-chunk Aria video.
            ratios = []
            for cf in shared:
                if cf not in prev_depth: continue
                da, db = prev_depth[cf], dep[cf]
                m = np.isfinite(da) & np.isfinite(db) & (da > 0.05) & (db > 0.05)
                if m.sum() > 500: ratios.append(float(np.median(da[m] / db[m])))
            s_c, _, _, _ = _sim3(np.array([cen[cf] for cf in shared]), np.array([prev[cf] for cf in shared]))
            s = float(np.median(ratios)) if ratios else float(s_c)
            # ROTATION from camera orientations (well conditioned even for a stationary wearer):
            # the mapped world->cam rotation Rv @ R^T must equal the previous chunk's, so R ~ Rprev^T @ Rv.
            Msum = sum(acc[cf][:3, :3].T @ np.asarray(extr[sub.index((cf, dict(sub)[cf]))][:3, :3], float)
                       for cf in shared)
            U, _, Vt = np.linalg.svd(Msum); R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            src = np.array([cen[cf] for cf in shared]); dst = np.array([prev[cf] for cf in shared])
            t = (dst - s * (R @ src.T).T).mean(0)                       # TRANSLATION from centres
            chain.append(round(float(np.sqrt(((s * (R @ src.T).T + t - dst) ** 2).sum(1).mean())), 4))
            chain_s.append(round(s, 4))
            diag.append({"depth_scale": round(s, 4), "centre_sim3_scale": round(float(s_c), 4),
                         "n_depth_frames": len(ratios),
                         "depth_ratio_iqr": ([round(float(x), 3) for x in np.percentile(ratios, [25, 75])] if ratios else None)})
            for i, (cf, _) in enumerate(sub):
                E = np.asarray(extr[i], float); Rg = E[:3, :3] @ R.T; Cg = s * (R @ cen[cf]) + t
                acc[cf] = np.hstack([Rg, (-Rg @ Cg).reshape(3, 1)])
            prev = {cf: s * (R @ cen[cf]) + t for cf in cen}
            prev_depth = {cf: s * dep[cf] for cf in list(dep)[-overlap:]}
        good = [False] * n_clip; poses = [np.zeros((3, 4)).tolist() for _ in range(n_clip)]
        for cf, E in acc.items():
            if 0 <= cf < n_clip:
                good[cf] = True; poses[cf] = [[round(float(v), 6) for v in row] for row in E]
        K = np.median(np.stack(Ks), 0) if Ks else None
        scales = [x for x in chain_s[1:] if x is not None]
        out[vid] = {"good_poses": good, "camera_poses": poses,
                    "K": (K.tolist() if K is not None else None), "res": [W, H],
                    "n_grid": len(keep), "n_posed": int(sum(good)), "n_chunks": len(starts),
                    "chain_rmse": chain, "chain_scale": chain_s, "chain_diag": diag,
                    "scale": 1.0, "scale_source": "da3_metric", "backend": "da3",
                    "model": CONFIG["da3_model"], "process_res": process_res,
                    "frame": "DA3 nested metric; chunk-0 frame; world = inv(pose) @ cam; METRES"}
        if verbose:
            ok = [c for c in chain if c]
            print(f"  {vid}: {sum(good)}/{len(keep)} grid frames posed over {len(starts)} chunk(s), "
                  f"chain rmse med {np.median(ok) if ok else 0.0:.3f} m, chunk scale med "
                  f"{np.median(scales) if scales else 1.0:.3f} (1.0 = metric chunks agree)", flush=True)
    return out

# ---------------------------------------------------------------- metric scale
def solve_metric_scale(mem, poses_entry, min_pairs=12, max_obs_per_object=40, base_frac=1.0,
                       min_frac_positive=0.80, max_mad_ratio=1.0):
    """Solve the ONE scalar that makes VGGT poses consistent with WildDet3D's metric lift.

    world = inv(pose) @ cam = R^T cam + C,  where C = -R^T t is the camera centre. WildDet3D's `cam`
    is metric; VGGT's C is in an arbitrary unit. Scaling pose translations by s scales C by s, so for
    a STATIC object observed in frames i and j:

        R_i^T cam_i + s*C_i  ==  R_j^T cam_j + s*C_j
        =>  s * (C_i - C_j)  ==  R_j^T cam_j - R_i^T cam_i

    Every same-tag observation pair gives one such 3-vector equation and the pooled least-squares
    solution is closed form -- no GPU, no re-lift, since `cam_3d` is already in objects_3d.json.

    TWO CHOICES HERE WERE MEASURED, NOT GUESSED (40-seed Monte Carlo, synthetic scenes at the ~0.2-0.3 m
    lift error the HD-EPIC analysis reports, with a third of tags deliberately carrying TWO distinct
    instances to imitate "two chairs"):

      * ALL pairs, gated on camera baseline (keep ||C_i - C_j|| >= base_frac * median baseline).
        The scale lives in the camera's motion, so short-baseline pairs are almost pure lift noise.
        An earlier version used only near-in-time pairs, reasoning they were likelier to be the same
        instance; that was backwards -- it scored 8.6-17.0% mean scale error against 2.2-3.0% here,
        and got WORSE as duplicate instances increased.
      * ROBUST reweighting around the median. On clean Gaussian noise it costs ~0.1 pt against plain
        least squares, but once a third of tags carry two instances it wins by 1.4 pt (2.4% vs 3.8%)
        -- cross-instance pairs are the real outliers, and this is what absorbs them.

    Expect ~3-5% scale error in practice. That is harmless for the ORDINAL questions UCS-Bench mostly
    asks ("which is closer") -- a global scale is monotone -- and matters for the metric ones
    (reachability thresholds).

    Returns (scale, diagnostics). scale is None when the video gives too little evidence -- a static
    camera, or too few repeated detections -- and the caller must then leave the memory unscaled and
    say so rather than invent a number.

    min_pairs was 20 and is now 12: it is only a crude VOLUME floor, and the identifiability gate
    below (sign agreement and MAD against the median) is the check that actually distinguishes signal
    from noise. Raising detector precision to conf 0.25 cut scene0720 from 72 observations to 26, so
    it fell to 18 pairs and was rejected on volume alone -- yet those 18 pairs agree on sign 100% of
    the time with MAD 0.10-0.18 of the median, and yield 2.38 against the 2.21 that the earlier,
    disjoint 72-observation sample produced. Two independent samples agreeing within 8% is evidence,
    not a fluke, and the floor was discarding it.
    """
    cp = poses_entry["camera_poses"]; gp = poses_entry["good_poses"]
    num, den, base = [], [], []
    for o in mem["objects"]:
        obs = [x for x in o["observations"]
               if x.get("cam_3d") and 0 <= x["clip_frame"] < len(gp) and gp[x["clip_frame"]]]
        if len(obs) < 2: continue
        obs = sorted(obs, key=lambda x: x["clip_frame"])
        if len(obs) > max_obs_per_object:      # uniform in TIME: keeps the long baselines that carry the signal
            obs = [obs[i] for i in np.linspace(0, len(obs) - 1, max_obs_per_object).astype(int)]
        C, Rt = [], []
        for x in obs:
            E = np.array(cp[x["clip_frame"]], float)
            C.append(-E[:3, :3].T @ E[:3, 3])
            Rt.append(E[:3, :3].T @ np.array(x["cam_3d"], float))
        C = np.array(C); Rt = np.array(Rt)
        ia, ib = np.triu_indices(len(obs), k=1)
        dC = C[ia] - C[ib]; rhs = Rt[ib] - Rt[ia]
        nb = np.linalg.norm(dC, axis=1)
        ok = nb > 1e-6                          # a camera that did not move carries no scale information
        if not ok.any(): continue
        num.append((dC[ok] * rhs[ok]).sum(1)); den.append((dC[ok] * dC[ok]).sum(1)); base.append(nb[ok])
    if not num:
        return None, {"n_pairs": 0, "reason": "no usable observation pairs"}
    num = np.concatenate(num); den = np.concatenate(den); base = np.concatenate(base)
    if base_frac > 0:
        keep = base >= base_frac * np.median(base)
        if keep.sum() >= min_pairs: num, den, base = num[keep], den[keep], base[keep]
    used = int(len(num))
    if used < min_pairs:
        return None, {"n_pairs": used, "reason": f"only {used} usable pairs (< {min_pairs})"}
    per = num / den                             # one scale estimate per pair
    s0 = float(np.median(per))
    mad = 1.4826 * float(np.median(np.abs(per - s0)))
    frac_pos = float(np.mean(per > 0))
    w = 1.0 / (1.0 + ((per - s0) / max(1e-6, mad)) ** 2)
    s = float((w * num).sum() / max(1e-12, (w * den).sum()))
    diag = {"n_pairs": used, "median_pair_scale": round(s0, 4), "weighted_scale": round(s, 4),
            "pair_scale_mad": round(mad, 4), "frac_positive": round(frac_pos, 3),
            "mad_over_median": round(mad / max(1e-9, abs(s0)), 3),
            "median_baseline_vggt": round(float(np.median(base)), 4)}
    # IDENTIFIABILITY GATE. Arithmetic always yields a number; that is not the same as the data
    # determining one. Measured on the first real pilot: only the 5-frame ScanNet clip was
    # identified (frac_positive 100%, mad/median 0.17). Every long egocentric video sat at
    # frac_positive 37-72% with mad up to 4x the median -- per-pair estimates spread symmetrically
    # about zero, i.e. carrying no scale information -- yet the pooled ratio still returned values
    # like 0.035 and 8.4 that would have been reported as metres.
    #
    # Why real data breaks an estimator that was exact in simulation: the derivation assumes the
    # object is STATIC between the two views and that both observations are the SAME instance.
    # Egocentric video violates both -- people move the objects they look at, and same-tag
    # observations are frequently different instances (the HD-EPIC analysis put ~75% of
    # spatially-correct retrievals on the wrong label). Robust reweighting absorbs outliers around a
    # real signal; it cannot manufacture one.
    #
    # Refusing is the correct outcome: an UNSCALED video is honestly in arbitrary units, whereas a
    # confidently wrong scale silently corrupts every distance downstream.
    if not np.isfinite(s) or s <= 0:
        return None, {**diag, "reason": f"degenerate solution s={s}"}
    if frac_pos < min_frac_positive:
        return None, {**diag, "reason": f"not identified: only {100*frac_pos:.0f}% of pair estimates "
                                        f"are positive (need {100*min_frac_positive:.0f}%)"}
    if mad > max_mad_ratio * abs(s0):
        return None, {**diag, "reason": f"not identified: pair-scale MAD {mad:.2f} is "
                                        f"{mad/max(1e-9,abs(s0)):.1f}x the median {s0:.2f} "
                                        f"(need <{max_mad_ratio})"}
    return s, diag

def apply_scale(mem, poses_entry, s):
    """Rescale a video's poses by s and recompute every observation's world position.

    Mutates both in place. Idempotent by construction: poses_entry['scale'] records what has already
    been applied, so re-running `scale` on an already-scaled video is a no-op rather than s^2.
    """
    cur = poses_entry.get("scale") or 1.0
    rel = s / cur
    if abs(rel - 1.0) > 1e-9:
        for cf, E in enumerate(poses_entry["camera_poses"]):
            if not poses_entry["good_poses"][cf]: continue
            E = np.array(E, float); E[:3, 3] *= rel
            poses_entry["camera_poses"][cf] = [[round(float(v), 6) for v in row] for row in E]
    poses_entry["scale"] = s
    poses_entry["scale_source"] = "wilddet3d_pair_lsq"
    cp = poses_entry["camera_poses"]; gp = poses_entry["good_poses"]
    n = 0
    for o in mem["objects"]:
        for x in o["observations"]:
            cf = x["clip_frame"]
            if not (0 <= cf < len(gp) and gp[cf] and x.get("cam_3d")): continue
            E = np.array(cp[cf], float)
            w = E[:3, :3].T @ (np.array(x["cam_3d"], float) - E[:3, 3])
            x["world_vggt"] = [round(float(v), 3) for v in w]; n += 1
        wv = np.array([x["world_vggt"] for x in o["observations"] if x.get("world_vggt")], float)
        o["world_vggt_median"] = ([round(float(v), 3) for v in np.median(wv, 0)] if len(wv) else None)
    return n

# ---------------------------------------------------------------- subset selection
def stratified_subset(n=10, seed=0, require_qa=True, prefer=None, max_grid=None):
    """Deterministic stratified pick of n videos, spread over the six upstream sources.

    UCS-Bench pools six datasets with wildly different lengths (ScanNet renders are 20 s, Ego4D runs
    to 98 min). A pilot drawn uniformly would be all EgoLife/Ego4D and would cost days of lift; one
    drawn by length alone would be all ScanNet and would measure nothing. So: allocate the n slots
    round-robin over the sources present, and inside each source take the videos closest to the
    source's MEDIAN question count, tie-broken by shorter duration -- representative content, cheapest
    instance of it.
    """
    vids = list_videos(require_qa=require_qa)
    prefer = prefer or ["egolife", "epic", "numeric5", "scannet", "ego4d", "egolife_raw"]
    by = {}
    for v in vids:
        by.setdefault(source_of(v), []).append(v)
    info = {}
    for src, vs in by.items():
        for v in vs:
            qs = json.load(open(qa_path(v)))
            dur = _hms(qs[0].get("video_length")) if qs else 0.0
            info[v] = {"src": src, "nq": len(qs), "dur": dur or 0.0}
    if max_grid:
        info = {v: d for v, d in info.items() if d["dur"] / CONFIG["grid_stride"] * FPS <= max_grid}
    pool = {}
    for v, d in info.items(): pool.setdefault(d["src"], []).append(v)
    for src, vs in pool.items():
        med = float(np.median([info[v]["nq"] for v in vs]))
        vs.sort(key=lambda v: (abs(info[v]["nq"] - med), info[v]["dur"], v))
    order = [s for s in prefer if s in pool] + [s for s in pool if s not in prefer]
    picked, i = [], 0
    while len(picked) < n and any(pool[s] for s in order):
        s = order[i % len(order)]; i += 1
        if pool[s]: picked.append(pool[s].pop(0))
    return sorted(picked, key=lambda v: (info[v]["src"], v)), info

# ---------------------------------------------------------------- CLI
def _cmd_prep(a):
    import cv2, time
    vids = _resolve(a.videos)
    print(f"[prep] {len(vids)} video(s)", flush=True)
    specs = {}
    for v in vids:
        s = write_grid_spec(v, a.grid_root, a.stride, a.max_frames)
        specs[v] = s
        W, H, n = video_meta(v)
        print(f"  {v}: {len(s['grid_clip_frames'])} grid frames, {W}x{H}, {n} video frames "
              f"({n/FPS:.0f}s, {source_of(v)})", flush=True)
    if a.skip_frames: print("UCS_PREP_DONE", flush=True); return
    print("[prep] grid frames", flush=True)
    for v in vids:
        grid = specs[v]["grid_clip_frames"]
        od = os.path.join(a.frames_root, v); os.makedirs(od, exist_ok=True)
        missing = [cf for cf in grid if not os.path.isfile(f"{od}/{cf:06d}.jpg")]
        if not missing:
            print(f"  {v}: {len(grid)} frames already; skip", flush=True); continue
        # sequential decode: these videos are long and CAP_PROP_POS_FRAMES seeking on a 98-min
        # re-encode costs more than reading straight through.
        cap = cv2.VideoCapture(video_path(v)); t0 = time.time(); want = set(missing); w = 0
        fv = -1
        while want:
            ok, bgr = cap.read()
            if not ok: break
            fv += 1                                  # count frames ourselves: POS_FRAMES is unreliable
            if fv % CLIP_RATIO: continue
            cf = fv // CLIP_RATIO
            if cf in want:
                cv2.imwrite(f"{od}/{cf:06d}.jpg", bgr); w += 1; want.discard(cf)
        cap.release()
        print(f"  {v}: {w}/{len(missing)} new frames in {time.time()-t0:.0f}s -> {od}", flush=True)
    print("UCS_PREP_DONE", flush=True)

def _cmd_poses(a):
    vids = _resolve(a.videos)
    out = json.load(open(a.out)) if (os.path.isfile(a.out) and not a.force) else {}
    todo = [v for v in vids if v not in out]
    print(f"[poses] {len(todo)}/{len(vids)} video(s) to estimate -> {a.out}", flush=True)
    if not todo: print("UCS_POSES_DONE", flush=True); return
    if a.backend == "da3":
        model = _load_da3()
        for v in todo:
            r = poses_da3_shape([v], a.grid_root, a.frames_root, chunk=a.da3_chunk,
                                overlap=a.da3_overlap, process_res=a.da3_process_res, model=model)
            out.update(r); json.dump(out, open(a.out, "w"))
        print(f"UCS_POSES_DONE {len(out)} videos -> {a.out}", flush=True); return
    model = _load_vggt(a.merging)
    for v in todo:                                  # write after each video: a crash keeps the rest
        r = poses_vggt_shape([v], a.grid_root, a.frames_root, merging=a.merging,
                             chunk=a.chunk, overlap=a.overlap, model=model)
        out.update(r); json.dump(out, open(a.out, "w"))
    print(f"UCS_POSES_DONE {len(out)} videos -> {a.out}", flush=True)

def _cmd_scale(a):
    vids = _resolve(a.videos)
    poses = json.load(open(a.poses))
    rep = {}
    for v in vids:
        mp = f"{a.mem_root}/{v}/objects_3d.json"
        if not (os.path.isfile(mp) and v in poses):
            print(f"  {v}: no objects_3d.json or no poses; skip", flush=True); continue
        mem = json.load(open(mp))
        if poses[v].get("scale_source") == "da3_metric":
            print(f"  {v}: DA3 metric poses -- metric by construction, nothing to solve", flush=True)
            rep[v] = {"scale": 1.0, "note": "da3_metric: poses and depth already in metres"}
            continue
        # NOT IDEMPOTENT WITHOUT THIS. apply_scale stores an ABSOLUTE scale but applies a RELATIVE
        # correction (rel = s/cur). Re-running stage 7 on already-metric poses re-solves to ~1.0 and
        # then multiplies translations by 1.0/cur -- silently reverting the scaling. Observed on
        # scene0720_00-0: 2.2084 on the first pass, 1.0000 on the second, poses back in VGGT units.
        if poses[v].get("scale_source") == "wilddet3d_pair_lsq" and not a.force:
            print(f"  {v}: already scaled ({poses[v].get('scale'):.4f}); skipping (--force to redo)",
                  flush=True)
            rep[v] = {"scale": poses[v].get("scale"), "note": "already scaled; not re-solved"}
            continue
        s, diag = solve_metric_scale(mem, poses[v], min_pairs=a.min_pairs)
        if s is None:
            print(f"  {v}: UNSCALED ({diag['reason']}) -- positions stay in VGGT units", flush=True)
            rep[v] = {"scale": None, **diag}; continue
        n = apply_scale(mem, poses[v], s)
        mem["scale_applied"] = s; mem["scale_diag"] = diag
        json.dump(mem, open(mp, "w"), indent=2)
        rep[v] = {"scale": round(s, 4), "n_world": n, **diag}
        print(f"  {v}: scale {s:.4f} m/vggt-unit from {diag['n_pairs']} pairs "
              f"(mad {diag['pair_scale_mad']}), {n} world positions rewritten", flush=True)
    json.dump(poses, open(a.poses, "w"))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=2)
    print(f"UCS_SCALE_DONE -> {a.report}", flush=True)

def _cmd_posegate(a):
    allp = json.load(open(a.poses)); e = allp.get(a.video)
    if not e: print("SKIP (no entry)"); return
    if e.get("backend") != "da3": print(f"SKIP (backend {e.get('backend', 'fastvggt')})"); return
    ok, reasons, stats = da3_gate(e)
    e["da3_gate"] = {"ok": ok, "reasons": reasons, "stats": stats}
    json.dump(allp, open(a.poses, "w"))
    print("PASS " + json.dumps(stats) if ok else "FAIL " + "; ".join(reasons))

def _cmd_posebackend(a):
    print(pose_backend_for(a.video, a.frames_root, a.mode))

def _cmd_segparams(a):
    pe = json.load(open(a.poses)).get(a.video, {})
    gp = f"{a.grid_root}/{a.video}/detections.json"
    grid = json.load(open(gp))["grid_clip_frames"] if os.path.isfile(gp) else []
    stride = (grid[1] - grid[0]) if len(grid) > 1 else None
    thr, k, units = seg_params(pe, stride)
    print(f"{thr} {k} {units}")

def _cmd_check(a):
    """Gate-1 SUBSTITUTE. UCS-Bench has no GT 3D and no GT 2D boxes, so the real projection test is
    impossible. These three checks are what remains; they can prove the geometry INCOHERENT, they
    cannot prove it correct. Read them in that spirit.

      A. frame-index convention -- decode the video at CLIP_RATIO*cf and compare to the grid JPEG.
         This is the check that guards the #1 documented corruption source, and it IS conclusive.
      B. pose chaining          -- the Sim3 residual joining each VGGT chunk to its predecessor.
      C. lift self-consistency  -- take each cluster's median world point, project it back into a
         DIFFERENT frame of the same cluster through that frame's pose and K, and ask whether it
         lands inside the box detected there. Tests pose, K and scale jointly. The HD-EPIC GT-based
         gate scored 98.29% inside / 22 px median; this one is measured against the detector's own
         boxes, so it is a weaker, more optimistic statistic. Treat <70% as broken, not as passing.
    """
    import cv2
    vids = _resolve(a.videos)
    poses = json.load(open(a.poses)) if os.path.isfile(a.poses) else {}
    rows = []
    for v in vids:
        r = {"video": v, "source": source_of(v)}
        # ---- A. frame-index convention
        grid = grid_clip_frames(v)
        probe = [grid[i] for i in np.linspace(0, len(grid) - 1, min(a.probe_frames, len(grid))).astype(int)]
        cap = cv2.VideoCapture(video_path(v)); diffs = []
        for cf in probe:
            p = f"{a.frames_root}/{v}/{cf:06d}.jpg"
            if not os.path.isfile(p): continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, CLIP_RATIO * cf); ok, ref = cap.read()
            if not ok: continue
            got = cv2.imread(p)
            if got is None or got.shape != ref.shape: continue
            diffs.append(float(np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean()))
        cap.release()
        r["frame_index_probe_n"] = len(diffs)
        r["frame_index_mae"] = round(float(np.median(diffs)), 2) if diffs else None
        r["frame_index_ok"] = bool(diffs) and float(np.median(diffs)) < a.frame_mae_tol
        # ---- B. pose chaining
        pe = poses.get(v)
        if pe:
            ch = [c for c in (pe.get("chain_rmse") or []) if c is not None]
            r["posed_frames"] = f'{pe.get("n_posed")}/{pe.get("n_grid")}'
            r["chain_rmse_med"] = round(float(np.median(ch)), 4) if ch else None
            r["chain_rmse_max"] = round(float(max(ch)), 4) if ch else None
            r["scale"] = pe.get("scale"); r["scale_source"] = pe.get("scale_source")
        # ---- C. lift self-consistency (reprojection into a different frame of the same cluster)
        mp = f"{a.mem_root}/{v}/objects_3d_trajectories.json"
        if os.path.isfile(mp) and pe and pe.get("K"):
            mem = json.load(open(mp)); K = np.array(pe["K"], float)
            cp = pe["camera_poses"]; gp = pe["good_poses"]
            inside, off, tot = 0, [], 0
            for o in mem["objects"]:
                tr = [x for x in (o.get("trajectory") or []) if x.get("world_vggt") or x.get("world_egoloc")]
                if len(tr) < 2: continue
                wc = np.median(np.array([(x.get("world_egoloc") or x.get("world_vggt")) for x in tr], float), 0)
                for x in tr:
                    cf = x["clip_frame"]
                    if not (0 <= cf < len(gp) and gp[cf]): continue
                    E = np.array(cp[cf], float)
                    cam = E[:3, :3] @ wc + E[:3, 3]
                    if cam[2] <= 1e-6: tot += 1; continue         # behind the camera: counts as a miss
                    u = K[0, 0] * cam[0] / cam[2] + K[0, 2]; vv = K[1, 1] * cam[1] / cam[2] + K[1, 2]
                    b = x["box_2d"]; tot += 1
                    if b[0] <= u <= b[2] and b[1] <= vv <= b[3]: inside += 1; off.append(0.0)
                    else:
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        off.append(float(np.hypot(u - cx, vv - cy)))
            r["reproj_n"] = tot
            r["reproj_inside_pct"] = round(100.0 * inside / tot, 1) if tot else None
            r["reproj_median_px"] = round(float(np.median(off)), 1) if off else None
        rows.append(r)
        print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rows, open(a.report, "w"), indent=2)
    bad = [r for r in rows if r.get("frame_index_ok") is False]
    print(f"\nGATE-A frame index: {len(rows)-len(bad)}/{len(rows)} videos consistent"
          + (f"  FAILED: {[r['video'] for r in bad]}" if bad else ""), flush=True)
    ins = [r["reproj_inside_pct"] for r in rows if r.get("reproj_inside_pct") is not None]
    if ins: print(f"GATE-C reprojection: median {np.median(ins):.1f}% inside own box "
                  f"(GT-based HD-EPIC gate was 98.3%; this is a weaker self-consistency proxy)", flush=True)
    print("UCS_CHECK_DONE", flush=True)

def _cmd_questions(a):
    vids = _resolve(a.videos)
    qs = all_questions(vids)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(qs, open(a.out, "w"), indent=1)
    import collections
    print(f"{len(qs)} questions over {len(vids)} videos -> {a.out}", flush=True)
    print("  by n_options:", dict(collections.Counter(q["_n_options"] for q in qs)), flush=True)
    print("  chance level: "
          f"{100*np.mean([1.0/q['_n_options'] for q in qs]):.1f}%", flush=True)
    for k, n in collections.Counter(q["_subcategory"] for q in qs).most_common():
        print(f"  {n:5d}  {k}", flush=True)

def _cmd_subset(a):
    picked, info = stratified_subset(a.num, seed=a.seed, max_grid=a.max_grid)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for v in picked: fh.write(v + "\n")
    tot_q = tot_g = 0
    print(f"{'video':34s} {'source':12s} {'dur_s':>7s} {'grid':>5s} {'nq':>4s}")
    for v in picked:
        d = info[v]; g = int(d["dur"] * FPS / CONFIG["grid_stride"]) + 1
        tot_q += d["nq"]; tot_g += g
        print(f"{v:34s} {d['src']:12s} {d['dur']:7.0f} {g:5d} {d['nq']:4d}")
    print(f"\n{len(picked)} videos, {tot_q} questions, ~{tot_g} grid frames -> {a.out}")

def _resolve(spec):
    if spec == "all": return list_videos()
    if os.path.isfile(spec): return [l.strip() for l in open(spec) if l.strip()]
    return [s.strip() for s in spec.split(",") if s.strip()]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="UCS-Bench adapter: subset, prep, poses, scale, check")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("subset", help="deterministic stratified pick of N videos")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-grid", type=int, default=0, help="drop videos above this many grid frames")
    p.add_argument("--out", default=f"{M}/ucs_logs/subset10.txt")
    p.set_defaults(fn=_cmd_subset)

    p = sub.add_parser("prep", help="grid specs + grid frames")
    p.add_argument("--videos", required=True)
    p.add_argument("--grid-root",   default=f"{M}/ucs_grid")
    p.add_argument("--frames-root", default=f"{M}/frames_grid_ucs")
    p.add_argument("--stride", type=int, default=CONFIG["grid_stride"])
    p.add_argument("--max-frames", type=int, default=CONFIG["max_frames"])
    p.add_argument("--skip-frames", action="store_true")
    p.set_defaults(fn=_cmd_prep)

    p = sub.add_parser("poses", help="FastVGGT poses + intrinsics (chunked, Sim3-chained)")
    p.add_argument("--videos", required=True)
    p.add_argument("--grid-root",   default=f"{M}/ucs_grid")
    p.add_argument("--frames-root", default=f"{M}/frames_grid_ucs")
    p.add_argument("--out", default=f"{M}/ucs_poses.json",
                   help="NOTE: rewritten whole, so concurrent shards must use distinct paths")
    p.add_argument("--merging", type=int, default=6)
    p.add_argument("--chunk",   type=int, default=CONFIG["vggt_chunk"])
    p.add_argument("--overlap", type=int, default=CONFIG["vggt_overlap"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--backend", choices=["fastvggt", "da3"], default=CONFIG["pose_backend"])
    p.add_argument("--da3-chunk", type=int, default=CONFIG["da3_chunk"])
    p.add_argument("--da3-overlap", type=int, default=CONFIG["da3_overlap"])
    p.add_argument("--da3-process-res", type=int, default=CONFIG["da3_process_res"])
    p.set_defaults(fn=_cmd_poses)

    p = sub.add_parser("posegate", help="PASS/FAIL the DA3 quality gate for one video's pose file")
    p.add_argument("--video", required=True)
    p.add_argument("--poses", required=True)
    p.set_defaults(fn=_cmd_posegate)

    p = sub.add_parser("posebackend", help="print the pose backend to use for one video")
    p.add_argument("--video", required=True)
    p.add_argument("--frames-root", default=f"{M}/frames_grid_ucs")
    p.add_argument("--mode", default="auto")
    p.set_defaults(fn=_cmd_posebackend)

    p = sub.add_parser("segparams", help="print 'threshold k units' for persistence segmentation")
    p.add_argument("--video", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--grid-root", default=f"{M}/ucs_grid")
    p.set_defaults(fn=_cmd_segparams)

    p = sub.add_parser("scale", help="solve+apply the metric scale onto objects_3d.json and the poses")
    p.add_argument("--videos", required=True)
    p.add_argument("--mem-root", default=f"{M}/outputs_mem2_ucs")
    p.add_argument("--poses",    default=f"{M}/ucs_poses.json")
    p.add_argument("--min-pairs", type=int, default=12)
    p.add_argument("--force", action="store_true", help="re-solve even if already scaled")
    p.add_argument("--report", default=f"{M}/ucs_logs/scale_report.json")
    p.set_defaults(fn=_cmd_scale)

    p = sub.add_parser("check", help="Gate-1 substitute: frame index, pose chaining, reprojection")
    p.add_argument("--videos", required=True)
    p.add_argument("--frames-root", default=f"{M}/frames_grid_ucs")
    p.add_argument("--mem-root",    default=f"{M}/outputs_mem2_ucs")
    p.add_argument("--poses",       default=f"{M}/ucs_poses.json")
    p.add_argument("--probe-frames", type=int, default=8)
    p.add_argument("--frame-mae-tol", type=float, default=3.0, help="mean abs pixel diff, jpeg noise")
    p.add_argument("--report", default=f"{M}/ucs_logs/check_report.json")
    p.set_defaults(fn=_cmd_check)

    p = sub.add_parser("questions", help="dump the normalized question set")
    p.add_argument("--videos", required=True)
    p.add_argument("--out", default=f"{M}/ucs_logs/questions.json")
    p.set_defaults(fn=_cmd_questions)

    a = ap.parse_args(); a.fn(a)
