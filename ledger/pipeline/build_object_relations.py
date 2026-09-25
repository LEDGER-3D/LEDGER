"""Object-object spatial relations, built DURING memory creation from same-frame geometry.

Runs as a pipeline stage right after tracker collapse -- the first point at which final object ids
exist -- and writes a SEPARATE sidecar, relations.json. It reads the per-observation camera-frame
positions (`cam_3d`) that every memory file already carries; no existing file, field or stage is
modified.

Why same-frame: two objects observed in one frame were lifted from one image with one camera, so
their relative position does not depend on camera pose at all -- pose error and any unknown pose
scale cancel. WildDet3D's cam_3d is metric on its own, so the distance thresholds are metres even on
a video whose poses never scaled.

Camera coordinates (OpenCV): +x right, +y down, +z forward. Relations are stated from the wearer's
viewpoint at that frame, which is the viewpoint UCS-Bench answers are written in.

Each relation is an EPISODE -- a label that held for a pair over a time span -- so a change such as
a cup moving from the table edge to its middle shows up as one episode ending and another starting.
"""
import argparse, json, os, sys, collections
import numpy as np
import ledger_paths as LP
M = LP.WORK

INVERSE = {"left of": "right of", "right of": "left of", "in front of": "behind", "behind": "in front of",
           "above": "below", "below": "above", "on": "under", "under": "on", "near": "near"}
DEFAULTS = {"rel_lat_m": 0.15, "rel_depth_m": 0.25, "rel_vert_m": 0.15, "rel_axis_margin": 1.5,
            "rel_near_m": 1.0, "rel_on_overlap": 0.5, "rel_on_xy_m": 0.6, "rel_min_frames": 2,
            "rel_gap_s": 8.0, "rel_max_objects_per_frame": 20}

def hoverlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, w) / max(1e-6, min(a[2] - a[0], b[2] - b[0]))

def pair_labels(ca, cb, ba, bb, P):
    """Labels for 'A <label> B' from same-frame camera coordinates and image boxes."""
    d = np.asarray(ca, float) - np.asarray(cb, float)
    dx, dy, dz = d
    out = []
    if abs(dx) >= P["rel_lat_m"] and abs(dx) >= P["rel_axis_margin"] * abs(dz):
        out.append("left of" if dx < 0 else "right of")
    if abs(dz) >= P["rel_depth_m"] and abs(dz) >= P["rel_axis_margin"] * abs(dx):
        out.append("in front of" if dz < 0 else "behind")
    on = False
    if dy <= -P["rel_vert_m"] and np.hypot(dx, dz) <= P["rel_on_xy_m"] and hoverlap(ba, bb) >= P["rel_on_overlap"]:
        bh = bb[3] - bb[1]
        if bb[1] - 0.15 * bh <= ba[3] <= bb[1] + 0.5 * bh:       # A's bottom edge sits at B's top
            out.append("on"); on = True
    if not on and abs(dy) >= P["rel_vert_m"]:
        out.append("above" if dy < 0 else "below")
    if np.linalg.norm(d) <= P["rel_near_m"]:
        out.append("near")
    return out, float(np.linalg.norm(d))

def load_params(dataset):
    P = dict(DEFAULTS)
    if dataset == "ucs":
        if M not in sys.path: sys.path.insert(0, LP.CODE)
        import ucs_adapter as UA
        P.update({k: v for k, v in UA.CONFIG.items() if k in DEFAULTS})
    return P

def build(objs, P):
    name = {o["id"]: (o.get("primary_tag") or o.get("tag") or "object") for o in objs}
    byf = collections.defaultdict(dict)                    # clip_frame -> {object id: best observation}
    for o in objs:
        for p in o.get("trajectory") or []:
            if p.get("cam_3d") is None or p.get("box_2d") is None or p.get("clip_frame") is None: continue
            cur = byf[p["clip_frame"]].get(o["id"])
            if cur is None or (p.get("score") or 0) > (cur.get("score") or 0):
                byf[p["clip_frame"]][o["id"]] = p
    hits = collections.defaultdict(list)                   # (a, b, label) -> [(t, dist)]
    n_frames = n_pairs = 0
    for cf, d in sorted(byf.items()):
        items = sorted(d.items(), key=lambda kv: -(kv[1].get("score") or 0))[:P["rel_max_objects_per_frame"]]
        if len(items) < 2: continue
        n_frames += 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ia, pa), (ib, pb) = items[i], items[j]
                if ia > ib: (ia, pa), (ib, pb) = (ib, pb), (ia, pa)
                labels, dist = pair_labels(pa["cam_3d"], pb["cam_3d"], pa["box_2d"], pb["box_2d"], P)
                n_pairs += 1
                for lab in labels:
                    if pa.get("time_s") is not None: hits[(ia, ib, lab)].append((pa["time_s"], dist))
    episodes = []
    for (ia, ib, lab), obs in hits.items():
        obs.sort()
        run = [obs[0]]
        def close(r):
            if len(r) >= P["rel_min_frames"]:
                episodes.append({"a": ia, "b": ib, "a_name": name[ia], "b_name": name[ib], "rel": lab,
                                 "t0": round(r[0][0], 1), "t1": round(r[-1][0], 1), "n_frames": len(r),
                                 "dist_m": round(float(np.median([x[1] for x in r])), 2)})
        for x in obs[1:]:
            if x[0] - run[-1][0] <= P["rel_gap_s"]: run.append(x)
            else: close(run); run = [x]
        close(run)
    episodes.sort(key=lambda e: (e["t0"], e["a"], e["b"], e["rel"]))
    return episodes, n_frames, n_pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mem-root", required=True)
    ap.add_argument("--dataset", default="ucs", choices=["ucs"])
    ap.add_argument("--out-name", default="relations.json")
    a = ap.parse_args()
    P = load_params(a.dataset)
    cdir = f"{a.mem_root}/{a.clip}"
    src = next((f"{cdir}/{fn}" for fn in ("objects_3d_trajectories_tracker.json", "objects_3d_trajectories.json")
                if os.path.isfile(f"{cdir}/{fn}")), None)
    if src is None: sys.exit(f"[{a.clip}] no trajectory memory in {cdir}")
    objs = json.load(open(src))["objects"]
    episodes, n_frames, n_pairs = build(objs, P)
    out = {"schema": "relations/v1", "clip_uid": a.clip, "source": os.path.basename(src),
           "frame": "camera coordinates of the frame both objects were seen in; viewer-relative; metres",
           "inverse": INVERSE, "params": P, "n_frames_with_pairs": n_frames,
           "n_pair_observations": n_pairs, "n_episodes": len(episodes), "episodes": episodes}
    json.dump(out, open(f"{cdir}/{a.out_name}", "w"), indent=1)
    c = collections.Counter(e["rel"] for e in episodes)
    print(f"[{a.clip}] {n_frames} frames, {n_pairs} pair observations -> {len(episodes)} relation episodes "
          f"{dict(c)}", flush=True)

if __name__ == "__main__": main()
