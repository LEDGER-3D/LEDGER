"""Stage 5.5 -- LLM descriptions per movement segment, so the retrieval agent can answer purely
from memory without re-opening the video.

For each object in the (tracker-collapsed) memory:
  * split its time-ordered trajectory into MOVEMENT SEGMENTS in metric 3D world coordinates --
    a new segment starts when the object moves more than --seg-thresh (default 0.3 m) from the
    current segment's running centroid. Consecutive observations of a stationary object collapse
    into one segment (near-identical frames -> one description instead of 24 near-duplicates);
    an actual move starts a new one, which is exactly what "moved from X to Y" questions need.
  * pick a REPRESENTATIVE observation per segment: nearest the segment's median world position
    (same rule build_trajectory_memory uses for instances/), tie-broken by detection score.
  * ask the VLM to describe the highlighted object IN CONTEXT using the FULL frame.
  * finally, a short object-level summary synthesised from the segment descriptions.

Writes object_memory.json (trajectory + descriptions) next to the input memory.
"""
import argparse, json, os, sys, glob, time, urllib.request
import numpy as np, cv2

import ledger_paths as LP
M = LP.WORK
SEG_PROMPT = ("The outlined object in this first-person kitchen frame is a {tag}. "
              "In 2-3 factual sentences describe: what the object is, what surface or container it is "
              "resting on or inside, and which other objects are immediately around it. "
              "Describe only what is visible. Do not speculate. No preamble.")
SUM_PROMPT = ("An object tracked through a first-person kitchen video was observed at these places:\n{obs}\n"
              "In ONE sentence, summarise what this object is and where it was kept/moved. No preamble.")

def ask(server, prompt, images=None, max_new_tokens=180):
    body = json.dumps({"prompt": prompt, "images": images or [],
                       "max_new_tokens": max_new_tokens, "no_think": True}).encode()
    rq = urllib.request.Request(server.rstrip("/") + "/generate", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=600) as r:
        return json.loads(r.read())["text"].strip()

def wpos(o):
    return o.get("world_egoloc") or o.get("world_vggt")

def segments(traj, thresh):
    """Split a time-ordered trajectory into movement segments in metric world coords."""
    segs = []
    for o in traj:
        p = wpos(o)
        if p is None: continue
        p = np.array(p, float)
        if segs and np.linalg.norm(p - np.mean([wpos(x) for x in segs[-1]], 0)) <= thresh:
            segs[-1].append(o)
        else:
            segs.append([o])
    return segs

def _seg_persist(traj, thresh, k):
    """Persistence rule, imported unchanged from reseg_memory."""
    if M not in sys.path: sys.path.insert(0, LP.CODE)
    from reseg_memory import seg_persist
    return seg_persist(traj, thresh, k)

def representative(seg):
    """Nearest the segment's median world position; ties broken by detection score."""
    cen = np.median([wpos(o) for o in seg], 0)
    return min(seg, key=lambda o: (round(float(np.linalg.norm(np.array(wpos(o)) - cen)), 2),
                                   -float(o.get("score") or 0)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mem-root", default=f"{M}/outputs_mem2")
    ap.add_argument("--server", required=True)
    ap.add_argument("--dataset", default="ucs", choices=["ucs"])
    ap.add_argument("--frames-grid", default=None, help="pre-extracted grid JPEGs (avoids video seeks)")
    ap.add_argument("--seg-thresh", type=float, default=0.3, help="metres of 3D motion that starts a new segment")
    ap.add_argument("--seg-mode", choices=["threshold", "persist"], default="threshold",
                    help="threshold = original rule; persist = a move counts only if sustained for --seg-persist observations")
    ap.add_argument("--seg-persist", type=int, default=3)
    ap.add_argument("--max-seg-per-object", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=896, help="downscale the full frame before the VLM")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--max-objects", type=int, default=0, help="0 = all")
    ap.add_argument("--flush-every", type=int, default=25)
    a = ap.parse_args()

    SERVERS = [x.strip() for x in a.server.split(",") if x.strip()]
    cdir = f"{a.mem_root}/{a.clip}"
    src = (f"{cdir}/objects_3d_trajectories_tracker.json" if os.path.isfile(f"{cdir}/objects_3d_trajectories_tracker.json")
           else f"{cdir}/objects_3d_trajectories.json")
    mem = json.load(open(src)); objs = mem["objects"]
    vid = mem.get("video_uid") or a.clip
    print(f"[{a.clip}] {len(objs)} objects from {os.path.basename(src)}", flush=True)

    if a.dataset in ("hdepic", "ucs"):
        sys.path.insert(0, LP.CODE)
        HA = __import__("hdepic_adapter" if a.dataset == "hdepic" else "ucs_adapter")
        vpath = HA.video_path(vid)
    else:
        vpath = f"{LP.EGO4D}/vq3d_subset/full_scale/{vid}.mp4"
    cap = None
    tmp = f"{cdir}/_desc_tmp.jpg"
    t0 = time.time(); ndesc = 0

    order = sorted(range(len(objs)), key=lambda i: -(objs[i].get("n_obs") or 0))
    if a.max_objects: order = order[:a.max_objects]
    out_path = f"{cdir}/object_memory.json"
    def flush():
        mem["descriptions"] = {"model": "Qwen3.5-9B", "seg_thresh_m": a.seg_thresh,
                               "frame": "full frame, object outlined", "n_descriptions": ndesc,
                               "described_objects": done_n, "total_objects": len(objs)}
        if a.seg_mode == "persist":
            mem["descriptions"].update({"seg_mode": "persist", "seg_persist_k": a.seg_persist})
        json.dump(mem, open(out_path, "w"), indent=2)
    done_n = 0
    for oi in order:
        o = objs[oi]
        traj = o.get("trajectory") or o.get("observations") or []
        segs = (_seg_persist(traj, a.seg_thresh, a.seg_persist) if a.seg_mode == "persist"
                else segments(traj, a.seg_thresh))
        if len(segs) > a.max_seg_per_object:                  # keep the temporally spread ones
            idx = np.linspace(0, len(segs) - 1, a.max_seg_per_object).astype(int)
            segs = [segs[i] for i in idx]
        # tracker-collapsed objects use primary_tag/secondary_tags; pre-collapse ones use tag
        srv = SERVERS[oi % len(SERVERS)]      # spread objects over all model servers
        tag = o.get("primary_tag") or o.get("tag") or "object"
        sec = [t for t in (o.get("secondary_tags") or []) if t and t != tag]
        tag_full = f"{tag} (also detected as {', '.join(sec)})" if sec else tag
        out_segs = []
        for si, seg in enumerate(segs):
            rep = representative(seg)
            frame = None
            if a.frames_grid and rep.get("clip_frame") is not None:
                fp = f"{a.frames_grid}/{a.clip}/{int(rep['clip_frame']):06d}.jpg"
                if os.path.isfile(fp): frame = cv2.imread(fp)
            if frame is None:
                if cap is None: cap = cv2.VideoCapture(vpath)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(rep["frame_video"])); ok, frame = cap.read()
                if not ok: continue
            b = [int(v) for v in rep["box_2d"]]
            vis = frame.copy()
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 215, 255), 6)   # highlight, full frame kept
            s = a.img_size / max(vis.shape[:2])
            if s < 1: vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
            cv2.imwrite(tmp, vis)
            try:
                desc = ask(srv, SEG_PROMPT.format(tag=tag_full), [tmp])
            except Exception as e:
                desc = None; print(f"    seg {si} desc failed: {type(e).__name__}", flush=True)
            wp = wpos(rep)
            out_segs.append({"segment": si, "n_obs": len(seg),
                             "time_range_s": [seg[0].get("time_s"), seg[-1].get("time_s")],
                             "rep_frame_video": rep.get("frame_video"), "rep_clip_frame": rep.get("clip_frame"),
                             "world": [round(float(v), 3) for v in wp] if wp else None,
                             "box_2d": rep.get("box_2d"), "description": desc})
            if desc: ndesc += 1
        o["segments"] = out_segs
        o["name"] = tag
        o["n_segments"] = len(out_segs)
        if not a.no_summary and out_segs:
            lines = "\n".join(f"- at {s['world']}: {s['description']}" for s in out_segs if s["description"])
            try: o["summary"] = ask(srv, SUM_PROMPT.format(obs=lines[:4000]), None, 120) if lines else None
            except Exception: o["summary"] = None
        done_n += 1
        if done_n % a.flush_every == 0: flush()
        print(f"  [{done_n}/{len(order)}] {tag:<20} {len(traj):>3} obs -> {len(out_segs)} segments", flush=True)

    if cap is not None: cap.release()
    if os.path.isfile(tmp): os.remove(tmp)
    flush(); out = out_path
    print(f"DONE {ndesc} descriptions over {len(objs)} objects in {time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == "__main__":
    main()
