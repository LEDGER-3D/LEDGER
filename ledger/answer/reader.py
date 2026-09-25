"""Memory reader: retrieve from the object memory, render it as text, build the answering prompt.

The answering model never sees the video -- only this text. Two benchmark configurations are kept,
exactly as evaluated (the functions below are the evaluated code, not a re-implementation):

  UCS-Bench  text retrieval (question words vs object names/descriptions + time proximity), the memory
             as it stood at the question time (causal), egocentric direction + distance of each
             candidate, relations, places, the wearer's path and the event layer; options visible.
  HD-EPIC    the question's <BBOX> lifted to a 3D point, candidates ranked by 3D distance to it, the
             question restated in world coordinates; options visible.

Both prompts ask for IDENTIFICATION / REASONING / ANSWER, so every answer carries the memory evidence
it was drawn from.
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
import ucs_sidecars as SC  # noqa: E402

UCS_FPS, UCS_CLIP_RATIO = 10.0, 1          # UCS memories index time on a 10 fps clip grid
LET = "ABCDE"
BBOX_RE = re.compile(r'<BBOX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
TIME_RE = re.compile(r'<TIME\s+(\d+):(\d+):(\d+(?:\.\d+)?)')
# consumes an optional 'identified by', the token and its '>' so the rewrite reads as English
BBOX_PHRASE = re.compile(r'(?:identified by\s+)?<BBOX\s+[\d.\s]+\s*/?>')

UCS_FEAT = {"causal": True, "ego_labels": "both", "relations": True, "places": True,
            "ego_elev_deg": 15.0, "relations_max": 10, "places_max_objects": 12}
EVENTS_RECENT, EVENTS_MATCH = 6, 10
DT, TOPN = 6.0, 4


# ------------------------------------------------------------------ geometry / retrieval
def parse_anchor(q):
    bb, tm = BBOX_RE.search(q), TIME_RE.search(q)
    if not tm: return None, None
    t = int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + float(tm.group(3))
    if not bb: return None, t
    y1, x1, y2, x2 = [float(v) for v in bb.groups()]      # y-first -> xyxy
    return [x1, y1, x2, y2], t

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

def box_area(b): return max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1])

def pos_at(o, t, max_gap=12.0):
    """The object's 3D position AT time t, linearly interpolated between the two bracketing
    observations when they are within max_gap s of each other; otherwise the nearer observation."""
    pts = []
    for p in (o.get("trajectory") or []):
        ts = p.get("time_s"); w = p.get("world_egoloc") or p.get("world_vggt")
        if ts is not None and w: pts.append((float(ts), np.array(w, float)))
    if not pts: return None, None
    pts.sort(key=lambda z: z[0])
    before = [q for q in pts if q[0] <= t]; after = [q for q in pts if q[0] >= t]
    if before and after:
        t0, w0 = before[-1]; t1, w1 = after[0]
        if t1 == t0: return w0, f"observed at t={t0:.1f}s"
        if t1 - t0 <= max_gap:
            f = (t - t0) / (t1 - t0)
            return w0 + f * (w1 - w0), f"interpolated between t={t0:.1f}s and t={t1:.1f}s"
        near = (t0, w0) if (t - t0) <= (t1 - t) else (t1, w1)
        return near[1], (f"observed at t={near[0]:.1f}s; not seen again for {t1-t0:.0f}s, "
                         f"too large a gap to interpolate")
    q = (before or after)[-1 if before else 0]
    return q[1], f"nearest observation, t={q[0]:.1f}s (question is outside its observed span)"

def retrieve3d(mem, qp, t, dt=6.0, topn=4):
    """Rank by 3D distance between each memory object's nearest-in-time observation and the lifted
    question point."""
    scored = []
    for o in mem["objects"]:
        best = None
        for p in o.get("trajectory", []):
            ts = p.get("time_s")
            if ts is None or abs(ts-t) > dt: continue
            w = p.get("world_egoloc") or p.get("world_vggt")
            if not w: continue
            d = float(np.linalg.norm(np.array(w, float)-qp))
            if best is None or d < best: best = d
        if best is not None: scored.append((best, o))
    scored.sort(key=lambda z: z[0])
    # a pseudo-score so downstream fields keep the same shape: closer -> higher
    return [(round(1.0/(1.0+d), 3), True, o) for d, o in scored[:topn]]

def retrieve(mem, box, t, dt=6.0, topn=4):
    """2D overlap at the anchor time; falls back to objects present near that time."""
    scored = []
    for o in mem["objects"]:
        best_iou, near = 0.0, False
        for p in o.get("trajectory", []):
            ts = p.get("time_s")
            if ts is None or abs(ts - t) > dt: continue
            near = True
            if box: best_iou = max(best_iou, iou(box, p["box_2d"]))
        if best_iou > 0 or near:
            scored.append((best_iou, near, o))
    scored.sort(key=lambda z: (-z[0], not z[1]))
    return scored[:topn]

_STOP = {"the","a","an","is","are","was","were","did","do","does","i","me","my","you","your","it",
         "in","on","at","of","to","from","and","or","that","this","these","those","what","which",
         "where","when","how","many","much","there","he","she","they","them","his","her","its",
         "closer","nearer","far","near","more","less","than","after","before","while","during",
         "object","objects","thing","things","room","video","see","seen","saw","put","take","took"}

def _terms(txt):
    return {w for w in re.findall(r"[a-z]+", (txt or "").lower()) if len(w) > 2 and w not in _STOP}

def retrieve_text(mem, qtext, t, dt=6.0, topn=4, name_w=3.0):
    """Box-free ranking for questions with no image anchor (UCS-Bench): content words shared with the
    object's name / description, plus a time-proximity term."""
    qt = _terms(qtext)
    scored = []
    for o in mem["objects"]:
        name = " ".join(filter(None, [o.get("primary_tag") or o.get("tag") or "",
                                      " ".join(o.get("secondary_tags") or [])]))
        desc = " ".join(str((sg or {}).get("description") or "") for sg in (o.get("segments") or []))
        desc += " " + str(o.get("summary") or "")
        n_hit = len(qt & _terms(name)); d_hit = len(qt & _terms(desc))
        near, best_dt = False, 1e9
        for pp in (o.get("trajectory") or []):
            ts = pp.get("time_s")
            if ts is None: continue
            best_dt = min(best_dt, abs(ts - t))
            if abs(ts - t) <= dt: near = True
        sc = name_w * n_hit + d_hit + (2.0 if near else 0.0) + 1.0 / (1.0 + best_dt / 60.0)
        if n_hit or d_hit or near:
            scored.append((round(float(sc), 3), near, o, best_dt))
    scored.sort(key=lambda z: (-z[0], z[3]))
    return scored[:topn]

def cam_at(pe, t, fps, clip_ratio, max_dt=6.0):
    """Camera pose (R, t, centre) at the clip frame nearest time t. None if unposed."""
    gp, cp = pe["good_poses"], pe["camera_poses"]
    want = int(round(t * fps / clip_ratio))
    best = None
    for cf in range(max(0, want - int(max_dt*fps/clip_ratio)), min(len(gp), want + int(max_dt*fps/clip_ratio) + 1)):
        if not gp[cf]: continue
        if best is None or abs(cf - want) < abs(best - want): best = cf
    if best is None: return None
    E = np.array(cp[best], float); R = E[:3, :3]; tv = E[:3, 3]
    return {"cf": best, "R": R, "t": tv, "C": -R.T @ tv}

def ego_of(cam, world, unit="m"):
    """Object position in the wearer's (camera) frame: OpenCV convention, +Z forward, +X right, +Y down."""
    if cam is None or world is None: return None
    c = cam["R"] @ np.asarray(world, float) + cam["t"]
    fwd = "ahead" if c[2] >= 0 else "BEHIND you"
    lr  = "right" if c[0] >= 0 else "left"
    ud  = "below" if c[1] >= 0 else "above"
    return (f"from where you stood: {abs(c[2]):.2f} {unit} {fwd}, {abs(c[0]):.2f} {unit} to your {lr}, "
            f"{abs(c[1]):.2f} {unit} {ud} eye level")


# ------------------------------------------------------------------ rendering
def render(o, max_seg=8):
    segs = o.get("segments") or []
    if len(segs) > max_seg:
        idx = np.linspace(0, len(segs) - 1, max_seg).astype(int); segs = [segs[i] for i in idx]
    if not segs:                      # no descriptions -> fall back to raw trajectory geometry
        tr = o.get("trajectory") or []
        pts = [f"t={p.get('time_s')}s at 3D {p.get('world_egoloc') or p.get('world_vggt')}" for p in tr[:8]]
        return (f"OBJECT: {o.get('name') or o.get('primary_tag') or o.get('tag')}\n"
                f"seen {o.get('time_range_s')} s over {len(tr)} observations\n  " + "\n  ".join(pts))
    lines = [f"OBJECT: {o.get('name') or o.get('primary_tag') or o.get('tag')}"
             + (f" (also seen as {', '.join(o.get('secondary_tags') or [])})" if o.get('secondary_tags') else "")]
    if o.get("summary"): lines.append(f"summary: {o['summary']}")
    lines.append(f"seen from {o.get('time_range_s')} s; {o.get('n_segments', len(segs))} distinct positions")
    for s in segs:
        lines.append(f"  - t={s['time_range_s']}s at 3D {s['world']}: {s.get('description') or '(no description)'}")
    return "\n".join(lines)

def cand_line(rank, sc, o, box=None, mode="iou", unit="m", ego=None):
    """One candidate, described in the units it was ranked by."""
    segs = o.get("segments") or []
    tr = o.get("trajectory") or []
    area = None
    if box: area = int(((box[2]-box[0])*(box[3]-box[1]))**0.5)
    if mode == "d3":
        signal = f"| 3D distance from the question's location = {1.0/max(sc,1e-9)-1.0:.2f} m"
    else:
        signal = f"| box-overlap with the question box = {sc:.2f}"
    ctr3 = o.get("world_egoloc_median") or o.get("world_vggt_median")
    if ctr3: signal += f" | its centre over its whole trajectory = ({ctr3[0]:.2f}, {ctr3[1]:.2f}, {ctr3[2]:.2f}) {unit}"
    if ego is not None: signal += " | " + ego
    return (f"  [{rank}] {o.get('name') or o.get('primary_tag') or o.get('tag')} "
            + signal
            + (f" | its box is ~{area}px across" if area else "")
            + f" | seen at {len(tr)} moments over {o.get('time_range_s')}s"
            + f" | {len(segs)} distinct positions")

def _nearest_boxes(hits, t, dt):
    out = []
    for h in hits:
        bp = None
        for p in (h[2].get("trajectory") or []):
            ts = p.get("time_s")
            if ts is None or t is None or abs(ts - t) > dt: continue
            if bp is None or abs(ts - t) < abs(bp["time_s"] - t): bp = p
        out.append(bp["box_2d"] if bp is not None else None)
    return out

def events_context(events, qtext, t, n_recent=EVENTS_RECENT, n_match=EVENTS_MATCH):
    """The most recent event lines up to t, plus those sharing the most content words with the question."""
    ev = [x for x in (events or []) if t is None or x["t"] <= t + 1e-6]
    if not ev: return None
    keep = set(range(max(0, len(ev) - n_recent), len(ev)))
    qt = _terms(qtext); T = [_terms(x["text"]) for x in ev]
    scored = sorted(((len(qt & T[i]), i) for i in range(len(T))), key=lambda z: (-z[0], -z[1]))
    for sc, i in scored:
        if sc <= 0 or len(keep) >= n_recent + n_match: break
        keep.add(i)
    return "\n".join(f"  t={ev[i]['t']:.0f}s: {ev[i]['text']}" for i in sorted(keep))


# ------------------------------------------------------------------ UCS-Bench prompt
def ucs_prompt(q, mem, poses=None, relations=None, places=None, events=None):
    """q: {question, choices, anchor_t}; mem: object_memory.json; poses: {good_poses, camera_poses,
    scale_source}; relations/places: sidecars; events: [{t, text}].  Returns (prompt, trace)."""
    F = UCS_FEAT
    t = q.get("anchor_t"); tt = t if t is not None else 0.0
    if F["causal"] and t is not None: mem = SC.snapshot_memory(mem, t, UCS_FPS, UCS_CLIP_RATIO)
    hits = retrieve_text(mem, q["question"], tt, DT, TOPN)
    scaled = bool(poses) and poses.get("scale_source") in ("wilddet3d_pair_lsq", "da3_metric")
    UNIT = "m" if scaled else "scene units"
    CAM = cam_at(poses, tt, UCS_FPS, UCS_CLIP_RATIO, DT) if poses else None
    EGO = lambda cam, world: SC.ego_text(cam, world, UNIT, F["ego_labels"], F["ego_elev_deg"], ego_of)
    qtext = BBOX_RE.sub("the highlighted object", q["question"])
    if hits:
        boxes = _nearest_boxes(hits, t, DT)
        cands = "\n".join(cand_line(i+1, h[0], h[2], boxes[i], "text", UNIT,
                                     EGO(CAM, (pos_at(h[2], tt)[0] if CAM else None)))
                           for i, h in enumerate(hits))
        detail = "\n\n".join(f"[{i+1}] " + render(h[2]) for i, h in enumerate(hits))
        how = ("ranked by how well each object's name and description match the question's wording, "
               "plus whether it was observed near the question's moment -- this benchmark's questions "
               "carry no image region, so there is NO geometric link between question and candidate")
        qsize = None                  # UCS-Bench questions carry no image region
        ctx = (f"The question points at a region of the frame roughly {qsize}px across"
               f"{' at t=' + str(t) + 's' if t is not None else ''}. "
               f"Candidates are {how}.\n\n"
               f"CANDIDATE OBJECTS from memory near that place and time:\n{cands}\n\n"
               f"THEIR STORED HISTORIES:\n{detail}")
    else:
        ctx = "(no candidate objects in memory near that time)"
    opts = "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(q["choices"]))
    ep = SC.ego_path_until(poses, tt, UCS_FPS, UCS_CLIP_RATIO) if poses else None
    evctx = events_context(events, qtext, t) if events else None
    cand_ids = [h[2].get("id") for h in hits if h[2].get("id") is not None]
    relctx = SC.relations_context(relations, cand_ids, t, F["relations_max"], F["causal"]) if F["relations"] else None
    plctx = SC.places_context(places, cand_ids, t, F["places_max_objects"], F["causal"]) if F["places"] else None
    feat_notes = (
        ("- 'at your front-left (-35 deg)' is the direction from you, in the same words the options "
         "use: 0 deg straight ahead, +90 right, -90 left, 180 behind.\n" if F["ego_labels"] != "numeric" else "")
        + ("- RELATIONS were measured within single frames, from your viewpoint at that moment.\n" if F["relations"] else "")
        + ("- PLACES come from a scene label on each sampled frame; 'seen in' means observed while "
           "you were in that place.\n" if F["places"] else ""))
    unit_note = ("Distances are in METRES.\n" if scaled else
                 "Distances are in SCENE UNITS, not metres: this video's metric scale could not "
                 "be recovered, so relative comparisons (which is nearer, which direction) are "
                 "meaningful but absolute sizes are NOT. Do not judge whether something is "
                 "within arm's reach from these numbers.\n")
    prompt = (
        "You answer questions about a first-person video using ONLY a stored object "
        "memory. You cannot watch the video.\n\n"
        "HOW TO READ THE MEMORY:\n"
        "- It samples the video about every 4 seconds, so an exact timestamp is usually "
        "absent. Never refuse for that reason; interpolate between the surrounding entries.\n"
        "- Each object lists the distinct positions it occupied over time. Two entries at "
        "the SAME time are two different objects, not one object moving.\n"
        "- To COUNT things, count distinct objects in the memory, not the number of "
        "observations: one object seen in ten frames is still one object.\n"
        "- 'from where you stood' is measured in YOUR frame at the moment the question "
        "asks about: ahead/behind, left/right, above/below eye level. It is the CAMERA's "
        "orientation, so a sideways glance is not a sideways step, and head tilt can "
        "rotate left/right and above/below.\n"
        + feat_notes
        + f"- {unit_note}"
        + (f"\nYOUR OWN PATH through the video (waypoints): {ep}\n" if ep else "")
        + (f"\nRELATIONS BETWEEN OBJECTS (from your viewpoint, as of the question time):\n{relctx}\n" if relctx else "")
        + (f"\n{plctx}\n" if plctx else "")
        + (f"\nEVENTS — what you were doing, one line per sampled moment (every 4 s): the most recent ones and "
           f"those matching the question, up to the question time. Use them for what you did, what happened and where "
           f"things were moved; use the objects below for where things are.\n{evctx}\n" if evctx else "")
        + f"\nCANDIDATE OBJECTS — the question is about ONE of these; decide which:\n{ctx}\n\n"
        + f"QUESTION: {qtext}\n\nOPTIONS:\n{opts}\n\n"
        "Reply in exactly these three sections:\n"
        "IDENTIFICATION: which memory object the question is about, and why you rejected "
        "the others (two sentences).\n"
        "REASONING: weigh the options against what the memory actually records. Say which "
        "options the memory rules out and which it supports (three to five sentences). If "
        "the memory does not contain what the question needs, say so here -- but still "
        "commit to your best option below.\n"
        f"ANSWER: the single letter ({LET[0]}-{LET[len(q['choices'])-1]}) of your chosen "
        "option, nothing else on that line. Never leave this blank and never decline; "
        "if the memory is silent, pick the most plausible option anyway.")
    trace = {"retrieved": [h[2].get("name") or h[2].get("primary_tag") or h[2].get("tag") for h in hits],
             "units": UNIT, "ego_available": CAM is not None}
    return prompt, trace

# ------------------------------------------------------------------ HD-EPIC prompt
def hdepic_prompt(q, mem):
    """q: {question (with <BBOX>/<TIME>), choices, question_point_3d | None}; mem: object_memory.json.
    question_point_3d is the question's box lifted into the room frame (same lift as the memory)."""
    box, t = parse_anchor(q["question"])
    tt = t if t is not None else 0.0
    qp = q.get("question_point_3d")
    if box is not None and qp is not None:
        hits = retrieve3d(mem, np.array(qp, float), tt, DT, TOPN); mode = "d3"
    else:
        hits = retrieve(mem, box, tt, DT, TOPN); mode = "iou"
    qtext = BBOX_RE.sub("the highlighted object", q["question"])
    rephrased = note = None
    if box is not None:
        side = int(((box[2]-box[0])*(box[3]-box[1]))**0.5)
        if qp is not None:
            rephrased = BBOX_PHRASE.sub(f"at world coordinates ({qp[0]:.2f}, {qp[1]:.2f}, {qp[2]:.2f}) m", q["question"])
            note = (f"That coordinate is the CENTRE of the region the question points at, obtained by "
                    f"lifting its {side}px image box into the room's 3D frame. Only a centre is "
                    f"available for it -- no 3D extent.")
        else:
            rephrased = BBOX_PHRASE.sub(f"occupying a {side}px region of the frame", q["question"])
            note = "Its 3D position could not be recovered for this question."
    if hits:
        boxes = _nearest_boxes(hits, t, DT)
        cands = "\n".join(cand_line(i+1, h[0], h[2], boxes[i], mode, "m") for i, h in enumerate(hits))
        detail = "\n\n".join(f"[{i+1}] " + render(h[2]) for i, h in enumerate(hits))
        qsize = int(((box[2]-box[0])*(box[3]-box[1]))**0.5) if box else None
        how = ("ranked by 3D distance between each object's stored world position and the "
               "question's location in the room" if mode == "d3" else
               "ranked by 2D overlap with the question's box")
        ctx = (f"The question points at a region of the frame roughly {qsize}px across"
               f"{' at t=' + str(t) + 's' if t is not None else ''}. "
               f"Candidates are {how}.\n\n"
               f"CANDIDATE OBJECTS from memory near that place and time:\n{cands}\n\n"
               f"THEIR STORED HISTORIES:\n{detail}")
    else:
        ctx = "(no candidate objects in memory near that time)"
    opts = "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(q["choices"]))
    prompt = (
        "You answer questions about a first-person kitchen video using ONLY a stored object "
        "memory. You cannot watch the video.\n\n"
        "HOW TO READ THE MEMORY:\n"
        "- It samples the video about every 4 seconds, so an exact timestamp is usually absent. "
        "Never refuse for that reason; interpolate between the surrounding entries.\n"
        "- Each object lists the distinct positions it occupied. Consecutive distinct positions "
        "mean it MOVED; count moves by counting position changes.\n"
        "- Box-overlap is a HINT, not proof. A large fixture (oven, countertop, cabinet, sink) "
        "can overlap a small object simply by being big. Prefer a candidate whose size and "
        "behaviour match what the question implies.\n\n"
        + ((f"{note} Every position below, including the question's "
            "own, is a metric 3D coordinate in one shared room frame, so you can compare "
            "them directly in metres.\n\n") if rephrased else "")
        + f"CANDIDATE OBJECTS — the question is about ONE of these; decide which:\n{ctx}\n\n"
        + f"QUESTION: {rephrased or qtext}\n\nOPTIONS:\n{opts}\n\n"
        "Reply in exactly these three sections:\n"
        "IDENTIFICATION: which memory object the question is about, and why you rejected the "
        "others (two sentences).\n"
        "REASONING: weigh the options against what the memory actually records. Say which "
        "options the memory rules out and which it supports (three to five sentences).\n"
        "ANSWER: the single letter (A-E) of your chosen option, nothing else on that line.")
    trace = {"retrieved": [h[2].get("name") or h[2].get("primary_tag") or h[2].get("tag") for h in hits],
             "rank_mode": mode}
    return prompt, trace


# ------------------------------------------------------------------ parsing
def parse_reply(free, n_choices):
    """-> {identification, reasoning, answer, chosen (0-based index or None)}"""
    out = {}
    for key, pat in (("identification", r"IDENTIFICATION:?\s*(.*?)(?=REASONING:?|ANSWER:?|$)"),
                     ("reasoning", r"REASONING:?\s*(.*?)(?=ANSWER:?|$)"),
                     ("answer", r"ANSWER:?\s*(.*)$")):
        m2 = re.search(pat, free, re.S)
        out[key] = (m2.group(1).strip() or None) if m2 else None
    tail = out.get("answer") or free[-60:]
    m = re.search(r"\b([A-E])\b", tail.upper())
    if m:
        ci = LET.index(m.group(1)); out["chosen"] = ci if ci < n_choices else None
    else:                                   # tolerate a bare digit if it ignores the format
        m = re.search(r"\d", tail)
        out["chosen"] = int(m.group(0)) if m and int(m.group(0)) < n_choices else None
    return out
