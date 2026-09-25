"""Answering-time helpers for UCS memory: the as-of-t snapshot, discrete egocentric direction labels,
and relation / place context from the sidecar files. Read-only -- nothing here writes memory.

Used by the UCS-Bench reader (ledger/answer/reader.py).
"""
import math, os, json, collections
import numpy as np

SECTORS = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]

def sector_label(right, down, ahead, elev_deg=15.0):
    """Camera-frame vector -> (8-way horizontal sector, azimuth deg, vertical label, distance).
    Azimuth 0 = straight ahead, +90 right, -90 left, +-180 behind; 45-degree sectors centred on those.
    The angle is unchanged by any unknown global scale, so labels are as valid on unscaled videos."""
    az = math.degrees(math.atan2(right, ahead))
    sec = SECTORS[int(((az + 22.5) % 360) // 45)]
    el = math.degrees(math.atan2(-down, math.hypot(right, ahead)))    # camera +y points DOWN
    vert = "above eye level" if el > elev_deg else ("below eye level" if el < -elev_deg else "about eye level")
    return sec, az, vert, math.sqrt(right * right + down * down + ahead * ahead)

def ego_text(cam, world, unit, mode, elev_deg, numeric_fn):
    """mode numeric = the existing text, unchanged; sector8 = label only; both = label + one distance."""
    if mode == "numeric": return numeric_fn(cam, world, unit)
    if cam is None or world is None: return None
    c = cam["R"] @ np.asarray(world, float) + cam["t"]
    if not np.all(np.isfinite(c)): return None          # undefined position (seen on stitched streams whose pose chain broke)
    sec, az, vert, dist = sector_label(c[0], c[1], c[2], elev_deg)
    txt = f"from where you stood: at your {sec} ({az:+.0f} deg), {vert}"
    return txt if mode == "sector8" else f"{txt}, {dist:.2f} {unit} away"

def snapshot_memory(mem, t, fps, clip_ratio):
    """The memory as it stood at time t. Nothing observed after t; positions and medians from
    observations <= t; segments starting after t dropped and the one spanning t cut at t; a segment
    description written from a frame after t is withheld (and its position replaced by the last
    observation <= t); object summaries dropped, because they were synthesised from every segment
    including future ones."""
    if t is None: return mem
    objs = []
    for o in mem["objects"]:
        tr = [p for p in (o.get("trajectory") or []) if p.get("time_s") is not None and p["time_s"] <= t]
        if not tr: continue
        q = dict(o); q["trajectory"] = tr; q["n_obs"] = len(tr)
        q["time_range_s"] = [tr[0]["time_s"], tr[-1]["time_s"]]
        for key, fld in (("world_egoloc_median", "world_egoloc"), ("world_vggt_median", "world_vggt")):
            if key in o:
                W = [p[fld] for p in tr if p.get(fld)]
                q[key] = [round(float(v), 3) for v in np.median(np.array(W, float), 0)] if W else None
        if "segments" in o:
            segs = []
            for s in o.get("segments") or []:
                rng = s.get("time_range_s") or [None, None]
                if rng[0] is None or rng[0] > t: continue
                s2 = dict(s); s2["time_range_s"] = [rng[0], min(rng[1], t) if rng[1] is not None else t]
                rcf = s.get("rep_clip_frame")
                if rcf is not None and rcf * clip_ratio / fps > t:
                    s2["description"] = None
                    inside = [p for p in tr if p["time_s"] >= rng[0] and (p.get("world_egoloc") or p.get("world_vggt"))]
                    if inside: s2["world"] = inside[-1].get("world_egoloc") or inside[-1].get("world_vggt")
                segs.append(s2)
            q["segments"] = segs; q["n_segments"] = len(segs)
        if "summary" in o: q["summary"] = None
        objs.append(q)
    m2 = dict(mem); m2["objects"] = objs
    return m2

def ego_path_until(pe, t, fps, clip_ratio, n=6):
    """Wearer waypoints restricted to frames at or before t (the causal twin of ego_path)."""
    gp, cp = pe["good_poses"], pe["camera_poses"]
    pts = []
    for cf in range(len(gp)):
        if not gp[cf] or (t is not None and cf * clip_ratio / fps > t): continue
        E = np.array(cp[cf], float); pts.append((cf, -E[:3, :3].T @ E[:3, 3]))
    if len(pts) < 2: return None
    idx = np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)
    return "; ".join(f"t={pts[i][0]*clip_ratio/fps:.0f}s ({pts[i][1][0]:.1f}, {pts[i][1][1]:.1f}, "
                     f"{pts[i][1][2]:.1f})" for i in idx)

def load_sidecar(mem_root, video, name):
    p = f"{mem_root}/{video}/{name}"
    return json.load(open(p)) if os.path.isfile(p) else None

_PRI = {"on": 0, "under": 0, "left of": 1, "right of": 1, "in front of": 1, "behind": 1,
        "above": 2, "below": 2, "near": 3}

def relations_context(rel, cand_ids, t, max_lines=10, causal=True, gap_s=8.0):
    """Relations among the retrieved candidates as of time t, plus what each candidate was resting
    ON even if that support object was not retrieved ('the spoon is on the bed')."""
    if not rel or not cand_ids: return None
    ids = set(cand_ids); best = {}
    for e in rel["episodes"]:
        if causal and t is not None and e["t0"] > t: continue
        inside = e["a"] in ids and e["b"] in ids
        support = e["rel"] == "on" and (e["a"] in ids or e["b"] in ids)
        if not (inside or support): continue
        k = (e["a"], e["b"], e["rel"])
        if k not in best or e["t0"] > best[k]["t0"]: best[k] = e
    if not best: return None
    def end(e): return min(e["t1"], t) if (causal and t is not None) else e["t1"]
    rows = sorted(best.values(), key=lambda e: (_PRI.get(e["rel"], 9), -end(e)))
    out = []
    for e in rows[:max_lines]:
        still = t is None or end(e) >= t - gap_s
        when = f"since {e['t0']:.0f}s" if still else f"{e['t0']:.0f}-{end(e):.0f}s, not seen since"
        out.append(f"  - {e['a_name']} is {e['rel']} {e['b_name']} ({when})")
    return "\n".join(out)

def places_context(pl, cand_ids, t, max_objects=12, causal=True):
    """Where the wearer is, the ordered places visited so far, what has been seen in the current place,
    and where each candidate was seen -- all as of time t."""
    if not pl: return None
    visits = [v for v in pl["visits"] if not causal or t is None or v["t0"] <= t]
    if not visits: return None
    label = {p["place_id"]: p["label"] for p in pl["places"]}
    cur = visits[-1]
    tend = lambda v: min(v["t1"], t) if (causal and t is not None) else v["t1"]
    lines = [f"WHERE YOU ARE at the question time: {cur['label']} (since {cur['t0']:.0f}s)"]
    shown = visits[-12:]
    lines.append("PLACES YOU HAVE BEEN, in order" + (" (most recent 12)" if len(visits) > 12 else "") + ": "
                 + " -> ".join(f"{v['label']} ({v['t0']:.0f}-{tend(v):.0f}s)" for v in shown))
    op = pl.get("object_places", {})
    here = sorted(((e["n_obs"], oid) for oid, lst in op.items() for e in lst
                   if e["place_id"] == cur["place_id"] and (not causal or t is None or e["first_s"] <= t)), reverse=True)
    if here:
        # distinct objects that share a name are listed once with a count ("chair x3"): the memory keeps
        # them as separate objects, and a counting question needs that number, not a repeated word
        cnt = collections.Counter(pl["name"].get(oid) or oid for _, oid in here)
        names = [f"{n} x{c}" if c > 1 else n for n, c in cnt.most_common(max_objects)]
        more = f" (+{len(cnt) - max_objects} more kinds)" if len(cnt) > max_objects else ""
        lines.append(f"OBJECTS SEEN IN {cur['label'].upper()} SO FAR: {', '.join(names)}{more}")
    cl = []
    for oid in cand_ids:
        lst = [e for e in op.get(str(oid), []) if not causal or t is None or e["first_s"] <= t]
        if lst:
            cl.append(f"  - {pl['name'].get(str(oid)) or oid}: seen in " + ", ".join(label[e["place_id"]] for e in lst))
    if cl: lines.append("WHERE EACH CANDIDATE WAS SEEN:\n" + "\n".join(cl))
    return "\n".join(lines)

# ---------------------------------------------------------------------------------------------------
# v4 answering helpers (`--profile ucs2`): geometry read DIRECTLY from each observation's camera-frame
# position (WildDet3D's metric cam_3d) instead of through the world poses. On UCS-Bench the poses come
# from FastVGGT/DA3 at a 4 s stride and are unreliable on most EgoLife/TeleEgo videos (no metric scale
# could be recovered for 92 of 116 finished memories), while every question refers to objects seen
# within seconds of the question time -- so "where was it, relative to you, when you last saw it" is
# both metric and answerable without any pose. Nothing here touches HD-EPIC / VQ3D answering.

_GENERIC = {"one", "ones", "thing", "things", "item", "items", "object", "objects", "kind", "kinds",
            "type", "types", "gray", "grey", "black", "white", "red", "blue", "green", "yellow", "brown",
            "big", "small", "large", "little", "left", "right", "front", "back", "today", "now", "still",
            "just", "again", "already", "yet", "total", "room", "here", "there"}

# the answering reader's stop words (ledger/answer/reader.py _STOP)
_STOP = {"the","a","an","is","are","was","were","did","do","does","i","me","my","you","your","it",
         "in","on","at","of","to","from","and","or","that","this","these","those","what","which",
         "where","when","how","many","much","there","he","she","they","them","his","her","its",
         "closer","nearer","far","near","more","less","than","after","before","while","during",
         "object","objects","thing","things","room","video","see","seen","saw","put","take","took"}

def cam_terms(txt):
    import re
    return {w for w in re.findall(r"[a-z]+", (txt or "").lower()) if len(w) > 2 and w not in _STOP}

def last_obs(o, t, max_age=None):
    """The latest trajectory point at or before t (the snapshot already drops later ones)."""
    best = None
    for p in (o.get("trajectory") or []):
        ts = p.get("time_s")
        if ts is None or (t is not None and ts > t) or not p.get("cam_3d"): continue
        if best is None or ts > best["time_s"]: best = p
    if best is None or (max_age is not None and t is not None and t - best["time_s"] > max_age): return None
    return best

def direct_obs_text(o, t, elev_deg=15.0, max_age=None, stale_s=20.0):
    """'seen 2 s before the question: 1.4 m from you, at your front-left (-30 deg), below eye level'."""
    p = last_obs(o, t, max_age)
    if p is None: return None
    c = p["cam_3d"]; sec, az, vert, dist = sector_label(c[0], c[1], c[2], elev_deg)
    age = (t - p["time_s"]) if t is not None else None
    if age is not None and age > stale_s:
        # The wearer walks on between sightings; a distance measured a minute ago says nothing about
        # NOW, and reading it as current was the main error of the first v4 run (deer at 1.2 m seen
        # 54 s earlier chosen over penguins at 2.5 m seen 14 s earlier).
        return (f"LAST SEEN {age:.0f} s BEFORE the question ({dist:.1f} m from you, at your {sec}, at that time) "
                f"-- STALE: you have most likely moved since, so this is NOT its current distance or direction")
    when = ("at the question time" if age is not None and age < 1.0 else
            f"{age:.0f} s before the question" if age is not None else f"at t={p['time_s']:.0f}s")
    return f"DIRECTLY OBSERVED {when}: {dist:.1f} m from you, at your {sec} ({az:+.0f} deg), {vert}"

def render_direct(o, t, elev_deg=15.0, max_seg=8):
    """Object history for the prompt with camera-relative geometry per segment (metric, pose-free)
    instead of world coordinates in unknown units."""
    segs = o.get("segments") or []
    tr = [p for p in (o.get("trajectory") or []) if p.get("time_s") is not None]
    if len(segs) > max_seg:
        idx = np.linspace(0, len(segs) - 1, max_seg).astype(int); segs = [segs[i] for i in idx]
    name = o.get("name") or o.get("primary_tag") or o.get("tag")
    lines = [f"OBJECT: {name}" + (f" (also seen as {', '.join(o.get('secondary_tags') or [])})" if o.get("secondary_tags") else "")]
    if o.get("summary"): lines.append(f"summary: {o['summary']}")
    lines.append(f"seen from {o.get('time_range_s')} s over {len(tr)} observations")
    def geo_at(ts):
        p = min((q for q in tr if q.get("cam_3d")), key=lambda q: abs(q["time_s"] - ts), default=None)
        if p is None: return ""
        sec, az, vert, dist = sector_label(*p["cam_3d"], elev_deg)
        return f" [{dist:.1f} m from you, {sec} ({az:+.0f} deg), {vert}, as seen at t={p['time_s']:.0f}s]"
    if segs:
        for s in segs:
            rng = s.get("time_range_s") or [None, None]
            ts = rng[0] if rng and rng[0] is not None else None
            lines.append(f"  - t={rng}s{geo_at(ts) if ts is not None else ''}: {s.get('description') or '(no description)'}")
    else:
        for p in tr[-6:]:
            lines.append(f"  - t={p['time_s']}s{geo_at(p['time_s'])}")
    return "\n".join(lines)

def pair_phrases(question):
    """'Which is closer to me, the gray bear or the black one?' -> ['the gray bear', 'the black one'].
    Only comparison shapes ('X or Y') are split; anything else returns []."""
    import re
    q = question.strip().rstrip("?").strip()
    m = re.search(r"(?:,|:|between|among)\s*(.+?)\s+or\s+(.+)$", q, re.I)
    if not m: m = re.search(r"\b(?:is|was|are|were)\s+(.+?)\s+or\s+(.+?)\s+(?:closer|nearer|farther|further|bigger|taller|higher|lower|larger|smaller)\b", q, re.I)
    if not m: return []
    a, b = m.group(1).strip(), m.group(2).strip()
    b = re.sub(r"\s+(closer|nearer|farther|further|to me|from me|to you|now|right now).*$", "", b, flags=re.I).strip()
    return [x for x in (a, b) if len(cam_terms(x) - _GENERIC) > 0 or len(x) > 2]

def count_context(mem, question, t, max_age=60.0):
    """For 'how many' questions: how many memory objects match the counted noun, how many of them were
    ever visible in ONE frame (a floor on the true count that duplicate tracks cannot inflate), and how
    many were seen in the last max_age seconds. Returns None when the question does not count objects
    or nothing matches."""
    import re, collections
    q = question.lower()
    if not re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q): return None
    terms = cam_terms(question) - _GENERIC
    if not terms: return None
    def obj_terms(o):
        return cam_terms(" ".join(filter(None, [o.get("name") or o.get("primary_tag") or o.get("tag") or "",
                                                 " ".join(o.get("secondary_tags") or [])])))
    hits = collections.defaultdict(list)                       # term -> objects whose NAME carries it
    for o in mem["objects"]:
        ot = obj_terms(o)
        for w in terms:
            if w in ot or (w.endswith("s") and w[:-1] in ot) or (w + "s") in ot: hits[w].append(o)
    if not hits: return None
    out = []
    for w, objs in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        byf = collections.defaultdict(set); recent = set()
        for o in objs:
            for p in (o.get("trajectory") or []):
                ts = p.get("time_s")
                if ts is None or (t is not None and ts > t): continue
                byf[p.get("clip_frame")].add(o["id"])
                if t is None or t - ts <= max_age: recent.add(o["id"])
        if not byf: continue
        fmax, smax = max(byf.items(), key=lambda kv: len(kv[1]))
        tmax = next((p["time_s"] for o in objs for p in (o.get("trajectory") or []) if p.get("clip_frame") == fmax), None)
        out.append(f"  - '{w}': {len(objs)} separate tracks in memory up to now (the same physical object seen "
                   f"again from elsewhere can appear as a new track, so this OVER-counts); at most {len(smax)} were "
                   f"visible together in one frame{f' (t={tmax:.0f}s)' if tmax is not None else ''}; "
                   f"{len(recent)} tracks seen in the last {max_age:.0f} s")
    if not out: return None
    return ("COUNTING AID (from the memory's tracks, as of the question time):\n" + "\n".join(out) +
            "\n  Read these as bounds: the true number of distinct objects is at least the one-frame maximum "
            "and usually well below the number of tracks.")


# ---------------------------------------------------------------- scene cuts (UCS profile option, 2026-09-22)
# A long stream may contain a change of scene (another room, another building, another recording). Causal retrieval then
# ranks name-matching objects of the EARLIER scene among the candidates ("cross-scene leakage": on the stitched mixed
# streams 29% of candidates, memory accuracy at blind level). scene_start() gives the start of the scene the wearer is in
# at time t from a cuts sidecar (stitch/big/scene_cuts.py: visual no-revisit test + place partition; the stitching
# registry is never used), scene_filter() keeps only the objects observed in that scene.
def load_scene_cuts(cuts_dir, video):
    p = f"{cuts_dir}/{video}.cuts.json" if cuts_dir else None
    if not p or not os.path.isfile(p): return None
    return sorted(float(c) for c in json.load(open(p)).get("cuts_s") or [])

def scene_start(cuts, t):
    if not cuts or t is None: return 0.0
    return max([c for c in cuts if c <= t], default=0.0)

def scene_filter(mem, t0):
    """Objects with at least one observation at or after t0 (the current scene); trajectories cut at t0."""
    if not t0: return mem
    objs = []
    for o in mem["objects"]:
        tr = [p for p in (o.get("trajectory") or []) if p.get("time_s") is not None and p["time_s"] >= t0]
        if not tr: continue
        q = dict(o); q["trajectory"] = tr; q["n_obs"] = len(tr); q["time_range_s"] = [tr[0]["time_s"], tr[-1]["time_s"]]
        if "segments" in o:
            q["segments"] = [s for s in (o.get("segments") or []) if (s.get("time_range_s") or [None, None])[1] is None
                             or s["time_range_s"][1] >= t0]
        objs.append(q)
    return {**mem, "objects": objs}
