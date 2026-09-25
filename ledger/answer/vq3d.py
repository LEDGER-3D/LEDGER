"""VQ3D (Ego4D) from the object memory: "Where is the <object>?" -> a 3D position in the scan frame.

The LLM sees every memory object of the clip (label, times seen, 3D position) and picks the one the
query names; if it picks nothing valid, an exact-name match is the fallback. The answer is the median
3D position of the picked object's observations.
"""
import json
import numpy as np


def load_cands(mem):
    """objects_3d_trajectories_tracker.json -> candidates with a 3D position."""
    out = []
    for o in mem.get("objects", []):
        o = dict(o)
        o["names"] = [o.get("primary_tag", "")] + list(o.get("secondary_tags") or [])
        o["observations"] = o.get("trajectory") or []
        if o.get("world_egoloc_median") or o.get("world_vggt_median"): out.append(o)
    return out


def match_prompt(title, cl):
    lines = [f"[{c['idx']}] label='{c['tag']}' source={c['source']} seen={c['n_obs']}x "
             f"world_xyz={c['world']}" for c in cl]
    return ("You match a visual-query object to one object in a scene memory.\n"
            f"QUERY: \"Where is the {title}?\" (find the object that best corresponds to '{title}').\n"
            "MEMORY CANDIDATES (id, label, times-seen, 3D world position):\n"
            + "\n".join(lines) +
            "\n\nPick the single candidate id that is the queried object. Labels may be synonyms "
            "(e.g. 'tool' can be a 'screw driver'). Respond with STRICT JSON: "
            '{\"id\": <int or -1>, \"why\": \"<short reason>\"}.')


def parse_id(out):
    try:
        s = out[out.index("{"): out.rindex("}") + 1]
        return int(json.loads(s).get("id", -1))
    except Exception:
        digits = "".join(ch for ch in out if (ch.isdigit() or ch == "-"))
        return int(digits) if digits.lstrip("-").isdigit() else -1


def answer(llm, title, mem):
    """-> (xyz or None, record)"""
    cands = load_cands(mem)
    cl = [{"idx": i, "tag": " / ".join(n for n in o["names"] if n),
           "world": o.get("world_egoloc_median") or o.get("world_vggt_median"), "n_obs": o["n_obs"],
           "source": "scene"} for i, o in enumerate(cands)]
    prompt = match_prompt(title, cl)
    reply = llm.ask(prompt, 320)
    ci = parse_id(reply)
    exact = [o for o in cands if any((n or "").lower() == title.lower() for n in o["names"])]
    pick = cands[ci] if 0 <= ci < len(cands) else (exact[0] if exact else None)
    rec = {"reply": reply, "picked": (" / ".join(n for n in pick["names"] if n) if pick else None),
           "how": "llm" if 0 <= ci < len(cands) else "exact-name fallback", "n_candidates": len(cands)}
    if pick is None: return None, rec
    W = [np.array(o["world_egoloc"]) for o in pick["observations"] if o.get("world_egoloc")]
    xyz = np.median(W, 0).tolist() if W else None
    rec["n_obs_of_pick"] = len(W)
    return xyz, rec


def score(pred, gt):
    """Official-convention metrics. gt = {annotation_1: {x,y,z}, annotation_2: {x,y,z}} in the scan frame;
    the annotation is rotated by Rz(90) into the pose frame: (x, y, z) -> (y, -x, z).
      L2      = ||pred - annotation_1||
      success = ||pred - mid(a1, a2)|| < 6 * (||a1 - a2|| + 1)"""
    mp = lambda a: np.array([a["y"], -a["x"], a["z"]], float)
    g1, g2 = mp(gt["annotation_1"]), mp(gt["annotation_2"])
    p = np.asarray(pred, float)
    return {"L2_m": round(float(np.linalg.norm(p - g1)), 3),
            "success": bool(np.linalg.norm((g1 + g2) / 2 - p) < 6 * (np.linalg.norm(g1 - g2) + 1.0))}
