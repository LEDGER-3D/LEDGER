"""Place nodes: per-frame scene tags grouped into places, with every object linked to the places it
was seen in. Built during memory creation as its own stage; writes a sidecar, places.json. No
existing file, field or stage is modified.

The structure mirrors a code graph: a PLACE is a folder, the OBJECTS seen there are its files, and
the wearer's ordered VISITS are the call sequence through them. Object-place links use time
co-occurrence only -- an object belongs to a place if it was observed in frames tagged with that
place -- so they need no metric positions and are equally valid on videos whose poses never scaled.
"""
import argparse, json, os, re, sys, time, bisect, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
import ledger_paths as LP
M = LP.WORK

PROMPT = ("What kind of place or area is this first-person photo taken in? Answer with a short noun "
          "phrase of 1 to 4 words, for example: kitchen, living room, supermarket aisle, checkout "
          "counter, office corridor, street, staircase. Reply with the phrase only.")
MERGE_PROMPT = ("These scene labels were produced frame by frame for ONE video. Merge labels that name "
                "the same kind of place (synonyms, singular/plural, extra adjectives) and keep genuinely "
                "different places separate. Return ONLY a JSON object mapping every input label to its "
                "canonical label.\nLabels: {labels}")

def gen(server, prompt, images=None, max_new_tokens=24, retries=4):
    body = json.dumps({"prompt": prompt, "images": images or [], "max_new_tokens": max_new_tokens,
                       "no_think": True}).encode()
    for k in range(retries):
        try:
            rq = urllib.request.Request(server.rstrip("/") + "/generate", data=body,
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=300) as r:
                return json.loads(r.read())["text"]
        except Exception:
            if k == retries - 1: raise
            time.sleep(3 + 5 * k)

def norm(label):
    s = re.sub(r"<think>.*?</think>", "", label or "", flags=re.S).strip().split("\n")[0].lower()
    s = re.sub(r"[^a-z0-9 \-/]", " ", s).strip()
    for _ in range(3):
        s = re.sub(r"^(a|an|the|this is|it is|inside|in|at|on)\s+", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return " ".join(s.split()[:4]) or "unknown"

def smooth_labels(seq, half):
    """Majority filter over a (2*half+1)-frame window: one-frame flickers are tagger noise."""
    if half <= 0 or len(seq) < 3: return list(seq)
    out = []
    for i in range(len(seq)):
        c = collections.Counter(seq[max(0, i - half): i + half + 1]).most_common()
        top = [x for x, n in c if n == c[0][1]]
        out.append(seq[i] if seq[i] in top else top[0])
    return out

def group(frames, seq, objs, fps, cr):
    tsec = lambda cf: round(cf * cr / fps, 1)
    visits = []
    for cf, l in zip(frames, seq):
        if visits and visits[-1]["label"] == l:
            visits[-1]["t1"] = tsec(cf); visits[-1]["n_frames"] += 1
        else:
            visits.append({"label": l, "t0": tsec(cf), "t1": tsec(cf), "n_frames": 1})
    labels = []
    for v in visits:
        if v["label"] not in labels: labels.append(v["label"])
    pid = {l: f"p{i}" for i, l in enumerate(labels)}
    for v in visits: v["place_id"] = pid[v["label"]]
    frame_place = {cf: pid[l] for cf, l in zip(frames, seq)}
    fl = sorted(frame_place)
    def place_of(cf):
        if cf in frame_place: return frame_place[cf]
        if not fl: return None
        i = bisect.bisect_left(fl, cf)
        near = [fl[j] for j in (i - 1, i) if 0 <= j < len(fl)]
        return frame_place[min(near, key=lambda x: abs(x - cf))]
    object_places, place_objects = {}, collections.defaultdict(collections.Counter)
    for o in objs:
        per = {}
        for p in o.get("trajectory") or []:
            cf, ts = p.get("clip_frame"), p.get("time_s")
            if cf is None or ts is None: continue
            q = place_of(cf)
            if q is None: continue
            e = per.setdefault(q, {"place_id": q, "n_obs": 0, "first_s": ts, "last_s": ts})
            e["n_obs"] += 1; e["first_s"] = min(e["first_s"], ts); e["last_s"] = max(e["last_s"], ts)
        if per:
            object_places[str(o["id"])] = sorted(per.values(), key=lambda e: e["first_s"])
            for q, e in per.items(): place_objects[q][o["id"]] += e["n_obs"]
    places = []
    for l in labels:
        q = pid[l]; vs = [v for v in visits if v["place_id"] == q]
        places.append({"place_id": q, "label": l, "n_visits": len(vs), "n_frames": sum(v["n_frames"] for v in vs),
                       "first_s": vs[0]["t0"], "last_s": vs[-1]["t1"], "visits": [[v["t0"], v["t1"]] for v in vs],
                       "objects": [oid for oid, _ in place_objects[q].most_common()]})
    return places, visits, object_places

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mem-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--grid-root", default=None)
    ap.add_argument("--tags-root", required=True, help="cache of raw per-frame scene tags")
    ap.add_argument("--server", required=True)
    ap.add_argument("--dataset", default="ucs", choices=["ucs"])
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--out-name", default="places.json")
    a = ap.parse_args()
    fps, cr, smooth = 30.0, 6, 1
    if a.dataset == "ucs":
        if M not in sys.path: sys.path.insert(0, LP.CODE)
        import ucs_adapter as UA
        fps, cr, smooth = UA.FPS, UA.CLIP_RATIO, UA.CONFIG.get("place_smooth_frames", 1)
    if a.smooth is not None: smooth = a.smooth
    servers = [s.strip() for s in a.server.split(",") if s.strip()]
    cdir = f"{a.mem_root}/{a.clip}"
    spec = f"{a.grid_root}/{a.clip}/detections.json" if a.grid_root else None
    if spec and os.path.isfile(spec):
        grid = json.load(open(spec))["grid_clip_frames"]
    else:
        grid = sorted(int(f[:-4]) for f in os.listdir(f"{a.frames_root}/{a.clip}") if f.endswith(".jpg"))
    os.makedirs(a.tags_root, exist_ok=True)
    cache_p = f"{a.tags_root}/{a.clip}.json"
    raw = json.load(open(cache_p)) if os.path.isfile(cache_p) else {}
    todo = [cf for cf in grid if str(cf) not in raw and os.path.isfile(f"{a.frames_root}/{a.clip}/{cf:06d}.jpg")]
    t0 = time.time()
    def work(i_cf):
        i, cf = i_cf
        return cf, gen(servers[i % len(servers)], PROMPT, [f"{a.frames_root}/{a.clip}/{cf:06d}.jpg"])
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, 2 * len(servers))) as ex:
            for n, (cf, txt) in enumerate(ex.map(work, enumerate(todo)), 1):
                raw[str(cf)] = txt
                if n % 25 == 0: json.dump(raw, open(cache_p, "w"))
        json.dump(raw, open(cache_p, "w"))
    print(f"[{a.clip}] scene tags: {len(todo)} new, {len(raw)} total ({time.time()-t0:.0f}s)", flush=True)

    lab = {int(k): norm(v) for k, v in raw.items()}
    frames = [cf for cf in grid if cf in lab]
    uniq = sorted(set(lab[cf] for cf in frames))
    canon = {u: u for u in uniq}
    if len(uniq) > 1:
        try:
            txt = gen(servers[0], MERGE_PROMPT.format(labels=json.dumps(uniq)), None, 1200)
            m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", txt, flags=re.S), re.S)
            mp = json.loads(m.group(0)) if m else {}
            for u in uniq:
                c = mp.get(u)
                if isinstance(c, str) and c.strip(): canon[u] = norm(c)
        except Exception as e:
            print(f"  label merge failed ({type(e).__name__}); keeping normalised labels", flush=True)
    seq = smooth_labels([canon[lab[cf]] for cf in frames], smooth)
    src = next((f"{cdir}/{fn}" for fn in ("objects_3d_trajectories_tracker.json", "objects_3d_trajectories.json")
                if os.path.isfile(f"{cdir}/{fn}")), None)
    objs = json.load(open(src))["objects"] if src else []
    places, visits, object_places = group(frames, seq, objs, fps, cr)
    out = {"schema": "places/v1", "clip_uid": a.clip, "source": os.path.basename(src) if src else None,
           "params": {"smooth_frames": smooth, "prompt": PROMPT}, "label_map": canon,
           "n_frames_tagged": len(frames), "places": places, "visits": visits,
           "object_places": object_places,
           "name": {str(o["id"]): (o.get("primary_tag") or o.get("tag")) for o in objs}}
    json.dump(out, open(f"{cdir}/{a.out_name}", "w"), indent=1)
    print(f"[{a.clip}] {len(frames)} tagged frames -> {len(places)} places, {len(visits)} visits, "
          f"{len(object_places)}/{len(objs)} objects linked", flush=True)

if __name__ == "__main__": main()
