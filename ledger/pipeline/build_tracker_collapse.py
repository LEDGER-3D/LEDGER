"""Tracker collapse (stage after the corrected 3D collapse). Within each LLM-grouped similar-name set
(e.g. {tire,tyre,wheel,wheel rim}), merge post-3D-collapse instances that are the SAME physical object:
  - order instances by first-occurrence timestamp;
  - iterative seeding: seed the SAM3 tracker with the earliest instance's REPRESENTATIVE (raw) frame +
    mask, propagate over the OTHER instances' representative (raw) frames; any whose propagated mask
    overlaps its own mask (IoU>=tau) merges with the seed; repeat on leftovers;
  - co-occurrence guard: never merge two instances that share a frame (both visible => distinct);
  - merge trajectories ordered by timestamp; store primary+secondary names.
Writes objects_3d_trajectories_tracker.json + instances_tracker_collapse/. Needs the SAM3 tracker (transformers).
"""
import argparse, os, sys, json, re, shutil
import numpy as np, cv2, httpx
from PIL import Image
import ledger_paths as LP
DATA=LP.EGO4D
FULLSCALE=DATA+"/vq3d_subset/full_scale"; M=LP.WORK
def safe(t): return re.sub(r"[^a-z0-9]+","_",t.strip().lower()).strip("_")
def decode(rle):
    from pycocotools import mask as mu
    r=dict(rle); r["counts"]=r["counts"].encode("ascii") if isinstance(r["counts"],str) else r["counts"]
    return mu.decode(r).astype(bool)
def _wpos(p): return p.get("world_egoloc") or p.get("world_vggt")
def closest_in_time_dist(tr_a, tr_b):
    """Min 3D separation between two instances at their nearest approach IN TIME.
    Same physical object seen twice must have been *somewhere* continuous, so the closest pair of
    observations in time is the strongest evidence: if it is metres apart, the two instances are two
    different objects (e.g. two identical counters, two fridges) that merely look alike to a mask
    tracker. Dataset-agnostic -- uses whichever world frame the lift populated."""
    A=[(p.get("time_s"),_wpos(p)) for p in tr_a if _wpos(p) and p.get("time_s") is not None]
    B=[(p.get("time_s"),_wpos(p)) for p in tr_b if _wpos(p) and p.get("time_s") is not None]
    if not A or not B: return None
    best=None
    for ta,pa in A:
        tb,pb=min(B,key=lambda z:abs(z[0]-ta))
        d=float(np.linalg.norm(np.array(pa,float)-np.array(pb,float)))
        cand=(abs(tb-ta),d)
        if best is None or cand[0]<best[0]: best=cand
    return None if best is None else best[1]

def miou(a,b):
    i=np.logical_and(a,b).sum(); u=np.logical_or(a,b).sum(); return float(i/u) if u else 0.0
def llm_groups(tags, server=None):
    if server:
        # Local persistent Qwen server. ROBUST PARSE: a smaller model wraps JSON in ``` fences, adds
        # prose, or truncates -- a bare re.search for {...} then returns None and we silently fell back
        # to singletons (which is how bread / bread bag, chopping board / cutting board never got
        # tested). Retry, strip fences, and log the raw text so a failure is diagnosable, not silent.
        import urllib.request
        prompt=("Group these object tags into sets that refer to the SAME physical object "
                "(synonyms, or a thing and its container/packaging, e.g. 'bread'+'bread bag', "
                "'chopping board'+'cutting board', 'paper towel roll'+'paper towel holder'). "
                "Do NOT group merely similar categories (a plate and a frying pan are DIFFERENT). "
                "Every tag must appear exactly once; unrelated tags form singleton groups. "
                "Reply with ONLY compact JSON {\"groups\":[[\"tag\",...],...]} and nothing else. "
                "Tags: "+", ".join(tags))
        for attempt in range(3):
            try:
                body=json.dumps({"prompt":prompt,"max_new_tokens":3000,"no_think":True}).encode()
                rq=urllib.request.Request(server.rstrip("/")+"/generate",data=body,
                                          headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(rq,timeout=900) as r:
                    txt=json.loads(r.read())["text"]
                txt=re.sub(r"<think>.*?</think>","",txt,flags=re.S)
                txt=re.sub(r"^\s*```(?:json)?|```\s*$","",txt.strip(),flags=re.M)   # strip fences
                m=re.search(r'\{\s*"groups".*\}',txt,re.S) or re.search(r"\{.*\}",txt,re.S)
                if m is None: raise ValueError(f"no JSON object in {len(txt)} chars; tail={txt[-120:]!r}")
                g=json.loads(m.group(0))["groups"]
                g=[[str(t).strip().lower() for t in grp if str(t).strip()] for grp in g if grp]
                seen=set(); clean=[]
                for grp in g:                                  # keep only known tags, no duplicates
                    kept=[t for t in grp if t in set(tags) and t not in seen]
                    seen.update(kept)
                    if kept: clean.append(kept)
                for t in tags:                                 # any tag the model dropped -> singleton
                    if t not in seen: clean.append([t])
                ng=sum(1 for grp in clean if len(grp)>1)
                print(f"  llm_groups(server): {len(clean)} groups, {ng} multi-tag",flush=True)
                return clean
            except Exception as e:
                print(f"  llm_groups(server) attempt {attempt+1}/3: {type(e).__name__} {str(e)[:110]}",flush=True)
        print("  llm_groups(server): all attempts failed -> singletons",flush=True)
        return [[t] for t in tags]
    url=os.getenv("LLM_BASE_URL"); key=os.getenv("LLM_API_KEY"); model=os.getenv("LLM_MODEL","gpt-5.4")
    prompt=("Group these object tags into sets that refer to the SAME or overlapping physical object type "
            "(synonyms / part-whole / near-duplicates) that a mask tracker could match across frames. "
            "Unrelated objects stay in their own singleton group. STRICT JSON {\"groups\":[[\"tag\",...],...]}. "
            "Tags: "+", ".join(tags))
    body={"model":model,"input":[{"role":"user","content":[{"type":"input_text","text":prompt}]}]}
    # retry with a SHORT per-attempt timeout: the Azure deployment intermittently stalls, and a single
    # httpx.Client(timeout=120) has hung for hours. A 45s hard cap kills a stall; the next try usually lands
    # on a good call. Final fallback = singleton groups (pipeline completes; no cross-tag synonym merge).
    for k in range(4):
        try:
            r=httpx.Client(timeout=httpx.Timeout(45.0, connect=15.0)).post(
                url,headers={"api-key":key,"Content-Type":"application/json"},json=body)
            r.raise_for_status()
            txt="".join(c.get("text","") for o in r.json().get("output",[]) for c in (o.get("content") or []) if c.get("type")=="output_text")
            return json.loads(re.search(r"\{.*\}",txt,re.S).group(0))["groups"]
        except Exception as e:
            print(f"  llm_groups attempt {k+1}/4 failed: {type(e).__name__} {str(e)[:50]}",flush=True)
    print("  llm_groups: all attempts failed -> singleton groups (no synonym merge)",flush=True)
    return [[t] for t in tags]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clip",required=True); ap.add_argument("--mem-root",default=f"{M}/outputs_mem2")
    ap.add_argument("--tau",type=float,default=0.3,help="mask-IoU to call same identity")
    ap.add_argument("--merge-max-dist",type=float,default=1.0,
                    help="3D guard (metres): veto a merge whose instances are further apart than this at "
                         "their closest approach in time. 0 disables. Matches cluster_associate's 1.0 m gate.")
    ap.add_argument("--server",default=None,help="local qwen_server.py URL instead of the Azure API")
    a=ap.parse_args()
    cdir=f"{a.mem_root}/{a.clip}"
    traj=json.load(open(f"{cdir}/objects_3d_trajectories.json"))
    inst={ (im["tag"],im["cluster_id"]):im for im in json.load(open(f"{cdir}/instances.json"))["instances"] }
    det=json.load(open(f"{cdir}/detections.json")); Wv,Hv=det["res"]; vid=det["video_uid"]
    # per-object bundle: trajectory obj + its instance-manifest entry + frame set
    objs=[]
    for o in traj["objects"]:
        im=inst.get((o["tag"],o["cluster_id"]))
        if im is None: continue
        objs.append({"tag":o["tag"],"is_query":o.get("is_query_title"),"cluster_id":o["cluster_id"],
                     "traj":o["trajectory"],"n_obs":o["n_obs"],"first_t":im.get("first_time_s"),
                     "raw":im["raw"],"rep_mask":im["rep_mask_rle"],"frames":set(p["frame_video"] for p in o["trajectory"])})
    groups=llm_groups(sorted(set(o["tag"] for o in objs)), server=a.server)
    print("groups:",[g for g in groups if len(g)>1] or groups,flush=True)
    tag2grp={}
    for gi,g in enumerate(groups):
        for t in g: tag2grp[t]=gi
    import torch, track_video_sam3_fuse as fuse
    device="cuda" if torch.cuda.is_available() else "cpu"
    _,_,tmodel,tproc=fuse.load_models(device=device); print("SAM3 tracker ready",flush=True)
    tdir=f"{cdir}/instances_tracker_collapse"
    if os.path.isdir(tdir): shutil.rmtree(tdir)
    os.makedirs(tdir,exist_ok=True)
    merged=[]; mid=0
    from collections import defaultdict
    bygrp=defaultdict(list)
    for o in objs: bygrp[tag2grp.get(o["tag"],-1)].append(o)
    for gi,members in bygrp.items():
        remaining=sorted(members,key=lambda x:(x["first_t"] if x["first_t"] is not None else 1e9))
        while remaining:
            seed=remaining[0]; grp=[seed]; ious={id(seed):1.0}
            cands=[x for x in remaining[1:] if seed["frames"].isdisjoint(x["frames"])]   # co-occurrence guard
            if cands:
                session=[Image.open(f"{cdir}/{seed['raw']}").convert("RGB")]+[Image.open(f"{cdir}/{c['raw']}").convert("RGB") for c in cands]
                tracks=fuse._track_seeds(session,[{"t":0,"mask":decode(seed["rep_mask"])}],
                                         tmodel,tproc,device,fuse.DTYPE,Hv,Wv,window=len(session)+1)
                tm=tracks[0]
                for j,c in enumerate(cands,start=1):
                    iou=miou(tm[j],decode(c["rep_mask"]))
                    if iou<a.tau: continue
                    if a.merge_max_dist>0:
                        d3=closest_in_time_dist(seed["traj"],c["traj"])
                        if d3 is not None and d3>a.merge_max_dist:
                            print(f"    VETO 3D [{seed['tag']}]+[{c['tag']}] iou={iou:.3f} but {d3:.2f}m apart "
                                  f"(> {a.merge_max_dist}m)",flush=True)
                            continue
                    grp.append(c); ious[id(c)]=round(float(iou),3)
            gid=set(id(x) for x in grp); remaining=[x for x in remaining if id(x) not in gid]
            # merged object
            tr=sorted([p for x in grp for p in x["traj"]],key=lambda p:(p["time_s"] if p.get("time_s") is not None else p["frame_video"]))
            tc=defaultdict(int)
            for x in grp: tc[x["tag"]]+=x["n_obs"]
            primary=max(tc,key=tc.get); secondary=sorted(t for t in tc if t!=primary)
            we=[p["world_egoloc"] for p in tr if p.get("world_egoloc")]
            merged.append({"id":mid,"primary_tag":primary,"secondary_tags":secondary,
                "is_query":any(x["is_query"] for x in grp),"n_members":len(grp),"n_obs":len(tr),
                "members":[{"tag":x["tag"],"cluster_id":x["cluster_id"],"n_obs":x["n_obs"],"match_iou":ious.get(id(x))} for x in grp],
                "world_egoloc_median":(np.median(we,0).round(3).tolist() if we else None),
                "time_range_s":[tr[0].get("time_s"),tr[-1].get("time_s")],"trajectory":tr})
            md=f"{tdir}/{safe(primary)}__m{mid}"; os.makedirs(md,exist_ok=True)
            for x in grp:
                src=f"{cdir}/{inst[(x['tag'],x['cluster_id'])]['full']}"
                if os.path.isfile(src): shutil.copy(src,f"{md}/member_{safe(x['tag'])}_c{x['cluster_id']}.jpg")
            if len(grp)>1: print(f"  MERGED [{primary}] <- {[ (x['tag'],x['cluster_id'],ious.get(id(x))) for x in grp]}",flush=True)
            mid+=1
    json.dump({"clip_uid":a.clip,"video_uid":vid,"tau":a.tau,"merge_max_dist":a.merge_max_dist,
               "method":"tracker_collapse (SAM3 mask propagation over representative frames within LLM similar-name groups; co-occurrence guard; 3D closest-in-time distance guard; iterative seeding)",
               "n_objects":len(merged),"objects":merged},open(f"{cdir}/objects_3d_trajectories_tracker.json","w"),indent=2)
    print(f"DONE tracker collapse: {len(objs)} post-3D instances -> {len(merged)} objects ({sum(1 for m in merged if m['n_members']>1)} merged) -> {tdir}/")
if __name__=="__main__": main()
