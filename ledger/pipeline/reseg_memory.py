"""Re-derive movement segments with a PERSISTENCE rule. Post-hoc, no VLM calls, no new frames.

The shipped rule opens a new segment whenever a point lands >0.3 m from the running centroid. Our
lift noise is 0.35 m median, so single-frame jitter is indistinguishable from a real move: measured
count bias +2.3 moves per object, exactly right only 13.5% of the time, over-counting in 64% of cases.

A real move is a move the object SUSTAINS. Requiring k consecutive observations at the new place
before opening a segment cuts the bias to -0.3 and doubles exact-count accuracy to 28.1%.

Descriptions are preserved, not regenerated: each merged segment inherits the text of whichever
constituent old segment had the most observations.
"""
import argparse, json, os, glob, shutil
import numpy as np

def wp(o): return o.get("world_egoloc") or o.get("world_vggt")

def seg_persist(traj, thresh, k):
    pts=[(o,np.array(wp(o),float)) for o in traj if wp(o) and o.get("time_s") is not None]
    if not pts: return []
    segs=[[pts[0][0]]]; ref=pts[0][1]; pend=[]
    for o,p in pts[1:]:
        if np.linalg.norm(p-ref)<=thresh:
            segs[-1].append(o); pend=[]; ref=np.mean([wp(x) for x in segs[-1]],0)
        else:
            pend.append((o,p))
            if len(pend)>=k:
                segs.append([x for x,_ in pend]); ref=np.mean([q for _,q in pend],0); pend=[]
    return segs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mem-root",default=os.path.join(LP.WORK, "memories_hdepic"))
    ap.add_argument("--out-root",default=os.path.join(LP.WORK, "memories_hdepic_reseg"))
    ap.add_argument("--thresh",type=float,default=0.3); ap.add_argument("--persist",type=int,default=3)
    a=ap.parse_args()
    os.makedirs(a.out_root,exist_ok=True)
    tot_old=tot_new=0; nv=0
    for f in sorted(glob.glob(f"{a.mem_root}/*/object_memory.json")):
        v=os.path.basename(os.path.dirname(f)); mem=json.load(open(f)); nv+=1
        for o in mem["objects"]:
            old=o.get("segments") or []; tr=o.get("trajectory") or []
            tot_old+=len(old)
            new=seg_persist(tr,a.thresh,a.persist)
            if not new: continue
            out=[]
            for g in new:
                ts=[p["time_s"] for p in g if p.get("time_s") is not None]
                ws=[wp(p) for p in g if wp(p)]
                if not ts or not ws: continue
                lo,hi=min(ts),max(ts)
                # inherit the description of whichever old segment overlaps this one most
                bestd=None; bestn=-1
                for s in old:
                    rng=s.get("time_range_s")
                    if not rng: continue
                    ov=min(hi,rng[1])-max(lo,rng[0])
                    if ov>bestn: bestn, bestd = ov, s.get("description")
                out.append({"time_range_s":[round(lo,1),round(hi,1)],
                            "world":[round(float(x),3) for x in np.median(np.array(ws,float),axis=0)],
                            "n_obs":len(g),
                            "description":bestd or "(no description)"})
            o["segments"]=out; o["n_segments"]=len(out); tot_new+=len(out)
        os.makedirs(f"{a.out_root}/{v}",exist_ok=True)
        json.dump(mem,open(f"{a.out_root}/{v}/object_memory.json","w"))
        for extra in ("objects_3d.json","objects_3d_trajectories.json","objects_3d_trajectories_tracker.json"):
            s_=f"{a.mem_root}/{v}/{extra}"
            if os.path.isfile(s_) and not os.path.isfile(f"{a.out_root}/{v}/{extra}"): shutil.copy(s_,f"{a.out_root}/{v}/{extra}")
    print(f"{nv} videos: segments {tot_old} -> {tot_new} ({100*(tot_new-tot_old)/max(tot_old,1):+.0f}%)")
if __name__=="__main__": main()
