"""Trajectory memory: keep ALL per-tag 3D clusters (no dominant-only collapse). Each (tag,cluster)
becomes ONE object whose state is a TIME-ORDERED trajectory of its observations (treated as a dynamic
object). Writes objects_3d_trajectories.json (leaves objects_3d.json intact) and regenerates
instances/ with one entry PER CLUSTER. Needs cv2 + pycocotools.
"""
import sys, argparse, json, os, re, shutil
import numpy as np, cv2
from pycocotools import mask as mask_util
import ledger_paths as LP
DATA=LP.EGO4D
FULLSCALE=DATA+"/vq3d_subset/full_scale"; M=LP.WORK
TRAJ_KEYS=["time_s","frame_video","clip_frame","box_2d","score","cam_3d","world_egoloc","world_vggt"]

def safe(t): return re.sub(r"[^a-z0-9]+","_",t.strip().lower()).strip("_")
def decode_mask(rle):
    r=dict(rle); r["counts"]=r["counts"].encode() if isinstance(r["counts"],str) else r["counts"]
    return mask_util.decode(r).astype(bool)
def wpos(o): return o.get("world_egoloc") or o.get("world_vggt")
def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    iw=max(0,x2-x1);ih=max(0,y2-y1);inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def cluster_associate(obs, radius, dedup_iou=0.7):
    """Corrected per-frame data association (per tag):
      - two detections at the SAME timestamp are DISTINCT objects -> never merged;
      - a detection associates to a PRIOR-timestamp cluster only if within `radius` of that
        cluster's LAST position (so it follows moving objects), one detection per cluster per frame
        (greedy 1-to-1); unmatched detections start new clusters;
      - high-IoU (>=dedup_iou) same-frame duplicate masks are de-duped first (keep higher score)."""
    byf={}
    for o in obs:
        if wpos(o) is not None: byf.setdefault(o["frame_video"],[]).append(o)
    clusters=[]   # {last:np.array, obs:[]}
    for fv in sorted(byf):
        dets=sorted(byf[fv],key=lambda x:-(x.get("score") or 0))
        kept=[]
        for d in dets:
            if all(box_iou(d["box_2d"],k["box_2d"])<dedup_iou for k in kept): kept.append(d)
        dets=kept
        pairs=[]                                   # (dist, det_idx, clu_idx) within radius of cluster's LAST pos
        for di,d in enumerate(dets):
            p=np.array(wpos(d),float)
            for ci,c in enumerate(clusters):
                dist=np.linalg.norm(c["last"]-p)
                if dist<radius: pairs.append((dist,di,ci))
        pairs.sort(); du=set(); cu=set(); assign={}
        for dist,di,ci in pairs:                   # greedy one-to-one
            if di in du or ci in cu: continue
            assign[di]=ci; du.add(di); cu.add(ci)
        for di,d in enumerate(dets):
            p=np.array(wpos(d),float)
            if di in assign: c=clusters[assign[di]]; c["obs"].append(d); c["last"]=p
            else: clusters.append({"last":p,"obs":[d]})
    for c in clusters: c["center"]=np.median([np.array(wpos(x)) for x in c["obs"]],0)  # summary center
    return clusters

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clip",required=True); ap.add_argument("--mem-root",default=f"{M}/outputs_mem2")
    ap.add_argument("--radius",type=float,default=1.0); ap.add_argument("--margin",type=float,default=0.15)
    ap.add_argument("--dataset",default="ucs", choices=["ucs"])
    a=ap.parse_args()
    cdir=f"{a.mem_root}/{a.clip}"
    o3=json.load(open(f"{cdir}/objects_3d.json")); objs=o3["objects"]
    vid=o3.get("video_uid") or json.load(open(f"{cdir}/detections.json"))["video_uid"]
    # ---- build trajectory objects: one per (tag, cluster) ----
    traj_objs=[]; oid=0
    for o in objs:
        for ci,c in enumerate(cluster_associate(o["observations"],a.radius)):
            tr=sorted(c["obs"],key=lambda x:(x.get("time_s") if x.get("time_s") is not None else x["frame_video"]))
            we=[wpos(x) for x in tr if x.get("world_egoloc")]; wv=[x["world_vggt"] for x in tr if x.get("world_vggt")]
            traj_objs.append({"id":oid,"tag":o["tag"],"is_query_title":bool(o.get("is_query_title")),
                "cluster_id":ci,"n_obs":len(tr),
                "world_egoloc_median":(np.median(we,0).round(3).tolist() if we else None),
                "world_vggt_median":(np.median(wv,0).round(3).tolist() if wv else None),
                "time_range_s":[tr[0].get("time_s"),tr[-1].get("time_s")],
                "trajectory":[{k:ob.get(k) for k in TRAJ_KEYS} for ob in tr]})   # mask_rle omitted (in detections.json)
            oid+=1
    json.dump({"clip_uid":a.clip,"video_uid":vid,"scan_uid":o3.get("scan_uid"),"radius":a.radius,
               "method":"per_tag_3d_cluster_trajectories (CORRECTED per-frame one-to-one association; same-frame distinct, prior-timestamp gating on last position, high-IoU dedup)",
               "n_objects":len(traj_objs),"objects":traj_objs},
              open(f"{cdir}/objects_3d_trajectories.json","w"),indent=2)
    print(f"objects_3d_trajectories.json: {len(traj_objs)} cluster-objects across {len(objs)} tags",flush=True)

    # ---- regenerate instances/ : one entry PER CLUSTER (representative frame = nearest cluster median) ----
    outdir=f"{cdir}/instances"
    if os.path.isdir(outdir): shutil.rmtree(outdir)
    os.makedirs(outdir,exist_ok=True)
    if a.dataset in ("hdepic","ucs"):
        sys.path.insert(0,LP.CODE)
        HA=__import__("hdepic_adapter" if a.dataset=="hdepic" else "ucs_adapter")
        _vp=HA.video_path(vid)
    else:
        _vp=f"{FULLSCALE}/{vid}.mp4"
    cap=cv2.VideoCapture(_vp); man=[]
    for o in objs:
        for ci,c in enumerate(cluster_associate(o["observations"],a.radius)):
            cen=np.array(c["center"]); tr=c["obs"]
            rep=min(tr,key=lambda x:np.linalg.norm(np.array(wpos(x))-cen))
            fv=int(rep["frame_video"]); cap.set(cv2.CAP_PROP_POS_FRAMES,fv); ok,frame=cap.read()
            if not ok: continue
            Hh,Ww=frame.shape[:2]; box=[int(v) for v in rep["box_2d"]]
            col=(0,215,255) if o.get("is_query_title") else (0,255,0)
            try: mk=decode_mask(rep["mask_rle"])
            except Exception: mk=None
            full=frame.copy()
            if mk is not None: full[mk]=(0.45*np.array(col)+0.55*full[mk]).astype(np.uint8)
            cv2.rectangle(full,(box[0],box[1]),(box[2],box[3]),col,3)
            q=" *" if o.get("is_query_title") else ""
            t0,t1=[round(x,1) if x is not None else None for x in (tr[0].get("time_s"),tr[-1].get("time_s"))]
            lab=f"{o['tag']}{q} c{ci} n={len(tr)} xyz=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f}) t[{t0},{t1}]s"
            (tw,th),_=cv2.getTextSize(lab,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(full,(box[0],max(box[1]-th-8,0)),(box[0]+tw+6,max(box[1],th+8)),col,-1)
            cv2.putText(full,lab,(box[0]+3,max(box[1]-6,th+2)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv2.LINE_AA)
            mx=int((box[2]-box[0])*a.margin); my=int((box[3]-box[1])*a.margin)
            x1,y1=max(box[0]-mx,0),max(box[1]-my,0); x2,y2=min(box[2]+mx,Ww),min(box[3]+my,Hh)
            thumb=frame.copy()
            if mk is not None:
                dim=(frame*0.35).astype(np.uint8); dim[mk]=frame[mk]; thumb=dim
            thumb=thumb[y1:y2,x1:x2]
            idir=f"{outdir}/{safe(o['tag'])}__c{ci}"; os.makedirs(idir,exist_ok=True)
            fp=f"{idir}/full_{fv}.jpg"; tp=f"{idir}/thumb_{fv}.jpg"; rp=f"{idir}/raw_{fv}.jpg"
            cv2.imwrite(fp,full); cv2.imwrite(tp,thumb); cv2.imwrite(rp,frame)   # raw = clean frame for the SAM3 tracker
            man.append({"tag":o["tag"],"is_query":bool(o.get("is_query_title")),"cluster_id":ci,"n_obs":len(tr),
                        "world_egoloc_center":cen.round(3).tolist(),"time_range_s":[t0,t1],
                        "first_time_s":tr[0].get("time_s"),"first_frame_video":int(tr[0]["frame_video"]),
                        "rep_frame_video":fv,"box_2d":rep["box_2d"],"rep_mask_rle":rep.get("mask_rle"),
                        "full":os.path.relpath(fp,cdir),"thumb":os.path.relpath(tp,cdir),"raw":os.path.relpath(rp,cdir)})
            print(f"[{o['tag']} c{ci}] n={len(tr)} rep f{fv} xyz=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f})",flush=True)
    cap.release()
    json.dump({"clip_uid":a.clip,"video_uid":vid,"method":"per_tag_3d_cluster (all clusters kept)",
               "radius":a.radius,"instances":man},open(f"{cdir}/instances.json","w"),indent=2)
    print(f"DONE instances/ regenerated: {len(man)} cluster-instances -> {outdir}/")
if __name__=="__main__": main()
