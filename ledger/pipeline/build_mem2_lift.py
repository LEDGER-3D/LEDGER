"""Stage 3 (wilddet3d env): lift per-frame detections to 3D with BOTH pose sets, then cluster in
3D world space into objects (robust for static VQ3D objects). Per observation store:
  time_s, frame indices, box_2d, cam_3d (WildDet3D camera frame),
  world_egoloc = inv(egoloc_pose[cf]) @ cam_3d,  world_vggt = inv(vggt_pose[cf]) @ cam_3d.
Per object store both median world positions + the timestamped observation list (for nearest/interp
lookup at query time). Output objects_3d.json per clip.
"""
import argparse, os, sys, json, glob
import numpy as np, cv2

import ledger_paths as LP
DATA=LP.EGO4D
VQ3D=LP.VQ3D
EGL=LP.EGOLOC_POSES
FULLSCALE=DATA+"/vq3d_subset/full_scale"

def pose4x4(m):
    m=np.array(m,dtype=float)
    return None if m.shape!=(3,4) else np.vstack([m,[0,0,0,1]])
def box_to_rle(x1,y1,x2,y2,H,W):
    from pycocotools import mask as mask_util
    m=np.zeros((H,W),dtype=np.uint8,order="F")
    m[max(0,int(y1)):min(H,int(y2)),max(0,int(x1)):min(W,int(x2))]=1
    return mask_util.encode(np.asfortranarray(m))
def world(pose3x4,cam):
    T=pose4x4(pose3x4)
    if T is None: return None
    w=np.linalg.inv(T)@np.append(cam,1.0); return [round(float(v),3) for v in (w[:3]/w[3])]

def cluster(obs, radius=0.6):
    """greedy per-tag 3D clustering on primary world pos (egoloc else vggt)."""
    clusters=[]
    for o in sorted(obs,key=lambda x:-(x.get("score") or 0)):
        p=o.get("world_egoloc") or o.get("world_vggt")
        if p is None: continue
        p=np.array(p); best=None;bd=radius
        for c in clusters:
            d=np.linalg.norm(np.array(c["centroid"])-p)
            if d<bd: bd=d; best=c
        if best is None:
            clusters.append({"centroid":p.tolist(),"obs":[o]})
        else:
            best["obs"].append(o)
            ps=np.array([ (x.get("world_egoloc") or x.get("world_vggt")) for x in best["obs"]])
            best["centroid"]=np.median(ps,0).tolist()
    return clusters

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-root", default=os.path.join(LP.WORK, "memories"))
    p.add_argument("--vggt-poses", default=None)
    p.add_argument("--dataset", default="ucs", choices=["ucs"])
    p.add_argument("--poses", default=None, help="hdepic/ucs: EgoLoc-shaped pose json from the adapter")
    p.add_argument("--rect-size", type=int, default=1802, help="hdepic: virtual pinhole size for the fisheye rectification")
    p.add_argument("--only-clip", default=None)
    p.add_argument("--clips-file", default=None, help="file with clip uids (one/line) to lift; loads model once")
    p.add_argument("--frames-grid", default=None, help="dir of pre-extracted grid JPEGs <clip>/<clip_frame:06d>.jpg (fast frame source)")
    p.add_argument("--scene-lift-budget", type=int, default=None,
                   help="max scene-tag detections lifted per clip (ego4d default 220; hdepic scales per-frame)")
    p.add_argument("--max-per-tag", type=int, default=None,
                   help="cap query-title dets lifted per tag (ego4d default 40; hdepic scales per-frame)")
    p.add_argument("--radius", type=float, default=0.6)
    p.add_argument("--checkpoint", default=None)
    a=p.parse_args()
    sys.path.insert(0,LP.WILDDET3D)
    from demo.tracking.inference import load_model, run_inference_single_frame
    HDE=(a.dataset=="hdepic"); UCS=(a.dataset=="ucs")
    if UCS:
        # UCS-Bench ships no poses and no intrinsics. Both come from FastVGGT via
        # `ucs_adapter.py poses`: per-video K (median over the grid, mapped back to native pixels)
        # and world->cam extrinsics in the video's own VGGT frame, chunk-chained.
        #
        # SCALE: those poses are up to scale while WildDet3D's cam_3d is metric, and world =
        # inv(pose)@cam mixes the two. world_vggt written here is therefore only PROVISIONAL --
        # `ucs_adapter.py scale` solves the one scalar per video from these very cam_3d values and
        # rewrites both the poses and every world position. Run it before the trajectory collapse.
        #
        # No rectification: UCS-Bench ships no calibration, so there is no fisheye model to invert.
        # Frames are used as-is with VGGT's pinhole K -- for the EgoLife/Aria 1408x1408 videos that
        # is a linear approximation of a fisheye camera and its lift is correspondingly worse.
        sys.path.insert(0,LP.CODE)
        import ucs_adapter as UA
        egl={}; vggt=json.load(open(a.poses or f"{UA.M}/ucs_poses.json"))
        scan_intr={}; clip_scan={}
    elif HDE:
        # HD-EPIC: single metric pose source (Aria MPS). Aria RGB is FISHEYE624 -- there is no pinhole
        # K -- so each frame is rectified to a virtual linear camera and its boxes mapped alongside;
        # detections and GT stay in RAW fisheye space. Verified: GT 3D reprojects inside the mapped
        # GT bbox 38/38 on the pilot (raw-space baseline 38/38).
        sys.path.insert(0,LP.CODE)
        import hdepic_adapter as HA
        egl=json.load(open(a.poses or f"{HA.__file__.rsplit('/',1)[0]}/hdepic_poses.json"))
        vggt={}; scan_intr={}; clip_scan={}
        _rect={}
        def rectifier(cu):
            if cu not in _rect: _rect[cu]=HA.Rectifier(cu,size=a.rect_size)
            return _rect[cu]
    else:
        egl=json.load(open(EGL)); vggt=json.load(open(a.vggt_poses))
        scan_intr=json.load(open(VQ3D+"/data/scan_to_intrinsics.json"))
        clip_scan={}
        for f in glob.glob("outputs_sam3_auto_vq3d/*/vq3d_questions.json"):
            for q in json.load(open(f))["questions"]:
                if q.get("scan_uid"): clip_scan[q["clip_uid"]]=q["scan_uid"]
    model=load_model(checkpoint=a.checkpoint,device="cuda"); print("WildDet3D ready")
    if a.only_clip: clips=[a.only_clip]
    elif getattr(a,"clips_file",None): clips=[l.strip() for l in open(a.clips_file) if l.strip()]
    else: clips=sorted(d for d in os.listdir(a.out_root) if os.path.isfile(os.path.join(a.out_root,d,"detections.json")))
    for cu in clips:
        if os.path.isfile(os.path.join(a.out_root,cu,"objects_3d.json")):
            print(f"  {cu[:8]}: objects_3d exists; skip"); continue
        cdir=os.path.join(a.out_root,cu); dj=json.load(open(os.path.join(cdir,"detections.json")))
        vid=dj["video_uid"]; off=dj["offset"]; Wv,Hv=dj["res"]; titles_l=set(t.lower() for t in dj["titles"])
        if UCS:
            su=None; RC=None; Wl,Hl=Wv,Hv
            _K=(vggt.get(cu,{}) or {}).get("K")
            if _K is None: print(f"  {cu[:8]}: no VGGT intrinsics; run `ucs_adapter.py poses` first; skip"); continue
            K=np.array(_K,dtype=np.float32)
        elif HDE:
            su=None; RC=rectifier(cu); K=RC.K().astype(np.float32)
            Wl=Hl=RC.size                      # lift/mask space = the virtual linear camera
        else:
            su=clip_scan.get(cu); Kd=(scan_intr.get(su,{}) or {})
            Kd=Kd.get(f"('{Wv}', '{Hv}')") or Kd.get("('1440', '1080')") or (list(Kd.values())[0] if Kd else None)
            if Kd is None: print(f"  {cu[:8]}: no intrinsics; skip"); continue
            K=np.array([[Kd["f"],0,Kd["cx"]],[0,Kd["f"],Kd["cy"]],[0,0,1]],dtype=np.float32)
            RC=None; Wl,Hl=Wv,Hv
        gp_e=egl.get(cu,{}).get("good_poses"); cp_e=egl.get(cu,{}).get("camera_poses")
        gp_v=vggt.get(cu,{}).get("good_poses"); cp_v=vggt.get(cu,{}).get("camera_poses")
        # Budget resolution. ego4d keeps its historical constants; hdepic scales with THIS video's grid
        # (0.2-74.8 min), because a global top-N starves long videos and biases survivors toward large
        # high-scoring fixtures. Explicit CLI values always win.
        if HDE or UCS:
            _b=(HA if HDE else UA).lift_budgets(len(dj.get("grid_clip_frames") or []))
            scene_budget = a.scene_lift_budget if a.scene_lift_budget is not None else _b["scene_lift_budget"]
            per_tag      = a.max_per_tag       if a.max_per_tag       is not None else _b["max_per_tag"]
        else:
            scene_budget = a.scene_lift_budget if a.scene_lift_budget is not None else 220
            per_tag      = a.max_per_tag       if a.max_per_tag       is not None else 40
        dets=dj["detections"]
        # budget: query-title dets capped per tag (evenly spaced by frame) + top-scored scene dets
        qd=[]
        for t in titles_l:
            td=sorted([d for d in dets if d["tag"].lower()==t],key=lambda d:d["clip_frame"])
            if len(td)>per_tag: td=[td[i] for i in np.linspace(0,len(td)-1,per_tag).astype(int)]
            qd+=td
        sd=sorted([d for d in dets if d["tag"].lower() not in titles_l],key=lambda d:-d["score"])[:scene_budget]
        keep=qd+sd
        # Fast frame source: multiple dets share a clip_frame, so cache per cf; read the pre-extracted
        # grid JPEG (frames_grid/<clip>/<cf>.jpg) instead of random-seeking the mp4 (~5-10x faster).
        # Falls back to a video seek if the JPEG is missing -> identical result, never wrong frame.
        FG=f"{a.frames_grid}/{cu}" if getattr(a,"frames_grid",None) else None
        cap=None; by_tag={}
        # Group dets by clip_frame: read+preprocess each frame ONCE and lift all its boxes in a single
        # run_inference_single_frame call (it maps obj_id->box_3d), instead of once per det. Cuts the
        # per-frame preprocess/decode ~5x (that CPU cost was throttling parallel shards); model forward
        # count is unchanged, so 3D results are identical.
        from collections import defaultdict as _dd
        byframe=_dd(list)
        for d in keep: byframe[d["clip_frame"]].append(d)
        for cf,ds in byframe.items():
            fv0=ds[0]["frame_video"]
            rgb=None
            if FG:
                fp=f"{FG}/{cf:06d}.jpg"
                if os.path.isfile(fp):
                    b=cv2.imread(fp)
                    if b is not None: rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
            if rgb is None:
                if cap is None: cap=cv2.VideoCapture(UA.video_path(vid) if UCS else
                                                     (HA.video_path(vid) if HDE else f"{FULLSCALE}/{vid}.mp4"))
                cap.set(cv2.CAP_PROP_POS_FRAMES,fv0); ok,b=cap.read()
                if ok: rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
            if rgb is None: continue
            if HDE:
                rgb=RC.rectify(rgb)                       # fisheye -> virtual pinhole (maps cached per video)
                boxes={}
                for i in range(len(ds)):
                    rb=RC.raw_box_to_rect(ds[i]["box"])
                    if rb is not None: boxes[i]=rb
                if not boxes: continue
                obj_masks={i:box_to_rle(*boxes[i],Hl,Wl) for i in boxes}
                categories={i:ds[i]["tag"] for i in boxes}
            else:
                obj_masks={i:box_to_rle(*ds[i]["box"],Hl,Wl) for i in range(len(ds))}
                categories={i:ds[i]["tag"] for i in range(len(ds))}
            outs=run_inference_single_frame(model=model,frame_rgb=rgb,intrinsics=K,
                    obj_masks=obj_masks,categories=categories)
            out_by_id={o["track_id"]:o for o in (outs or [])}
            for i,d in enumerate(ds):
                o=out_by_id.get(i)
                if o is None: continue
                cam=np.array(o["box_3d"][:3],dtype=np.float64)
                we=world(cp_e[cf],cam) if (gp_e and 0<=cf<len(gp_e) and gp_e[cf]) else None
                wv=world(cp_v[cf],cam) if (gp_v and 0<=cf<len(gp_v) and gp_v[cf]) else None
                if we is None and wv is None: continue
                by_tag.setdefault(d["tag"],[]).append({"frame_video":d["frame_video"],"clip_frame":cf,"time_s":d["time_s"],
                    "box_2d":d["box"],"score":round(d["score"],3),"cam_3d":[round(float(v),3) for v in cam],
                    "world_egoloc":we,"world_vggt":wv,"mask_rle":d.get("mask_rle")})   # mask_rle: temp, for tracker collapse
        if cap is not None: cap.release()
        # store FLAT per-tag observations (clustering is done cheaply at agent time, re-tunable
        # without re-running the expensive WildDet3D lift).
        objs=[]; oid=0
        for tag,obs in by_tag.items():
            O=sorted(obs,key=lambda o:o["time_s"] if o["time_s"] is not None else 0)
            we=np.array([o["world_egoloc"] for o in O if o["world_egoloc"]])
            wv=np.array([o["world_vggt"] for o in O if o["world_vggt"]])
            objs.append({"id":oid,"tag":tag,"is_query_title":tag.lower() in titles_l,"n_obs":len(O),
                "world_egoloc_median":([round(v,3) for v in np.median(we,0)] if len(we) else None),
                "world_vggt_median":([round(v,3) for v in np.median(wv,0)] if len(wv) else None),
                "observations":O})
            oid+=1
        mem={"clip_uid":cu,"video_uid":vid,"scan_uid":su,"res":[Wv,Hv],
             "world_frame":(("ucs DA3 nested metric frame; world=inv(pose)@cam; K from DA3; METRES"
                             if (vggt.get(cu,{}) or {}).get("backend")=="da3" else
                             "ucs FastVGGT per-video frame; world=inv(pose)@cam; K from VGGT; "
                             "PROVISIONAL SCALE until `ucs_adapter.py scale` runs") if UCS else
                            "hdepic aria MPS metric world; world=inv(pose)@cam; pose=inv(T_world_device@T_device_camera); "
                            "boxes RAW fisheye, lift via virtual pinhole" if HDE else
                            "matterport mp (Rz90); world=inv(pose)@cam; two pose sources"),
             "dataset":a.dataset,"objects":objs}
        json.dump(mem,open(os.path.join(cdir,"objects_3d.json"),"w"),indent=2)
        nq=sum(1 for o in objs if o["is_query_title"])
        print(f"[{cu[:8]}] lifted {len(keep)}/{len(dets)} dets over {len(byframe)} frames "
              f"-> {len(objs)} objects ({nq} query-title) [budget={scene_budget}, per_tag={per_tag}]")

if __name__=="__main__":
    main()
