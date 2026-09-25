"""YOLO detection memory (parallel to build_mem2_detect but with a YOLO detector). Prompts YOLO with
the SAME union canonical vocab SAM3 used, on the SAME grid frames -> 2D boxes -> detections.json
(box + box-rect mask_rle, same schema) for WildDet3D lift. Run in `cg` env with PYTHONPATH=ultra84:pycoco."""
import argparse, json, glob, os, time
import numpy as np
from pycocotools import mask as mask_util
import ledger_paths as LP
M=LP.WORK
def box_rle(x1,y1,x2,y2,H,W):
    m=np.zeros((H,W),np.uint8,order="F"); m[max(0,int(y1)):min(H,int(y2)),max(0,int(x1)):min(W,int(x2))]=1
    r=mask_util.encode(np.asfortranarray(m)); r["counts"]=r["counts"].decode("ascii"); return r
def to_cuda(m):
    try:
        m.model.cuda()
        for mod in m.model.modules():
            if getattr(mod,"txt_feats",None) is not None: mod.txt_feats=mod.txt_feats.cuda()
    except Exception: pass
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--mode",required=True,choices=["world","yoloe_text"])
    ap.add_argument("--frames-root",required=True); ap.add_argument("--tags-file",required=True)
    ap.add_argument("--sam3-root",default=f"{M}/outputs_traj_union"); ap.add_argument("--out-root",required=True)
    ap.add_argument("--conf",type=float,default=0.25); ap.add_argument("--topk",type=int,default=12)
    ap.add_argument("--only-clip",default=None)
    ap.add_argument("--imgsz",type=int,default=640,
                    help="inference size. 640 is fine for Ego4D 456x256-ish crops; HD-EPIC frames are "
                         "1408x1408 and its queried objects are small manipulated items, so 1408 + low conf "
                         "+ high topk raises recall on them (measured 6/22 -> 9/22 on HD-EPIC anchors).")
    a=ap.parse_args()
    from ultralytics import YOLO, YOLOE
    tags_by_clip=json.load(open(a.tags_file))
    clips=[a.only_clip] if a.only_clip else sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{a.sam3_root}/*/detections.json"))
    for cu in clips:
        if os.path.isfile(f"{a.out_root}/{cu}/detections.json"): print(f"[{cu[:8]}] exists; skip",flush=True); continue
        sj=f"{a.sam3_root}/{cu}/detections.json"
        if not os.path.isfile(sj): print(f"[{cu[:8]}] no SAM3 grid; skip",flush=True); continue
        s=json.load(open(sj)); vid=s["video_uid"]; off=s["offset"]; Wv,Hv=s["res"]; grid=s["grid_clip_frames"]; titles=s["titles"]
        # Frame-index convention comes from the grid spec. Ego4D/HD-EPIC specs carry neither key and
        # keep the historical 6 / 30.0; UCS-Bench is 10 fps with clip_frame == video_frame.
        cr=int(s.get("clip_ratio",6)); fps=float(s.get("fps",30.0))
        tags=tags_by_clip.get(cu,{}).get("tags",[]) or []
        if not tags: print(f"[{cu[:8]}] no union tags; skip",flush=True); continue
        # Fresh model per clip: YOLO-World's 2nd set_classes puts new CLIP text embeddings on CPU while the
        # model stays on CUDA -> device-mismatch crash. Re-instantiating (weights RAM-cached) sidesteps it.
        m=YOLO(a.model) if a.mode=="world" else YOLOE(a.model)
        if a.mode=="world": m.set_classes(tags)
        else: m.set_classes(tags, m.get_text_pe(tags))
        to_cuda(m); names=m.names
        dets=[]
        imgs=[f"{a.frames_root}/{cu}/{cf:06d}.jpg" for cf in grid]
        imgs=[(cf,p) for cf,p in zip(grid,imgs) if os.path.isfile(p)]
        import torch as _t; _t.cuda.synchronize() if _t.cuda.is_available() else None
        t0=time.perf_counter()
        for i in range(0,len(imgs),16):
            batch=imgs[i:i+16]
            res=m.predict([p for _,p in batch],conf=a.conf,verbose=False,imgsz=a.imgsz)
            for (cf,_),r in zip(batch,res):
                if r.boxes is None: continue
                cand=[]
                for c,bx,sc in zip(r.boxes.cls.tolist(),r.boxes.xyxy.tolist(),r.boxes.conf.tolist()):
                    cand.append({"tag":str(names[int(c)]),"score":float(sc),"box":[round(float(v),1) for v in bx]})
                cand=sorted(cand,key=lambda d:-d["score"])[:a.topk]
                fv=cr*cf+off
                for d in cand:
                    d["mask_rle"]=box_rle(*d["box"],Hv,Wv); d["clip_frame"]=cf; d["frame_video"]=fv; d["time_s"]=round(fv/fps,2)
                    dets.append(d)
        _t.cuda.synchronize() if _t.cuda.is_available() else None
        det_s=round(time.perf_counter()-t0,3); nfr=len(imgs)
        od=f"{a.out_root}/{cu}"; os.makedirs(od,exist_ok=True)
        json.dump({"clip_uid":cu,"video_uid":vid,"offset":off,"N":s.get("N"),"res":[Wv,Hv],
                   "clip_ratio":cr,"fps":fps,"dataset":s.get("dataset"),
                   "grid_clip_frames":grid,"titles":titles,"model":a.model,"n_tags":len(tags),
                   "detect_seconds":det_s,"n_frames":nfr,"imgsz":a.imgsz,"conf":a.conf,"topk":a.topk,"sec_per_frame":round(det_s/max(1,nfr),4),
                   "detections":dets},open(f"{od}/detections.json","w"),indent=2)
        print(f"[{cu[:8]}] {nfr}fr {len(tags)}tags -> {len(dets)} dets in {det_s}s ({round(det_s/max(1,nfr),3)}s/fr) ({a.model})",flush=True)
    print("YOLO_DETECT_DONE",flush=True)
if __name__=="__main__": main()
