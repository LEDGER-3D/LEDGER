"""FastVGGT-based, COLMAP-free camera poses for VQ3D, placed into the GT/mp frame by a
per-clip robust Sim3 to the EgoLoc reference camera centers (user-chosen anchor).

Pipeline per clip:
  1. sample frames = union(all query response-track frames for the clip) + uniform context,
     capped; map clip-frame -> video-frame (video = 6*clip + offset, offset from the RT).
  2. FastVGGT forward -> per-frame extrinsics (world->cam, VGGT frame).  [VGGT depth unused;
     WildDet3D does the metric lift.]
  3. VGGT camera centers C_v = -R^T t.  EgoLoc centers C_e = -R_e^T t_e (poses are world->cam).
  4. robust Sim3 (RANSAC umeyama) C_v -> C_e over sampled frames with EgoLoc good_poses.
  5. transform each VGGT extrinsic into the GT frame:
        R_gt = R_v @ R_s^T ;  C_gt = s*R_s@C_v + t_s ;  pose = [R_gt | -R_gt@C_gt]  (world->cam)
  6. emit {good_poses, camera_poses} per clip (drop-in for vq3d_validate3d.py --poses).

Output frame == EgoLoc/mp frame, so grade with the SAME protocol as the table.
"""
import argparse, os, sys, json, glob, shutil
import numpy as np, cv2, torch

import ledger_paths as LP
DATA=LP.EGO4D
EGL_PATH=LP.EGOLOC_POSES
FULLSCALE=DATA+"/vq3d_subset/full_scale"
QDIR=os.path.join(LP.WORK, "sam3_auto_vq3d")

def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Sc,Dc=src-mu_s,dst-mu_d
    Sig=(Dc.T@Sc)/len(src); U,D,Vt=np.linalg.svd(Sig); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; var=(Sc**2).sum()/len(src); s=np.trace(np.diag(D)@S)/var; t=mu_d-s*R@mu_s
    return s,R,t
def ap(s,R,t,X): return (s*(R@X.T).T+t)
def ransac_sim3(src,dst,it=2000,thr=0.3,seed=0):
    rng=np.random.RandomState(seed); n=len(src); best=(0,None)
    for _ in range(it):
        i=rng.choice(n,3,replace=False)
        try: s,R,t=umeyama(src[i],dst[i])
        except: continue
        r=np.linalg.norm(ap(s,R,t,src)-dst,axis=1); inl=r<thr
        if inl.sum()>best[0]: best=(inl.sum(),inl)
    inl=best[1] if best[1] is not None else np.ones(n,bool)
    s,R,t=umeyama(src[inl],dst[inl]); return s,R,t,inl

def center_w2c(m):  # world->cam [R|t] -> camera center -R^T t
    m=np.array(m,dtype=float); R=m[:3,:3]; t=m[:3,3]; return -R.T@t

def load_vggt(merging, device="cuda"):
    sys.path.insert(0,LP.FASTVGGT)  # FastVGGT's vggt pkg first
    from vggt.models.vggt import VGGT
    m=VGGT.from_pretrained("facebook/VGGT-1B", merging=merging, merge_ratio=0.9).to(device).eval()
    return m

def vggt_extrinsics(model, paths, device="cuda"):
    # square (center-pad) 518x518 so every image has an identical 37x37 patch grid ->
    # FastVGGT token-merging assertion (tokens_per_img = w*h+5) holds for all clips/aspect ratios.
    # We use only the extrinsics (WildDet3D lifts with scan intrinsics), so square-padding is harmless.
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dt=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
    images=load_and_preprocess_images_square(paths, target_size=518)[0].to(device)  # (imgs, coords)
    # FastVGGT's token-merge reads static per-block patch_width/height (default 37x28); the plain
    # aggregator() call never updates them. Set them from the actual square grid (518//14 = 37).
    pw=ph=images.shape[-1]//14
    for blk in list(model.aggregator.global_blocks)+list(model.aggregator.frame_blocks):
        if hasattr(blk,"attn"): blk.attn.patch_width=pw; blk.attn.patch_height=ph
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dt):
        agg,ps=model.aggregator(images[None])
        pose_enc=model.camera_head(agg)[-1]
        extr,intr=pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    return extr.squeeze(0).float().cpu().numpy()   # (N,3,4) world->cam, VGGT frame

def main():
    ap_=argparse.ArgumentParser()
    ap_.add_argument("--only-clip", default=None, help="pilot: process just this clip_uid")
    ap_.add_argument("--merging", type=int, default=6)
    ap_.add_argument("--context", type=int, default=48, help="uniform context frames per clip")
    ap_.add_argument("--max-frames", type=int, default=110)
    ap_.add_argument("--min-anchors", type=int, default=8)
    ap_.add_argument("--include-memory-frames", action="store_true",
                     help="also sample the memory-observation clip-frames (for the agent re-grade)")
    ap_.add_argument("--grid-stride", type=int, default=0,
                     help="if >0: BLIND grid mode -- sample clip-frames at this stride (matches memory tracker grid)")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--work", default=os.path.join(LP.WORK, "tmp_vggt_frames"))
    a=ap_.parse_args()

    egl=json.load(open(EGL_PATH))
    # gradable clips + their query response-track frames, from vq3d_questions + vq_val
    vqv=json.load(open(DATA+"/annotations/vq_val.json"))
    rt_by={}; clip2vid={}
    for v in vqv["videos"]:
        for c in v["clips"]:
            clip2vid[c["clip_uid"]]=v["video_uid"]
            for ann in c["annotations"]:
                for qk,qs in ann["query_sets"].items():
                    rt=qs.get("response_track") or []
                    if rt: rt_by.setdefault((c["clip_uid"],qs["object_title"]),[]).append(
                        [(int(r["frame_number"]), int(r["video_frame_number"])) for r in rt])
    # clips we need = clips present in the question files
    clips={}
    for f in glob.glob(f"{QDIR}/*/vq3d_questions.json"):
        for q in json.load(open(f))["questions"]:
            clips.setdefault(q["clip_uid"], []).append(q)
    todo=[a.only_clip] if a.only_clip else sorted(clips.keys())

    # optional: memory-observation clip-frames per clip (so the agent re-grade has VGGT poses there)
    mem_frames={}
    if a.include_memory_frames:
        for mf in glob.glob(f"{QDIR}/*_memory3d/objects_3d.json"):
            for o in json.load(open(mf)).get("objects",[]):
                cu=o.get("clip")
                for ob in (o.get("observations") or []):
                    mem_frames.setdefault(cu,set()).add(int(ob["clip_frame"]))

    model=load_vggt(a.merging); print(f"[vggt] model loaded (merging={a.merging})")
    out={}
    for cu in todo:
        if cu not in egl or cu not in clips:
            print(f"  {cu[:8]}: skip (not in egl/questions)"); continue
        qs=clips[cu]; N=len(egl[cu]["good_poses"]); gpe=egl[cu]["good_poses"]; cpe=egl[cu]["camera_poses"]
        vid=clip2vid[cu]
        # offset (video = 6*clip + offset) from any RT of this clip
        offset=0
        for (c2,t2),trks in rt_by.items():
            if c2==cu:
                cf,vf=trks[0][0]; offset=vf-6*cf; break
        if a.grid_stride>0:
            # BLIND grid: sample clip-frames at a fixed stride (no GT/RT), matching the memory tracker grid
            frames=sorted(set(range(0, N, a.grid_stride)))
            if len(frames)>a.max_frames:
                frames=[frames[i] for i in np.linspace(0,len(frames)-1,a.max_frames).astype(int)]
        else:
            # MUST-KEEP: last-seen (max RT) of each query -- the harness lifts here
            need=set()
            for q in qs:
                for cc in rt_by.get((cu,q["object_title"]),[]):
                    need.add(max([cf for cf,_ in cc]))
            # OPTIONAL pool (subsamplable): RT-track frames + memory-observation frames + uniform context
            optional=set()
            for q in qs:
                for cc in rt_by.get((cu,q["object_title"]),[]):
                    fns=[cf for cf,_ in cc]; optional.update(fns[::max(1,len(fns)//4)])
            optional |= {f for f in mem_frames.get(cu,set()) if 0<=f<N}
            goodidx=[f for f in range(N) if gpe[f]]
            if goodidx:
                optional |= {goodidx[i] for i in np.linspace(0,len(goodidx)-1,min(a.context,len(goodidx))).astype(int)}
            optional -= need
            budget=max(0, a.max_frames-len(need))
            opt=sorted(optional)
            if len(opt)>budget and budget>0:
                opt=[opt[i] for i in np.linspace(0,len(opt)-1,budget).astype(int)]
            frames=sorted(need | set(opt))
        # extract video frames
        work=os.path.join(a.work,cu[:8]);
        if os.path.isdir(work): shutil.rmtree(work)
        os.makedirs(work)
        cap=cv2.VideoCapture(f"{FULLSCALE}/{vid}.mp4"); paths=[]; kept=[]
        for cf in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 6*cf+offset); ok,b=cap.read()
            if ok:
                p=os.path.join(work,f"{cf:06d}.jpg"); cv2.imwrite(p,b); paths.append(p); kept.append(cf)
        cap.release()
        if len(kept)<a.min_anchors+3:
            print(f"  {cu[:8]}: only {len(kept)} frames extracted, skip"); continue
        extr=vggt_extrinsics(model, paths)              # (n,3,4) VGGT world->cam, order == kept
        Cv={cf:center_w2c(extr[i]) for i,cf in enumerate(kept)}
        # Sim3 anchors: sampled frames with EgoLoc good pose
        anc=[cf for cf in kept if cf<N and gpe[cf]]
        if len(anc)<a.min_anchors:
            print(f"  {cu[:8]}: only {len(anc)} EgoLoc anchors, skip"); continue
        src=np.array([Cv[cf] for cf in anc]); dst=np.array([center_w2c(cpe[cf]) for cf in anc])
        s,R,t,inl=ransac_sim3(src,dst); res=np.linalg.norm(ap(s,R,t,src)-dst,axis=1)
        # emit poses (transform every sampled VGGT extrinsic into GT frame)
        good=[False]*N; cps=[np.zeros((3,4)).tolist() for _ in range(N)]
        for i,cf in enumerate(kept):
            Rv=extr[i][:3,:3]; Cvi=Cv[cf]
            R_gt=Rv@R.T; C_gt=ap(s,R,t,Cvi[None])[0]; t_gt=-R_gt@C_gt
            M=np.zeros((3,4)); M[:3,:3]=R_gt; M[:3,3]=t_gt
            if cf<N: cps[cf]=M.tolist(); good[cf]=True
        out[cu]={"good_poses":good,"camera_poses":cps}
        print(f"  {cu[:8]}: frames={len(kept)} anchors={len(anc)} inliers={int(inl.sum())} "
              f"Sim3 scale={s:.3f} residual med={np.median(res[inl]):.3f}m")
    json.dump(out, open(a.out,"w"))
    print(f"saved -> {a.out}  ({len(out)} clips)")

if __name__=="__main__":
    main()
