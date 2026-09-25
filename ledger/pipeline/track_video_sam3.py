"""SAM3-Tracker-Video mask propagation baseline.

Takes a source image (object_in_frame.png) + seed mask (sam3_mask.png), prepends
the source as a synthetic frame 0 of the navigation video, propagates the mask
across all frames with Sam3TrackerVideo, and writes per-frame masks, overlays,
and a GIF.

Requires the patched transformers fork at
    <a transformers checkout that provides Sam3VideoModel>
which provides the SAM3 model family. Install once with:
    pip install -e <a transformers checkout that provides Sam3VideoModel>
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# --- Defaults (override from launchers) -----------------------------------------
HF_NAME = "facebook/sam3"
DTYPE = torch.bfloat16
OBJ_ID = 1
STRIDE = 1
GIF_FPS = 6
MASK_COLOR = (255, 64, 64)   # RGB tint for overlay
MASK_ALPHA = 0.55


def load_model(device, hf_name=HF_NAME, dtype=DTYPE):
    model = Sam3TrackerVideoModel.from_pretrained(hf_name).to(device, dtype=dtype)
    processor = Sam3TrackerVideoProcessor.from_pretrained(hf_name)
    model.eval()
    return model, processor


def _load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _overlay_mask(frame_rgb, mask_bool, color=MASK_COLOR, alpha=MASK_ALPHA):
    out = frame_rgb.copy()
    tint = np.zeros_like(out)
    tint[:] = color
    sel = mask_bool.astype(bool)
    out[sel] = (alpha * tint[sel] + (1 - alpha) * out[sel]).astype(np.uint8)
    contours, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return out


def process_video(video_path, mask_path, source_image_path, out_dir, label, model, processor,
                  device, dtype=DTYPE, obj_id=OBJ_ID, stride=STRIDE, gif_fps=GIF_FPS,
                  mask_color=MASK_COLOR, mask_alpha=MASK_ALPHA):
    frames_dir = os.path.join(out_dir, "frames")
    masks_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlay")
    for d in (frames_dir, masks_dir, overlay_dir):
        os.makedirs(d, exist_ok=True)

    vid_frames = _load_video_frames(video_path)
    if not vid_frames:
        print(f"[{label}] could not read frames, skipping.")
        return
    H_vid, W_vid = vid_frames[0].shape[:2]

    src_rgb = np.array(Image.open(source_image_path).convert("RGB"))
    if src_rgb.shape[:2] != (H_vid, W_vid):
        src_rgb = cv2.resize(src_rgb, (W_vid, H_vid), interpolation=cv2.INTER_AREA)

    seed_mask = np.array(Image.open(mask_path).convert("L"))
    if seed_mask.shape[:2] != (H_vid, W_vid):
        seed_mask = cv2.resize(seed_mask, (W_vid, H_vid), interpolation=cv2.INTER_NEAREST)
    seed_mask_bool = seed_mask > 0
    if seed_mask_bool.sum() == 0:
        raise RuntimeError(f"Seed mask {mask_path} is empty after binarization.")

    print(f"[{label}] video {len(vid_frames)} @ {W_vid}x{H_vid}, seed mask pixels: {int(seed_mask_bool.sum())}")

    Image.fromarray(src_rgb).save(os.path.join(frames_dir, "source.png"))
    for t, f in enumerate(vid_frames):
        Image.fromarray(f).save(os.path.join(frames_dir, f"frame_{t:03d}.jpg"))

    src_overlay = _overlay_mask(src_rgb, seed_mask_bool, color=mask_color, alpha=mask_alpha)
    Image.fromarray(src_overlay).save(os.path.join(overlay_dir, "source.png"))
    Image.fromarray((seed_mask_bool.astype(np.uint8) * 255)).save(os.path.join(masks_dir, "source.png"))

    pil_seq = [Image.fromarray(src_rgb)] + [Image.fromarray(f) for f in vid_frames]
    inference_session = processor.init_video_session(
        video=pil_seq,
        inference_device=device,
        dtype=dtype,
    )

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=obj_id,
        input_masks=[seed_mask_bool.astype(np.uint8)],
    )

    video_segments = {}
    with torch.no_grad():
        for out in model.propagate_in_video_iterator(inference_session, start_frame_idx=0):
            res_masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=False,
            )[0]
            video_segments[out.frame_idx] = res_masks[0, 0].float().cpu().numpy()

    n_seq = len(pil_seq)
    all_masks = np.zeros((n_seq, H_vid, W_vid), dtype=np.float32)
    for fi, m in video_segments.items():
        all_masks[fi] = m
    masks_bool = all_masks > 0.0

    vid_masks = masks_bool[1:]
    np.save(os.path.join(out_dir, "masks_orig_space.npy"), vid_masks)
    np.save(os.path.join(out_dir, "seed_mask.npy"), seed_mask_bool)

    gif_frames = [Image.fromarray(src_overlay)]
    for t in range(0, len(vid_frames), stride):
        m = vid_masks[t]
        Image.fromarray((m.astype(np.uint8) * 255)).save(os.path.join(masks_dir, f"frame_{t:03d}.png"))
        overlay = _overlay_mask(vid_frames[t], m, color=mask_color, alpha=mask_alpha)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"frame_{t:03d}.png"))
        gif_frames.append(Image.fromarray(overlay))

    gif_path = os.path.join(out_dir, f"{label}_tracks.gif")
    gif_frames[0].save(
        gif_path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / gif_fps), loop=0, optimize=False,
    )
    coverage = float(vid_masks.reshape(len(vid_masks), -1).any(-1).mean())
    print(f"[{label}] done -> {gif_path} (frames with mask: {coverage:.2f})")
