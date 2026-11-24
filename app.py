"""
Vehicle Re-ID Search (Streamlit)
================================
Upload ONE cropped target vehicle image + MANY videos. The app detects every vehicle
(UVH-26 YOLO11s), matches each detection against the target with the fused Re-ID model
(ResNetmid + Swin-S), draws white TARGET boxes (and optional green boxes for all other
vehicles) and plays every annotated video together in a grid.

SETUP
-----
    pip install streamlit ultralytics timm torch torchvision opencv-python-headless \
                huggingface_hub pillow numpy imageio-ffmpeg

FILES (default locations, next to this app.py)
    models/fused_reid_veri.pt                      <- saved from the training notebook
    models/UVH-26-MV-YOLOv11-S.pt                  <- auto-downloaded on first run if missing

Or point to other locations with environment variables:
    REID_WEIGHTS=/path/fused_reid_veri.pt  YOLO_WEIGHTS=/path/yolo.pt  streamlit run app.py

RUN
---
    streamlit run app.py
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms as T
from ultralytics import YOLO

# ------------------------------------------------------------------ config
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
YOLO_WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"))
REID_WEIGHTS = Path(os.environ.get("REID_WEIGHTS", MODELS_DIR / "fused_reid_veri.pt"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

st.set_page_config(page_title="Vehicle Re-ID Search", layout="wide")
st.markdown(
    """<style>
    video { max-height: 42vh; width: 100%; object-fit: contain; background: #000; }
    .block-container { padding-top: 1.5rem; }
    </style>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ models
class ResNetMidInfer(nn.Module):
    """ResNet50: concat GAP(layer3) + GAP(layer4) -> BatchNorm (BNNeck)."""

    def __init__(self, last_stride=1):
        super().__init__()
        r = models.resnet50(weights=None)
        if last_stride == 1:
            r.layer4[0].conv2.stride = (1, 1)
            r.layer4[0].downsample[0].stride = (1, 1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(1024 + 2048)

    def forward(self, x):
        x = self.stem(x)
        mid = self.layer3(x)
        high = self.layer4(mid)
        return self.bn(torch.cat([self.pool(mid).flatten(1), self.pool(high).flatten(1)], 1))


class SwinInfer(nn.Module):
    """Swin-S: global-average-pooled features -> BatchNorm (BNNeck)."""

    def __init__(self, name):
        super().__init__()
        self.body = timm.create_model(name, pretrained=False, num_classes=0)
        self.bn = nn.BatchNorm1d(self.body.num_features)

    def forward(self, x):
        return self.bn(self.body(x))


def ensure_yolo_weights():
    if YOLO_WEIGHTS.is_file():
        return
    from huggingface_hub import hf_hub_download

    YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = YOLO_WEIGHTS.parent
    downloaded = hf_hub_download(
        repo_id="iisc-aim/UVH-26",
        repo_type="model",
        filename="weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt",
        local_dir=str(tmp_dir),
    )
    if Path(downloaded) != YOLO_WEIGHTS:
        shutil.copy(downloaded, YOLO_WEIGHTS)


@st.cache_resource(show_spinner="Loading detector and Re-ID models (first run only)...")
def load_models():
    ensure_yolo_weights()
    detector = YOLO(str(YOLO_WEIGHTS))
    b = torch.load(REID_WEIGHTS, map_location="cpu")
    res = ResNetMidInfer(b["last_stride"])
    res.load_state_dict(b["resnet"], strict=True)
    swin = SwinInfer(b["swin_name"])
    swin.load_state_dict(b["swin"], strict=True)

    def make_tf(size):
        return T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(b["mean"], b["std"])])

    return dict(
        detector=detector,
        res=res.to(DEVICE).eval(),
        swin=swin.to(DEVICE).eval(),
        w_res=float(b["w_res"]),
        res_tf=make_tf(b["res_size"]),
        swin_tf=make_tf(b["swin_size"]),
    )


@torch.inference_mode()
def embed(M, rgb_images, use_fp16=True, chunk=16, flip=False):
    """Unit-norm fused embeddings. ResNetmid is skipped entirely when its weight is 0."""
    w, use_res, out = M["w_res"], M["w_res"] > 0, []
    amp = use_fp16 and DEVICE.type == "cuda"
    for s in range(0, len(rgb_images), chunk):
        pils = [Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in rgb_images[s:s + chunk]]
        xs = torch.stack([M["swin_tf"](p) for p in pils]).to(DEVICE)
        xr = torch.stack([M["res_tf"](p) for p in pils]).to(DEVICE) if use_res else None
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp):
            fs = M["swin"](xs).float()
            if flip:
                fs = fs + M["swin"](torch.flip(xs, dims=[3])).float()
            if use_res:
                fr = M["res"](xr).float()
                if flip:
                    fr = fr + M["res"](torch.flip(xr, dims=[3])).float()
        fs = F.normalize(fs, dim=1)
        if use_res:
            fr = F.normalize(fr, dim=1)
            out.append(torch.cat([w ** 0.5 * fr, (1 - w) ** 0.5 * fs], 1).cpu())
        else:
            out.append(fs.cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------ video processing
def get_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def process_chunk(M, frames, frame_ids, writer, target, draw_all, thr, stride, batch, fp16, stats):
    det = M["detector"]
    sample = [i for i, fid in enumerate(frame_ids) if fid % stride == 0]
    results = []
    if sample:
        results = det.predict(
            [frames[i] for i in sample],
            imgsz=640,
            conf=0.15,
            batch=min(batch, len(sample)),
            device=0 if DEVICE.type == "cuda" else "cpu",
            half=bool(fp16 and DEVICE.type == "cuda"),
            verbose=False,
        )

    by_pos = {p: [] for p in sample}
    crop_refs, crops = [], []
    for pos, r in zip(sample, results):
        if r.boxes is None:
            continue
        h, w = frames[pos].shape[:2]
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cid = int(box.cls[0].item())
            names = det.names
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
            by_pos[pos].append(dict(box=(x1, y1, x2, y2), cls=cname,
                                    conf=float(box.conf[0].item()), sim=None))
            stats["boxes"] += 1
            crops.append(cv2.cvtColor(frames[pos][y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            crop_refs.append((pos, len(by_pos[pos]) - 1))

    if crops:
        for (pos, di), sim in zip(crop_refs, embed(M, crops, fp16) @ target):
            by_pos[pos][di]["sim"] = float(sim)
            stats["best"] = max(stats["best"], float(sim))
            stats["hits"] += int(sim >= thr)

    for pos, frame in enumerate(frames):
        for d in by_pos.get(pos, []):
            if d["sim"] is not None and d["sim"] >= thr:
                color, th, label = (255, 255, 255), 3, f"TARGET {d['sim']:.2f}"
            elif draw_all:
                color, th, label = (0, 220, 80), 2, f"{d['cls']} {d['conf']:.2f}"
            else:
                continue
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
            cv2.putText(frame, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
        writer.write(frame)


def process_video(M, video_path, out_path, target, draw_all, thr, stride, batch, fp16, on_progress):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    raw = Path(out_path).with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the video writer.")

    stats = {"boxes": 0, "hits": 0, "best": -1.0}
    frames, ids, n, t0 = [], [], 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            ids.append(n)
            n += 1
            if len(frames) >= batch:
                process_chunk(M, frames, ids, writer, target, draw_all, thr, stride, batch, fp16, stats)
                frames, ids = [], []
                if total:
                    on_progress(min(n / total, 1.0))
        if frames:
            process_chunk(M, frames, ids, writer, target, draw_all, thr, stride, batch, fp16, stats)
    finally:
        cap.release()
        writer.release()
    if n == 0:
        raise RuntimeError(f"No frames found in {video_path}")

    ffmpeg = get_ffmpeg()
    if ffmpeg:  # browsers can't play OpenCV's mp4v codec, so re-encode to H.264
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out_path)],
                       check=True)
        raw.unlink(missing_ok=True)
    else:
        raw.replace(out_path)  # may not play in the browser: pip install imageio-ffmpeg
    rate = n / max(time.perf_counter() - t0, 1e-6)
    tag = "FOUND" if stats["hits"] else "MISSING"
    return (f"{tag} | best cosine {stats['best']:.3f} | {stats['hits']} hits | "
            f"{stats['boxes']} boxes | {rate:.1f} FPS")


# ------------------------------------------------------------------ UI
st.title("Vehicle Re-ID Search")

if not REID_WEIGHTS.is_file():
    st.error(f"Re-ID weights not found: `{REID_WEIGHTS}`\n\n"
             "Put `fused_reid_veri.pt` in a `models/` folder next to app.py, "
             "or set the REID_WEIGHTS environment variable.")
    st.stop()
if get_ffmpeg() is None:
    st.warning("ffmpeg not found: output videos may not play in the browser. "
               "Fix with `pip install imageio-ffmpeg`.")

with st.sidebar:
    st.header("Settings")
    st.caption(f"Device: {DEVICE}")
    thr = st.slider("Match threshold (cosine)", 0.0, 1.0, 0.70, 0.01,
                    help="Lower it if the car shows MISSING; compare with the 'best cosine' shown per video.")
    draw_all = st.checkbox("Also box all other vehicles (green)", value=True)
    stride = st.selectbox("Frame stride", [1, 2, 3, 5], index=0,
                          help="Analyze every Nth frame (faster, but boxes flicker).")
    batch = st.selectbox("Frame batch", [1, 2, 4, 8], index=2, help="Lower if the GPU runs out of memory.")
    fp16 = st.checkbox("FP16 (GPU only)", value=True)

left, right = st.columns(2)
with left:
    st.subheader("1. Target vehicle")
    ref_file = st.file_uploader("Upload ONE cropped image of the target", type=IMAGE_TYPES, key="ref")
    if ref_file:
        st.image(ref_file, width=260)
with right:
    st.subheader("2. Videos to search")
    vid_files = st.file_uploader("Upload all videos", type=VIDEO_TYPES,
                                 accept_multiple_files=True, key="vids")
    if vid_files:
        st.caption(f"{len(vid_files)} video(s) selected")

run = st.button("Run search", type="primary", disabled=not (ref_file and vid_files))

if run:
    M = load_models()
    ref_rgb = np.asarray(Image.open(ref_file).convert("RGB"))
    target = embed(M, [ref_rgb], fp16, flip=True)[0].astype(np.float32)
    work = Path(tempfile.mkdtemp(prefix="reid_"))
    results, n = [], len(vid_files)
    bar = st.progress(0.0, text="Starting...")
    for i, vf in enumerate(vid_files):
        src = work / f"{i}_{vf.name}"
        src.write_bytes(vf.getbuffer())
        dst = work / f"{i}_annotated.mp4"

        def cb(frac, i=i, name=vf.name):
            bar.progress((i + frac) / n, text=f"Video {i + 1}/{n}: {name} ({frac:.0%})")

        try:
            status = process_video(M, src, dst, target, draw_all, thr, stride, batch, fp16, cb)
            results.append(dict(name=vf.name, path=str(dst), status=status))
        except Exception as e:
            results.append(dict(name=vf.name, path=None, status=f"ERROR: {e}"))
    bar.progress(1.0, text="Done")
    st.session_state["results"] = results

results = st.session_state.get("results")
if results:
    st.divider()
    st.subheader("Results")
    n = len(results)
    ncols = 1 if n == 1 else 2 if n <= 4 else 3  # grid sized to fit the screen
    for r0 in range(0, n, ncols):
        for col, res in zip(st.columns(ncols), results[r0:r0 + ncols]):
            with col:
                st.markdown(f"**{res['name']}**")
                st.caption(res["status"])
                if res["path"]:
                    st.video(res["path"], autoplay=True, muted=True, loop=True)
