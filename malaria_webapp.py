
import os
import io
import time
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO
import imageio.v3 as iio

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Malaria Object Detection (YOLO)", layout="wide")
st.title("🧫 Malaria Object Detection (YOLO) — RBC / Parasite (no OpenCV)")
st.caption("Load your YOLO weights (.pt), upload an image or video, and visualize detections — all without cv2.")

with st.sidebar:
    st.header("⚙️ Settings")

    # Model path (default to your path/file name)
    model_path = st.text_input(
        "YOLO model path (.pt)",
        value="malaria11n.pt",
        help="Path to your trained YOLO model file (put it in the same folder or give an absolute path).",
    )

    imgsz = st.number_input("Inference image size (imgsz)", min_value=256, max_value=2048, value=640, step=32)
    conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_thres = st.slider("IoU threshold (NMS)", 0.1, 0.95, 0.45, 0.01)
    show_labels = st.checkbox("Draw labels (class & conf)", value=True)
    show_conf = st.checkbox("Include confidence in labels", value=True)
    line_thickness = st.slider("Box thickness (preview)", 1, 8, 3, 1)

    st.markdown("---")
    source_type = st.radio("Source type", ["Image", "Video"], horizontal=True)
    st.caption("Tip: Use **Camera input** below for a quick test image.")

# ==============================
# Helpers (no cv2 drawing)
# ==============================
def pil_draw_boxes(image: Image.Image, boxes_xyxy: np.ndarray, classes: np.ndarray, confs: np.ndarray, names: Dict[int, str]) -> Image.Image:
    """Draw boxes on a PIL image using ImageDraw (avoid cv2)."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        cls_id = int(classes[i])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        if show_conf and confs is not None:
            label = f"{label} {float(confs[i]):.2f}"

        # rectangle
        for t in range(line_thickness):  # thickness by drawing multiple rectangles
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=(255, 0, 0))

        if show_labels:
            # text background
            text_w, text_h = draw.textbbox((0,0), label, font=font)[2:]
            pad = 2
            draw.rectangle([x1, y1 - text_h - 2*pad, x1 + text_w + 2*pad, y1], fill=(255,0,0))
            draw.text((x1 + pad, y1 - text_h - pad), label, fill=(255,255,255), font=font)

    return img

# ==============================
# Caching the model
# ==============================
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model = YOLO(weights_path)
    return model

def draw_info_panel(counts: Dict[str, int]):
    total = sum(counts.values())
    cols = st.columns(max(1, min(4, len(counts) or 1)))
    i = 0
    for cname, cnum in counts.items():
        with cols[i % len(cols)]:
            st.metric(label=f"🧬 {cname}", value=cnum)
        i += 1
    st.info(f"**Total detections:** {total}")

def yolo_predict_on_image(model, image: np.ndarray) -> Tuple[Image.Image, Dict[str, int]]:
    # Run inference
    results = model(image, imgsz=imgsz, conf=conf_thres, iou=iou_thres, verbose=False)
    res = results[0]

    class_counts = {}
    names = res.names

    if res.boxes is not None and res.boxes.data is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        # Count classes
        for cid in cls_ids:
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
            class_counts[cname] = class_counts.get(cname, 0) + 1

        # Draw with PIL
        pil_img = Image.fromarray(image if image.dtype==np.uint8 else image.astype(np.uint8))
        drawn = pil_draw_boxes(pil_img, boxes, cls_ids, confs, names)
    else:
        drawn = Image.fromarray(image if image.dtype==np.uint8 else image.astype(np.uint8))

    return drawn, class_counts

def process_video(model, video_path: str) -> str:
    """Read/write video with imageio (ffmpeg), perform YOLO per frame; no cv2."""
    # probe to get fps if possible
    try:
        meta = iio.immeta(video_path, plugin="ffmpeg")
        fps = float(meta.get("fps", 25.0)) if meta else 25.0
    except Exception:
        fps = 25.0

    # writer
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "yolo_output.mp4")
    writer = iio.imopen(out_path, "w", plugin="ffmpeg", fps=fps, codec="libx264")

    # iterate frames
    # (ffmpeg yields frames as RGB uint8 arrays)
    frame_iter = iio.imiter(video_path, plugin="ffmpeg")
    # If total frames is unknown, we update progress indeterminately
    pbar = st.progress(0.0, text="Processing video...")
    processed = 0
    for frame in frame_iter:
        # Ensure uint8 RGB
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Inference
        results = model(frame, imgsz=imgsz, conf=conf_thres, iou=iou_thres, verbose=False)
        res = results[0]

        # Draw using PIL
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names
            pil_frame = Image.fromarray(frame)
            drawn = pil_draw_boxes(pil_frame, boxes, cls_ids, confs, names)
            out_frame = np.asarray(drawn)
        else:
            out_frame = frame

        writer.write(out_frame)
        processed += 1
        # simple progress pulse
        pbar.progress(min((processed % 100) / 100.0, 1.0))

    writer.close()
    pbar.progress(1.0, text="Done!")
    return out_path

# ==============================
# Main UI
# ==============================
try:
    model = load_model(model_path)
    st.success(f"✅ Loaded model: `{os.path.basename(model_path)}`")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if source_type == "Image":
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Upload an image")
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

        st.markdown("or capture from camera")
        camera_img = st.camera_input("Camera snapshot (optional)")

    image_to_use = None
    if camera_img is not None:
        image_to_use = Image.open(camera_img).convert("RGB")
    elif uploaded is not None:
        image_to_use = Image.open(uploaded).convert("RGB")

    if image_to_use is not None:
        np_img = np.array(image_to_use)
        with st.spinner("Running inference..."):
            drawn, counts = yolo_predict_on_image(model, np_img)

        with col_right:
            st.subheader("Result")
            st.image(drawn, use_column_width=True)
            if counts:
                draw_info_panel(counts)
    else:
        st.info("Upload or capture an image to get started.")

else:
    st.subheader("Upload a video")
    vid = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
    if vid is not None:
        # Save to temp
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid.read())
        tfile.flush()

        with st.spinner("Running video inference..."):
            out_path = process_video(model, tfile.name)

        st.success("Inference complete.")
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("Download processed video", data=f, file_name="yolo_output.mp4", mime="video/mp4")
    else:
        st.info("Upload a video file to start processing.")
