
import os
import io
import time
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Malaria Object Detection (YOLO)", layout="wide")
st.title("🧫 Malaria Object Detection (YOLO) — RBC / Parasite")
st.caption("Load your YOLO weights (.pt), upload an image or video, and visualize detections.")

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
    line_thickness = st.slider("Box thickness (preview)", 1, 8, 2, 1)

    st.markdown("---")
    source_type = st.radio("Source type", ["Image", "Video"], horizontal=True)
    st.caption("Tip: Use **Camera input** below for a quick test image.")

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

def yolo_predict_on_image(model, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    # Run inference
    results = model(image, imgsz=imgsz, conf=conf_thres, iou=iou_thres, verbose=False)
    res = results[0]

    # Render boxes on the image (Ultralytics helper)
    plotted = res.plot(labels=show_labels, conf=show_conf, line_width=line_thickness)

    # Collect per-class counts
    class_counts = {}
    if res.boxes is not None and res.boxes.data is not None and len(res.boxes) > 0:
        # res.names is a mapping int->str
        names = res.names
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        for cid in cls_ids:
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
            class_counts[cname] = class_counts.get(cname, 0) + 1

    return plotted, class_counts

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
            plotted, counts = yolo_predict_on_image(model, np_img)

        with col_right:
            st.subheader("Result")
            st.image(plotted, channels="BGR", use_column_width=True)
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
