# app.py
# Streamlit + Ultralytics YOLO example for malaria parasite detection
# - Supports image and video upload
# - Adjustable confidence/IoU thresholds
# - Shows per-class counts and lets you download detections as CSV
#
# How to run:
#   1) pip install -U streamlit ultralytics opencv-python pillow numpy pandas
#   2) put your weights file (malaria11n.pt) next to app.py (or update the path in the sidebar)
#   3) streamlit run app.py

import os
import io
import time
import tempfile
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    from ultralytics import YOLO
except Exception as e:
    st.error("Ultralytics isn't installed. Run: pip install ultralytics")
    raise

# ---------------------------
# Helpers
# ---------------------------

def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _result_to_dataframe(result) -> pd.DataFrame:
    """Convert a single YOLO result to a tidy DataFrame."""
    names = result.names  # dict id->name
    boxes = getattr(result, "boxes", None)
    rows = []
    if boxes is None or boxes.data is None or len(boxes) == 0:
        return pd.DataFrame(columns=["class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])
    for b in boxes:
        # b.xyxy[0] shape (4,), b.conf[0], b.cls[0]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0]) if b.conf is not None else None
        cid = int(b.cls[0]) if b.cls is not None else -1
        rows.append({
            "class_id": cid,
            "class_name": names.get(cid, str(cid)) if isinstance(names, dict) else str(cid),
            "conf": conf,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        })
    return pd.DataFrame(rows)


def _class_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"class_name": [], "count": []})
    counts = df.groupby("class_name").size().reset_index(name="count").sort_values("count", ascending=False)
    return counts


def _export_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _get_device_choice(device_str: str) -> Any:
    # Streamlit selectbox returns a string; YOLO accepts: 0 or "0" for first GPU, or "cpu" for CPU
    if device_str == "Auto":
        # Let YOLO decide; return None
        return None
    if device_str == "CPU":
        return "cpu"
    # device like "GPU:0", "GPU:1"
    try:
        return int(device_str.split(":")[1])
    except Exception:
        return None


# ---------------------------
# UI Layout
# ---------------------------

st.set_page_config(page_title="Malaria YOLO Detector", page_icon="🧫", layout="wide")
st.title("🧫 Malaria Parasite Detection – YOLO")
st.caption("Demo Streamlit app using Ultralytics YOLO with your trained weights (malaria11n.pt)")

with st.sidebar:
    st.header("⚙️ Settings")
    default_weights = "malaria11n.pt"
    weights_path = st.text_input("Weights path (.pt)", value=default_weights)
    conf_thres = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
    iou_thres = st.slider("IoU", min_value=0.1, max_value=0.95, value=0.5, step=0.05)
    imgsz = st.select_slider("Image size (inference)", options=[320, 384, 448, 512, 576, 640, 704, 768, 832, 896], value=640)
    device_str = st.selectbox("Device", options=["Auto", "CPU", "GPU:0", "GPU:1"], index=0)
    show_labels = st.checkbox("Show labels on image", value=True)
    show_conf = st.checkbox("Show confidence on image", value=True)

    st.markdown("---")
    st.subheader("📘 Model Info")
    if os.path.exists(weights_path):
        try:
            _model_tmp = YOLO(weights_path)
            names_list = [n for _, n in sorted(_model_tmp.names.items())]
            st.caption("Classes: " + ", ".join(names_list))
        except Exception as e:
            st.warning(f"Could not load model yet: {e}")
    else:
        st.info("Place malaria11n.pt next to app.py or update the path above.")

# Lazy-load the model (only once)
@st.cache_resource(show_spinner=True)
def load_model_cached(path: str):
    return YOLO(path)


# Tabs for different sources
img_tab, vid_tab = st.tabs(["📷 Image", "🎥 Video"])  # add Webcam tab later if needed

# ---------------------------
# Image Tab
# ---------------------------
with img_tab:
    st.subheader("Detect on an image")
    upimg = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    col1, col2 = st.columns([1, 1])

    if upimg is not None:
        image_bytes = upimg.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil)

        with st.spinner("Running YOLO inference..."):
            model = load_model_cached(weights_path)
            device_choice = _get_device_choice(device_str)
            results = model.predict(
                source=np_img,
                conf=conf_thres,
                iou=iou_thres,
                imgsz=imgsz,
                device=device_choice,
                verbose=False,
                stream=False,
            )

        result = results[0]
        plotted_bgr = result.plot(labels=show_labels, conf=show_conf)  # BGR
        plotted_rgb = _bgr_to_rgb(plotted_bgr)

        df = _result_to_dataframe(result)
        counts = _class_counts(df)

        with col1:
            st.image(pil, caption="Original", use_column_width=True)
        with col2:
            st.image(plotted_rgb, caption="Detections", use_column_width=True)

        st.markdown("### 📊 Detections")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                label="⬇️ Download detections CSV",
                data=_export_df_to_csv_bytes(df),
                file_name="detections.csv",
                mime="text/csv",
            )
        with c2:
            st.dataframe(counts, use_container_width=True)

    else:
        st.info("Upload an image to start.")

# ---------------------------
# Video Tab
# ---------------------------
with vid_tab:
    st.subheader("Detect on a video")
    upvid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])
    auto_play = st.checkbox("Auto-play processed video", value=True)

    if upvid is not None:
        # Save to a temporary file
        tdir = tempfile.mkdtemp(prefix="yolo_vid_")
        in_path = os.path.join(tdir, upvid.name)
        with open(in_path, "wb") as f:
            f.write(upvid.read())

        with st.spinner("Running YOLO on video (this can take a bit)..."):
            model = load_model_cached(weights_path)
            device_choice = _get_device_choice(device_str)
            # Let Ultralytics handle saving the annotated video
            results = model.predict(
                source=in_path,
                conf=conf_thres,
                iou=iou_thres,
                imgsz=imgsz,
                device=device_choice,
                save=True,
                verbose=False,
                stream=False,
            )
        # The library places outputs under results[0].save_dir
        save_dir = getattr(results[0], "save_dir", None)

        # Find the processed video in save_dir
        processed_video = None
        if save_dir and os.path.isdir(save_dir):
            for fn in os.listdir(save_dir):
                if fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    processed_video = os.path.join(save_dir, fn)
                    break

        if processed_video and os.path.exists(processed_video):
            st.success("Video processed!")
            st.video(processed_video, start_time=0) if auto_play else st.write("Processed video ready below.")
            with open(processed_video, "rb") as f:
                st.download_button("⬇️ Download processed video", f, file_name=os.path.basename(processed_video))
        else:
            st.warning("Couldn't locate the processed video. Check the logs/permissions.")
    else:
        st.info("Upload a video to start.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
---
**Tips**
- If you see CUDA warnings, switch *Device* to **CPU** in the sidebar.
- For speed on GPU, keep the image size close to what you trained with (e.g., 640).
- The CSV includes one row per detection with (x1,y1,x2,y2) in pixels on the uploaded image.
""")
