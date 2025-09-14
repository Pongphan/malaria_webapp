import os
import io
import re
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# ---- UI Setup ----
st.set_page_config(page_title="Malaria Object Detection (YOLO)", layout="wide")
st.title("🧫 Malaria Object Detection (YOLO) — RBC / Ring & % Parasitemia")

with st.sidebar:
    st.header("⚙️ Settings")

    # Model path (default to your path)
    model_path = st.text_input(
        "YOLO model path (.pt)",
        value="/content/drive/MyDrive/YOLO/malaria11x.pt",
        help="Path to your trained YOLO model file.",
    )

    conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    font_scale = st.slider("Label font scale (ring only)", 0.5, 3.0, 1.2, 0.1)
    box_thickness = st.slider("Box thickness", 1, 6, 2, 1)

    show_all_labels = st.checkbox("Also draw labels for RBC/others", value=False)
    show_table = st.checkbox("Show detections table", value=True)

    st.markdown("---")
    st.caption("Tip: If your class names look like `ring,` or `rbc,`, this app will auto-clean them.")

st.markdown("#### Upload an image or provide a file path below")

# Image inputs: uploader or file path
col_up, col_path = st.columns([2, 2])
with col_up:
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "tif", "tiff"])
with col_path:
    img_path = st.text_input(
        "…or use an image path",
        value="/content/dataset/Point Set/144C3thin_original/Img/IMG_20150608_162835.jpg",
    )

run_btn = st.button("🚀 Run Detection")

# ---- Utilities ----
def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Read uploaded file into RGB numpy array."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def load_image_from_path(path) -> np.ndarray:
    """Read image from path into RGB numpy array."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def sanitize_label(label: str) -> str:
    """Lowercase, strip whitespace/punctuation from end, normalize common variants."""
    if label is None:
        return ""
    # Remove trailing punctuation like commas/colons/semicolons/dots
    lab = re.sub(r"[\s,;:\.]+$", "", label.lower().strip())
    # Normalize known variants
    aliases = {
        "ring form": "ring",
        "ring_form": "ring",
        "ring-form": "ring",
        "ringform": "ring",
        "red blood cell": "rbc",
        "rbc_cell": "rbc",
        "erythrocyte": "rbc",
    }
    return aliases.get(lab, lab)

# Color map (BGR because we'll draw in BGR)
COLOR_RING = (255, 0, 0)   # blue-ish
COLOR_RBC  = (0, 200, 0)   # green
COLOR_OTHER= (30, 50, 220) # red-ish

# Class buckets (after sanitize_label)
RING_ALIASES = {"ring"}
RBC_ALIASES  = {"rbc"}

# ---- Lazy-load YOLO model ----
@st.cache_resource(show_spinner=True)
def get_model(path: str):
    from ultralytics import YOLO  # import here to avoid import cost if not used
    return YOLO(path)

def run_yolo(model, image_rgb: np.ndarray, conf: float):
    # Ultralytics can take numpy RGB arrays directly
    results = model.predict(source=image_rgb, conf=conf, verbose=False)
    return results

def draw_and_summarize(image_rgb: np.ndarray, results, class_name_map: dict,
                       conf: float, font_scale: float, thickness: int, draw_all_labels: bool):
    """
    Draw boxes on a copy of the image and compute counts and table.
    Returns (annotated_rgb, count_rbc, count_ring, rows:list[dict])
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    count_ring = 0
    count_rbc = 0
    rows = []

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, 0, rows

    boxes = results[0].boxes
    # Safe access for tensors
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else boxes.cls.astype(int)
    confs = boxes.conf.cpu().numpy().astype(float) if hasattr(boxes.conf, "cpu") else boxes.conf.astype(float)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        cid = int(clss[i])
        c = float(confs[i])
        if c < conf:
            continue

        raw_label = class_name_map.get(cid, str(cid))
        lab = sanitize_label(raw_label)

        # Choose color + counts
        if lab in RING_ALIASES:
            color = COLOR_RING
            count_ring += 1
        elif lab in RBC_ALIASES:
            color = COLOR_RBC
            count_rbc += 1
        else:
            color = COLOR_OTHER

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        # Labels: by default only for RING (like your original script), optional for all
        if lab in RING_ALIASES or draw_all_labels:
            label_text = f"{raw_label} {c:.2f}"
            cv2.putText(
                img_bgr,
                label_text,
                (x1, max(0, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, thickness),
                lineType=cv2.LINE_AA,
            )

        rows.append({
            "label_raw": raw_label,
            "label_sanitized": lab,
            "conf": round(c, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area": int((x2 - x1) * (y2 - y1))
        })

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, count_rbc, count_ring, rows


# ---- Main run ----
image_rgb = None
filename = None

if file is not None:
    image_rgb = load_image_from_upload(file)
    filename = getattr(file, "name", "uploaded_image")
elif img_path and os.path.exists(img_path):
    image_rgb = load_image_from_path(img_path)
    filename = os.path.basename(img_path)

if run_btn:
    if not model_path or not os.path.exists(model_path):
        st.error("Model file not found. Please provide a valid `.pt` path.")
    elif image_rgb is None:
        st.error("Please upload an image or provide a valid image path.")
    else:
        with st.spinner("Loading model and running detection..."):
            model = get_model(model_path)

            # Prefer model.names if available; fallback to results[0].names later
            class_name_map = None
            try:
                # ultralytics keeps names as dict {id: name}
                if hasattr(model, "names") and isinstance(model.names, (dict, list)):
                    if isinstance(model.names, list):
                        class_name_map = {i: n for i, n in enumerate(model.names)}
                    else:
                        class_name_map = dict(model.names)
            except Exception:
                class_name_map = None

            results = run_yolo(model, image_rgb, conf_thres)

            # If model.names missing, try pulling from results
            if class_name_map is None and results and hasattr(results[0], "names"):
                nm = results[0].names
                class_name_map = {i: nm[i] for i in range(len(nm))} if isinstance(nm, list) else dict(nm)

            if class_name_map is None:
                class_name_map = {}

            annotated_rgb, count_rbc, count_ring, rows = draw_and_summarize(
                image_rgb, results, class_name_map, conf_thres, font_scale, box_thickness, show_all_labels
            )

        total_cells = count_rbc + count_ring
        parasitemia = (count_ring / total_cells * 100.0) if total_cells > 0 else 0.0

        # ---- Layout: left image, right metrics/table ----
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("🖼️ Annotated Image")
            st.image(annotated_rgb, caption=f"{filename}", use_column_width=True)

            # Download button
            pil_img = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download annotated image",
                buf.getvalue(),
                file_name=f"annotated_{filename.replace(os.sep, '_')}.png",
                mime="image/png",
            )

        with right:
            st.subheader("📊 Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("RBC count", f"{count_rbc}")
            m2.metric("Ring count", f"{count_ring}")
            m3.metric("% Parasitemia", f"{parasitemia:.2f}%")

            st.caption("Parasitemia = ring / (RBC + ring) × 100. Set your confidence in the sidebar.")

            if show_table and rows:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.markdown("#### Detections")
                st.dataframe(df, use_container_width=True)
            elif show_table:
                st.info("No detections to show at current threshold.")

else:
    st.info("👈 Choose your model, upload an image or set a path, then click **Run Detection**.")
