import os
import io
import re
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import streamlit as st

# ---- UI Setup (must be the first Streamlit call) ----
st.set_page_config(page_title="Malaria Object Detection (YOLO)", layout="wide")
st.title("🧫 Malaria Object Detection (YOLO) — RBC / Ring & % Parasitemia")

with st.sidebar:
    st.header("⚙️ Settings")

    model_path = st.text_input(
        "YOLO model path (.pt) or hub name/URL",
        value="malaria11n.pt",
        help="Local .pt path, Ultralytics hub model name, or a URL supported by Ultralytics.",
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
def load_image_from_upload(uploaded_file) -> np.ndarray | None:
    """Read uploaded file into RGB numpy array."""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    except UnidentifiedImageError:
        return None

def load_image_from_path(path) -> np.ndarray | None:
    """Read image from path into RGB numpy array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def sanitize_label(label: str) -> str:
    """Lowercase, strip trailing punctuation/space, normalize common variants."""
    if not label:
        return ""
    lab = re.sub(r"[\s,;:\.]+$", "", str(label).lower().strip())
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
@st.cache_resource
def get_model(path: str):
    try:
        from ultralytics import YOLO  # import here to avoid import cost if not used
    except Exception as e:
        raise RuntimeError(
            "Failed to import Ultralytics. Make sure `ultralytics` is installed."
        ) from e
    try:
        return YOLO(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from: {path}") from e

def to_numpy(x):
    """Safely convert torch/tensor-like to numpy, else passthrough."""
    try:
        # torch Tensor
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        # torch-like with .cpu()
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
    except Exception:
        pass
    # already numpy or list-like
    return np.asarray(x)

def run_yolo(model, image_rgb: np.ndarray, conf: float):
    # Ultralytics v8 accepts numpy RGB arrays directly
    results = model.predict(source=image_rgb, conf=conf, verbose=False)
    # Ensure we have a list-like of results
    if results is None:
        return []
    try:
        # Some versions return a Results object iterable; make it list
        return list(results)
    except TypeError:
        return [results]

def extract_name_map(model, results0):
    """Derive class id -> name map from model or results; fallback to {}."""
    # model.names
    try:
        names = getattr(model, "names", None)
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    except Exception:
        pass

    # results[0].names
    try:
        nm = getattr(results0, "names", None)
        if isinstance(nm, list):
            return {i: nm[i] for i in range(len(nm))}
        if isinstance(nm, dict):
            return {int(k): v for k, v in nm.items()}
    except Exception:
        pass

    return {}

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

    if not results:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, 0, rows

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(getattr(boxes, "xyxy", [])) == 0:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, 0, rows

    xyxy = to_numpy(boxes.xyxy)
    clss = to_numpy(boxes.cls).astype(int) if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), int)
    confs = to_numpy(boxes.conf).astype(float) if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy), float)

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

        # Labels: by default only for RING; optional for all
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
            "area": int(max(0, (x2 - x1)) * max(0, (y2 - y1)))
        })

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, count_rbc, count_ring, rows


# ---- Main run ----
image_rgb = None
filename = None

if file is not None:
    image_rgb = load_image_from_upload(file)
    if image_rgb is None:
        st.error("Unable to read the uploaded image. Please try a different file.")
    else:
        filename = getattr(file, "name", "uploaded_image")
elif img_path:
    if os.path.exists(img_path):
        image_rgb = load_image_from_path(img_path)
        if image_rgb is None:
            st.error("Failed to load image from the given path (unsupported or corrupted).")
        else:
            filename = os.path.basename(img_path)
    else:
        # Don't error here; user might upload instead
        pass

if run_btn:
    # Model path sanity: if it's a local path and doesn't exist, block; otherwise allow (hub/URL)
    is_local_like = not (model_path.startswith("http://") or model_path.startswith("https://"))
    if not model_path:
        st.error("Please provide a model path or name.")
    elif is_local_like and not os.path.exists(model_path):
        st.error("Local model file not found. Provide a valid `.pt` path, or a hub/URL model name.")
    elif image_rgb is None:
        st.error("Please upload an image or provide a valid image path.")
    else:
        with st.spinner("Loading model and running detection..."):
            try:
                model = get_model(model_path)
            except Exception as e:
                st.exception(e)
                st.stop()

            results = run_yolo(model, image_rgb, conf_thres)

            # Build class-name map
            class_name_map = extract_name_map(model, results[0] if results else None)

            annotated_rgb, count_rbc, count_ring, rows = draw_and_summarize(
                image_rgb, results, class_name_map, conf_thres, font_scale, box_thickness, show_all_labels
            )

        total_cells = count_rbc + count_ring
        parasitemia = (count_ring / total_cells * 100.0) if total_cells > 0 else 0.0

        # ---- Layout: left image, right metrics/table ----
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("🖼️ Annotated Image")
            st.image(annotated_rgb, caption=f"{filename or 'image'}", use_container_width=True)

            # Download button
            pil_img = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            safe_name = (filename or "image").replace(os.sep, "_")
            st.download_button(
                "⬇️ Download annotated image",
                buf,
                file_name=f"annotated_{safe_name}.png",
                mime="image/png",
            )

        with right:
            st.subheader("📊 Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("RBC count", f"{count_rbc}")
            m2.metric("Ring count", f"{count_ring}")
            m3.metric("% Parasitemia", f"{parasitemia:.2f}%")

            st.caption("Parasitemia = ring / (RBC + ring) × 100. Adjust confidence in the sidebar.")

            if show_table:
                import pandas as pd
                if rows:
                    df = pd.DataFrame(rows)
                    st.markdown("#### Detections")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No detections to show at current threshold.")

else:
    st.info("👈 Choose your model, upload an image or set a path, then click **Run Detection**.")
