import os
import io
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ===========================
# Ultralytics (YOLO)
# ===========================
try:
    from ultralytics import YOLO
except Exception as e:
    st.error("Ultralytics isn't installed. Run: pip install ultralytics")
    raise

# ===========================
# Helpers
# ===========================
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
        return None
    if device_str == "CPU":
        return "cpu"
    # device like "GPU:0", "GPU:1"
    try:
        return int(device_str.split(":")[1])
    except Exception:
        return None

def _guess_class_sets(class_names: List[str]) -> Tuple[List[str], List[str]]:
    lower = [c.lower() for c in class_names]
    rbc_like = []
    infected_like = []
    infected_keywords = ("infect", "parasite", "ring", "troph", "schiz", "mero", "gameto", "hemozoin")
    rbc_keywords = ("rbc", "red blood cell", "erythro", "uninfected")

    for c in class_names:
        cl = c.lower()
        if any(k in cl for k in rbc_keywords):
            rbc_like.append(c)
        if any(k in cl for k in infected_keywords):
            infected_like.append(c)

    # If nothing is found, leave empty; user will select manually
    return sorted(set(rbc_like)), sorted(set(infected_like))

def _calc_parasitemia(counts_df: pd.DataFrame,
                      rbc_classes: List[str],
                      infected_classes: List[str],
                      denominator_mode: str = "RBC-only (strict)") -> Dict[str, Any]:

    if counts_df.empty:
        return {"infected": 0, "denominator": 0, "percent": 0.0, "note": "No detections."}

    # Create mapping for quick lookup
    count_map = dict(zip(counts_df["class_name"], counts_df["count"]))

    infected = int(sum(count_map.get(c, 0) for c in infected_classes))
    rbc = int(sum(count_map.get(c, 0) for c in rbc_classes))
    total = int(counts_df["count"].sum())

    if denominator_mode.startswith("RBC-only"):
        denom = rbc
        note = "Denominator = uninfected RBCs only."
    elif denominator_mode.startswith("RBC + infected"):
        denom = rbc + infected
        note = "Denominator = uninfected RBCs + infected RBCs."
    else:
        denom = total
        note = "Denominator = all detections (rough estimate)."

    percent = (infected / denom * 100.0) if denom > 0 else 0.0
    return {"infected": infected, "denominator": denom, "percent": percent, "note": note,
            "rbc_count": rbc, "total": total}

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Malaria Detector", page_icon="🧫", layout="wide")
st.title("🧫 Malaria Parasite Detection")

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
    model_names_sorted = []
    if os.path.exists(weights_path):
        try:
            _model_tmp = YOLO(weights_path)
            # maintain deterministic order by id
            model_names_sorted = [name for _, name in sorted(_model_tmp.names.items(), key=lambda kv: kv[0])]
            st.caption("Classes: " + ", ".join(model_names_sorted))
        except Exception as e:
            st.warning(f"Could not load model yet: {e}")
    else:
        st.info("Place your weights next to app.py or update the path above.")

# Lazy-load the model (only once)
@st.cache_resource(show_spinner=True)
def load_model_cached(path: str):
    return YOLO(path)

st.subheader("Detect on an image")
upimg = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

# ===========================
# Optional: Class selection for parasitemia
# ===========================
with st.expander("🧮 Parasitemia Settings", expanded=True):
    guessed_rbc, guessed_infected = _guess_class_sets(model_names_sorted) if model_names_sorted else ([], [])
    st.caption("Select which classes count as **RBC** (uninfected) and which count as **Infected**.")
    colr, coli = st.columns(2)
    with colr:
        rbc_classes = st.multiselect(
            "RBC classes (uninfected)",
            options=model_names_sorted,
            default=guessed_rbc
        )
    with coli:
        infected_classes = st.multiselect(
            "Infected classes (parasite-positive / stages)",
            options=model_names_sorted,
            default=guessed_infected
        )

    denominator_mode = st.radio(
        "Denominator mode for % parasitemia",
        options=["RBC-only (strict)", "RBC + infected (fallback)", "All detections (very rough)"],
        index=0,
        help=(
            "RBC-only: denominator is uninfected RBCs only.\n"
            "RBC + infected: denominator includes both uninfected and infected RBCs (useful if the model doesn't detect uninfected RBCs).\n"
            "All detections: uses every detection as denominator (rough)."
        )
    )

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

    # Tables
    df = _result_to_dataframe(result)
    counts = _class_counts(df)

    # === % Parasitemia ===
    par = _calc_parasitemia(counts, rbc_classes, infected_classes, denominator_mode)

    with col1:
        st.image(pil, caption="Original", use_column_width=True)
    with col2:
        st.image(plotted_rgb, caption="Detections", use_column_width=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="% Parasitemia", value=f"{par['percent']:.2f}%")
    m2.metric(label="Infected RBCs (numerator)", value=int(par["infected"]))
    m3.metric(label="Denominator", value=int(par["denominator"]))
    m4.metric(label="All Detections", value=int(par["total"]))
    st.caption(f"Note: {par['note']}")

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

    # Quick summary of which classes were counted
    st.markdown("#### Class Mapping Used")
    cm1, cm2 = st.columns(2)
    cm1.write("**RBC classes** (uninfected): " + (", ".join(rbc_classes) if rbc_classes else "—"))
    cm2.write("**Infected classes**: " + (", ".join(infected_classes) if infected_classes else "—"))

else:
    st.info("Upload an image to start.")
