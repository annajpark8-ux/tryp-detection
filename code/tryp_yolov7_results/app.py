"""Trypanosome Detection - Styled Streamlit Web Interface

Usage:
    pip install streamlit onnxruntime opencv-python numpy Pillow
    streamlit run trypanosome_detector_styled.py

Place your model at:
    weights/best.onnx
next to this script.
"""
import os
import time
import numpy as np
import cv2
import streamlit as st
import onnxruntime as ort

# Auto-find model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "weights", "best.onnx")


@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)


def nms(boxes, scores, iou_threshold=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.array(keep)


def detect(session, img_bgr, img_size=640, conf_thresh=0.25, iou_thresh=0.45):
    img, ratio, (dw, dh) = letterbox(img_bgr, img_size)
    inp = np.expand_dims(img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0, 0)

    t0 = time.time()
    pred = session.run(None, {session.get_inputs()[0].name: inp})[0]
    infer_ms = (time.time() - t0) * 1000

    if pred.ndim == 3:
        pred = pred[0]

    scores = pred[:, 4] * (pred[:, 5:].max(1) if pred.shape[1] > 5 else 1)
    mask = scores > conf_thresh
    pred, scores = pred[mask], scores[mask]

    if len(pred) == 0:
        return [], infer_ms

    boxes = np.column_stack([
        pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
        pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    ])

    keep = nms(boxes, scores, iou_thresh)
    boxes, scores = boxes[keep], scores[keep]

    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / ratio, 0, img_bgr.shape[1])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / ratio, 0, img_bgr.shape[0])

    detections = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s)) for b, s in zip(boxes, scores)]
    return detections, infer_ms


def draw_detections(img_bgr, detections):
    """Draw blue-purple detection boxes to match the UI theme."""
    img = img_bgr.copy()
    box_color = (226, 88, 123)      # BGR: bright violet-blue accent
    label_color = (164, 82, 255)    # BGR: purple label fill

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        label = f"trypanosome {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_label = max(y1 - th - 10, 0)
        cv2.rectangle(img, (x1, y_label), (x1 + tw + 8, y_label + th + 10), label_color, -1)
        cv2.putText(img, label, (x1 + 4, y_label + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #f7f8ff;
            --card: rgba(255, 255, 255, 0.92);
            --ink: #17213f;
            --muted: #667085;
            --blue: #2563eb;
            --purple: #7c3aed;
            --soft-blue: #dbeafe;
            --soft-purple: #ede9fe;
            --border: rgba(124, 58, 237, 0.16);
            --shadow: 0 18px 45px rgba(31, 41, 55, 0.10);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.16), transparent 32rem),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 30rem),
                linear-gradient(180deg, #ffffff 0%, var(--bg) 48%, #ffffff 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eef2ff 0%, #ffffff 55%, #f8fafc 100%);
            border-right: 1px solid rgba(124, 58, 237, 0.12);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #312e81;
        }

        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(124, 58, 237, 0.95));
            color: white;
            box-shadow: var(--shadow);
            margin-bottom: 1.6rem;
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -80px;
            top: -80px;
            background: rgba(255,255,255,0.16);
            border-radius: 50%;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.22);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 1rem;
        }

        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4.4rem);
            line-height: 0.95;
            margin: 0 0 1rem 0;
            letter-spacing: -0.06em;
            color: #ffffff;
        }

        .hero p {
            max-width: 780px;
            font-size: 1.08rem;
            line-height: 1.65;
            color: rgba(255, 255, 255, 0.88);
            margin: 0;
        }

        .panel {
            padding: 1.25rem;
            border-radius: 24px;
            background: var(--card);
            border: 1px solid var(--border);
            box-shadow: 0 14px 34px rgba(31, 41, 55, 0.07);
            margin-bottom: 1.1rem;
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #1e1b4b;
            margin-bottom: 0.35rem;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 0.9rem;
        }

        .result-card {
            padding: 1.4rem;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(124, 58, 237, 0.16);
            box-shadow: 0 16px 38px rgba(31, 41, 55, 0.08);
            margin: 1.3rem 0;
        }

        .file-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--soft-blue), var(--soft-purple));
            color: #312e81;
            font-weight: 800;
            margin-bottom: 1rem;
        }

        .image-label {
            font-weight: 800;
            color: #312e81;
            margin-bottom: 0.35rem;
        }

        .status-empty {
            padding: 2rem;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(219, 234, 254, 0.9), rgba(237, 233, 254, 0.9));
            border: 1px dashed rgba(124, 58, 237, 0.35);
            text-align: center;
            color: #312e81;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(124, 58, 237, 0.14);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 10px 25px rgba(31, 41, 55, 0.06);
        }

        div[data-testid="stMetric"] label {
            color: #667085 !important;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: #312e81;
            font-weight: 800;
        }

        .stFileUploader label {
            font-weight: 800;
            color: #312e81;
        }

        .stFileUploader section {
            border: 1.5px dashed rgba(124, 58, 237, 0.32);
            background: rgba(255, 255, 255, 0.78);
            border-radius: 24px;
            padding: 1rem;
        }

        .stSlider [data-baseweb="slider"] > div:first-child {
            background: linear-gradient(90deg, var(--blue), var(--purple));
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.22), transparent);
            margin: 1.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Streamlit UI ────────────────────────────────────────────

st.set_page_config(page_title="Trypanosome Detector", page_icon="🔬", layout="wide")
inject_css()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">🔬 AI Microscopy Assistant</div>
        <h1>Trypanosome Parasite Detection</h1>
        <p>
            Upload microscopy images of blood smears and automatically detect
            <em>Trypanosoma brucei brucei</em> parasites using a lightweight YOLOv7-tiny ONNX model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.markdown("## Detection Controls")
    st.caption("Tune the model before uploading images.")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values reduce false positives but may miss faint or small parasites.",
    )

    st.markdown("---")
    st.markdown("## Model Card")
    st.markdown(
        """
        **Architecture:** YOLOv7-tiny  
        **Parameters:** ~6M  
        **Input size:** 640 × 640  
        **Runtime:** ONNX, CPU  
        **Task:** Blood smear object detection
        """
    )

    st.markdown("---")
    st.info("Tip: start around 25–35% confidence, then raise the threshold if too many boxes appear.")

# Load model
if not os.path.exists(DEFAULT_MODEL):
    st.error(f"Model not found at `{DEFAULT_MODEL}`. Make sure `weights/best.onnx` exists next to this script.")
    st.stop()

session = load_model(DEFAULT_MODEL)

st.markdown(
    """
    <div class="panel">
        <div class="section-title">Upload images</div>
        <div class="section-subtitle">
            Accepted formats: JPG, JPEG, PNG, BMP, and TIFF. You can upload multiple images at once.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload blood smear image(s)",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} image{'s' if len(uploaded_files) != 1 else ''}. Running detection below.")

    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error(f"Could not read {uploaded_file.name}")
            continue

        detections, infer_ms = detect(session, img_bgr, conf_thresh=conf_threshold)
        result_img = draw_detections(img_bgr, detections)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="file-chip">🧪 {uploaded_file.name}</div>', unsafe_allow_html=True)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Parasites detected", len(detections))
        metric_cols[1].metric("Inference time", f"{infer_ms:.0f} ms")
        metric_cols[2].metric("Confidence threshold", f"{conf_threshold:.0%}")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="image-label">Original image</div>', unsafe_allow_html=True)
            st.image(original_rgb, use_container_width=True)

        with col2:
            st.markdown('<div class="image-label">Detection overlay</div>', unsafe_allow_html=True)
            st.image(result_rgb, use_container_width=True)

        if detections:
            with st.expander("View detection details"):
                for i, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                    st.markdown(
                        f"**#{i}** — Confidence: **{conf:.1%}** — "
                        f"Box: `({x1}, {y1})` to `({x2}, {y2})`"
                    )
        else:
            st.warning("No parasites detected at the current confidence threshold.")

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="status-empty">
            <h3>Ready when you are.</h3>
            <p>Upload one or more microscopy images to see original images, detection overlays, and confidence details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
