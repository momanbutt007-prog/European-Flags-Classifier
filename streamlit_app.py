from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="European Flags AI",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS  (UNCHANGED)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "FINAL_European_Flags_EfficientNetB2.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR
    / "class_names.json"
)

IMG_SIZE = (260, 260)

ALLOWED_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png",
    "webp",
]


# ============================================================
# EUROPEAN FLAGS
# ============================================================

DEFAULT_CLASS_NAMES = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Holland",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Slovakia",
    "Slovenia",
    "South Cyprus",
    "Spain",
    "Sweden",
]


# ============================================================
# CUSTOM THEME / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Google Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif !important;
    }

    /* ---------- App background: deep professional navy/teal ---------- */
    .stApp {
        background:
            radial-gradient(circle at 12% -8%, rgba(45, 212, 191, 0.10) 0%, transparent 42%),
            radial-gradient(circle at 88% 6%, rgba(99, 102, 241, 0.14) 0%, transparent 50%),
            radial-gradient(circle at 50% 100%, rgba(56, 189, 248, 0.06) 0%, transparent 55%),
            linear-gradient(160deg, #0a1120 0%, #0d1830 45%, #081020 100%);
        color: #eef1fa;
    }

    /* ---------- Hero banner ---------- */
    .hero-banner {
        padding: 2.2rem 2.5rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #0f766e 0%, #2563eb 55%, #7c3aed 100%);
        box-shadow: 0 20px 45px rgba(20, 60, 120, 0.35);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.9);
        margin: 0 0 0.4rem 0;
        font-weight: 500;
    }

    .hero-desc {
        font-size: 0.92rem;
        color: rgba(255,255,255,0.75);
        max-width: 640px;
    }

    /* ---------- Glass cards ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        height: 100%;
    }

    .glass-card h4 {
        margin-top: 0;
        color: #ffffff;
        font-size: 1.05rem;
    }

    .glass-card p {
        color: rgba(255,255,255,0.72);
        font-size: 0.88rem;
        margin-bottom: 0;
    }

    /* ---------- Metric-like pill row ---------- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }

    div[data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.65) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif;
    }

    /* ---------- Section dividers ---------- */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ---------- Showcase card wrapping the uploaded image / prediction ---------- */
    .showcase-card {
        background: linear-gradient(165deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 1.3rem 1.3rem 1.5rem 1.3rem;
        box-shadow: 0 16px 42px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.05);
        height: 100%;
    }

    .showcase-header {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.08rem;
        color: #ffffff;
        margin-bottom: 0.9rem;
    }

    .showcase-header .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(120deg, #2dd4bf, #7c3aed);
        box-shadow: 0 0 10px rgba(124,58,237,0.8);
    }

    .image-frame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 28px rgba(0,0,0,0.42);
        margin-bottom: 0.5rem;
    }

    .image-frame img {
        display: block;
        width: 100%;
    }

    .image-caption-pill {
        display: inline-block;
        margin-top: 0.6rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.09);
        color: rgba(255,255,255,0.65);
        font-size: 0.78rem;
    }

    /* ---------- Winner / prediction hero card ---------- */
    .winner-card {
        position: relative;
        background: linear-gradient(135deg, rgba(15,118,110,0.35), rgba(124,58,237,0.30));
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 20px;
        padding: 1.6rem 1.6rem 1.4rem 1.6rem;
        text-align: center;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }

    .winner-card::before {
        content: "";
        position: absolute;
        top: -40%;
        left: -20%;
        width: 160%;
        height: 160%;
        background: radial-gradient(circle at 30% 20%, rgba(45,212,191,0.18), transparent 55%);
        pointer-events: none;
    }

    .winner-flag {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.25rem 0 0.1rem 0;
        font-family: 'Poppins', sans-serif;
        position: relative;
    }

    .winner-label {
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.65);
        position: relative;
    }

    .winner-confidence-badge {
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: #ffe08a;
        font-weight: 700;
        font-size: 0.95rem;
        position: relative;
    }

    /* ---------- Rank row cards ---------- */
    .rank-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.55rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 30% 0%, rgba(45,212,191,0.08), transparent 45%),
            radial-gradient(circle at 90% 40%, rgba(124,58,237,0.10), transparent 50%),
            linear-gradient(180deg, #0c1226 0%, #060a16 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #eef1fa !important;
    }

    .sb-logo-wrap {
        text-align: center;
        padding: 1rem 0 0.6rem 0;
    }

    .sb-logo-badge {
        width: 62px;
        height: 62px;
        margin: 0 auto 0.6rem auto;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #0f766e, #2563eb 55%, #7c3aed);
        box-shadow: 0 10px 26px rgba(37,99,235,0.35);
    }

    .sb-title {
        font-size: 1.28rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 0.15rem;
    }

    .sb-subtitle {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.5) !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .sb-section-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
        color: #ffffff !important;
    }

    .sb-section-label .icon-chip-sm {
        width: 26px;
        height: 26px;
        min-width: 26px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        background: linear-gradient(135deg, rgba(45,212,191,0.28), rgba(124,58,237,0.28));
        border: 1px solid rgba(255,255,255,0.12);
    }

    .sb-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-top: 0.3rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    }

    .sb-card p {
        margin: 0.2rem 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.78) !important;
    }

    .sb-tip-row {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        font-size: 0.83rem;
        color: rgba(255,255,255,0.78) !important;
        margin: 0.35rem 0;
    }

    /* ---------- File uploader ---------- */
    div[data-testid="stFileUploaderDropzone"] {
        background:
            radial-gradient(circle at 20% 15%, rgba(45,212,191,0.10), transparent 55%),
            radial-gradient(circle at 80% 85%, rgba(124,58,237,0.10), transparent 55%),
            rgba(255,255,255,0.03);
        border: 2px dashed rgba(45, 212, 191, 0.45);
        border-radius: 20px;
        padding: 1.6rem 1.2rem;
        transition: all 0.25s ease;
    }

    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(45, 212, 191, 0.85);
        background:
            radial-gradient(circle at 20% 15%, rgba(45,212,191,0.16), transparent 55%),
            radial-gradient(circle at 80% 85%, rgba(124,58,237,0.16), transparent 55%),
            rgba(255,255,255,0.045);
    }

    div[data-testid="stFileUploaderDropzone"] svg {
        color: #2dd4bf !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: rgba(255,255,255,0.55) !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(120deg, #0f766e, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 18px rgba(20,60,120,0.35);
    }

    div[data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 0.3rem 0.6rem;
    }

    /* ---------- Upload / status showcase wrapper ---------- */
    .upload-card {
        background: linear-gradient(165deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 1.3rem 1.3rem 0.6rem 1.3rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.35);
        height: 100%;
    }

    .upload-card-header {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }

    .upload-card-header .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(120deg, #2dd4bf, #7c3aed);
        box-shadow: 0 0 10px rgba(124,58,237,0.8);
    }

    .status-badge-row {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        margin-top: 0.4rem;
    }

    .status-badge {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 0.6rem 0.85rem;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.85);
    }

    .status-badge .icon-chip {
        width: 30px;
        height: 30px;
        min-width: 30px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        background: linear-gradient(135deg, rgba(45,212,191,0.25), rgba(124,58,237,0.25));
        border: 1px solid rgba(255,255,255,0.1);
    }

    .status-ready-tag {
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.68rem;
        font-weight: 700;
        color: #2dd4bf;
        margin-bottom: 0.3rem;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: linear-gradient(120deg, #0f766e, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(20,60,120,0.35);
    }

    /* ---------- Progress bars ---------- */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #2dd4bf, #7c3aed) !important;
    }

    /* ---------- Footer caption ---------- */
    .footer-caption {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        padding-top: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if CLASS_NAMES_PATH.exists():

        try:

            with open(
                CLASS_NAMES_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(data, dict):

                # Handle {"0": "Austria", ...}
                if all(
                    str(key).isdigit()
                    for key in data.keys()
                ):
                    return [
                        data[key]
                        for key in sorted(
                            data.keys(),
                            key=lambda x: int(x),
                        )
                    ]

                return list(data.values())

            if isinstance(data, list):
                return data

        except Exception:
            pass

    return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, (
            f"Model not found: {MODEL_PATH}"
        )

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        return model, None

    except Exception as exc:

        return None, str(exc)


model, model_error = load_model()


# ============================================================
# EXACT FLASK PREPROCESSING  (UNCHANGED)
# ============================================================

def prepare_image(image):
    """
    This intentionally matches the Flask app:

        image = Image.open(path).convert("RGB")
        image = image.resize(
            IMG_SIZE,
            Image.Resampling.LANCZOS
        )
        array = np.asarray(
            image,
            dtype=np.float32
        )
        return np.expand_dims(array, axis=0)

    IMPORTANT:
    No /255 normalization.
    """

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    return np.expand_dims(
        array,
        axis=0,
    )


# ============================================================
# EXACT FLASK PREDICTION LOGIC  (UNCHANGED)
# ============================================================

def predict_flag(image):

    if model is None:

        raise RuntimeError(
            model_error
            or "Model could not be loaded."
        )

    batch = prepare_image(image)

    probabilities = model.predict(
        batch,
        verbose=0,
    )[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    predictions = []

    for index in top_indices:

        index = int(index)

        predictions.append(
            {
                "class_name": CLASS_NAMES[index],
                "confidence": float(
                    probabilities[index] * 100
                ),
            }
        )

    return predictions


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sb-logo-wrap">
            <div class="sb-logo-badge">🇪🇺</div>
            <div class="sb-title">European Flags AI</div>
            <div class="sb-subtitle">Computer Vision • Deep Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div class="sb-section-label"><span class="icon-chip-sm">🤖</span> Model Status</div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:

        st.markdown(
            """
            <div class="sb-card" style="border-color: rgba(45,212,191,0.35); margin-bottom:0.6rem;">
                <p style="color:#5eead4 !important; font-weight:700;">● Model Online</p>
            </div>
            <div class="sb-card">
                <p>🧠 <b>EfficientNetB2</b></p>
                <p>📐 Input: 260 × 260</p>
                <p>⚙️ TensorFlow / Keras</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.error("Model Error")

        if model_error:
            st.caption(model_error)

    st.divider()

    st.markdown(
        """
        <div class="sb-section-label"><span class="icon-chip-sm">🌍</span> Classes</div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "European Flags",
        len(CLASS_NAMES),
    )

    with st.expander(
        "View supported countries"
    ):

        for number, country in enumerate(
            CLASS_NAMES,
            start=1,
        ):

            st.write(
                f"{number}. {country}"
            )

    st.divider()

    st.markdown(
        """
        <div class="sb-section-label"><span class="icon-chip-sm">📷</span> Image Tips</div>
        <div class="sb-card">
            <div class="sb-tip-row">✅ Use a clear, well-lit image</div>
            <div class="sb-tip-row">✅ Keep the flag centered</div>
            <div class="sb-tip-row">✅ Avoid heavy filters or glare</div>
            <div class="sb-tip-row" style="margin-top:0.5rem; color: rgba(255,255,255,0.55) !important;">
                Supported: JPG, JPEG, PNG, WEBP
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER (hero banner)
# ============================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🇪🇺 European Flags AI</div>
        <div class="hero-subtitle">Identify European flags with Artificial Intelligence</div>
        <div class="hero-desc">
            Upload a flag image and the trained EfficientNetB2 model will analyze it
            and return the top predictions, instantly and locally.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.metric(
        "🌍 Countries",
        len(CLASS_NAMES),
    )

with info2:

    st.metric(
        "🧠 Model",
        "EfficientNetB2",
    )

with info3:

    st.metric(
        "📐 Input",
        "260 × 260",
    )

with info4:

    st.metric(
        "⚡ Inference",
        "Local",
    )


st.divider()


# ============================================================
# MODEL ERROR
# ============================================================

if model is None:

    st.error(
        "The European Flags model could not be loaded."
    )

    st.code(
        "models/FINAL_European_Flags_EfficientNetB2.keras"
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.header("📸 Flag Recognition")

upload_col, status_col = st.columns([1.3, 1])

with upload_col:

    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-card-header"><span class="dot"></span> 📤 Upload Your Flag</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your flag image here, or browse from your computer",
        type=ALLOWED_EXTENSIONS,
        help=(
            "Upload a clear JPG, JPEG, PNG, "
            "or WEBP image."
        ),
        label_visibility="visible",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with status_col:

    if uploaded_file is None:

        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-header"><span class="dot"></span> ✨ Ready to Analyze</div>
                <div class="status-ready-tag">Waiting for image</div>
                <p style="color:rgba(255,255,255,0.68); font-size:0.9rem; margin-bottom:1rem;">
                    Upload a clear photo of a European flag and the model will
                    predict the country with a full confidence breakdown.
                </p>
                <div class="status-badge-row">
                    <div class="status-badge"><span class="icon-chip">🌍</span> 24 European flag classes</div>
                    <div class="status-badge"><span class="icon-chip">🧠</span> EfficientNetB2 architecture</div>
                    <div class="status-badge"><span class="icon-chip">🏆</span> Top-5 confidence ranking</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-header"><span class="dot"></span> ✅ Image Received</div>
                <div class="status-ready-tag" style="color:#7dd3fc;">Analyzing below</div>
                <p style="color:rgba(255,255,255,0.68); font-size:0.9rem; margin-bottom:1rem;">
                    Your image has been uploaded successfully. Scroll down to see
                    the predicted country and full confidence breakdown.
                </p>
                <div class="status-badge-row">
                    <div class="status-badge"><span class="icon-chip">📄</span> """ + uploaded_file.name + """</div>
                    <div class="status-badge"><span class="icon-chip">🧠</span> EfficientNetB2 • 260×260</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload an image above to start "
        "the AI prediction."
    )

    st.write("### 🚩 What this app does")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:

        st.markdown(
            """
            <div class="glass-card">
                <h4>🧠 AI Recognition</h4>
                <p>Uses your trained EfficientNetB2 classification model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature2:

        st.markdown(
            """
            <div class="glass-card">
                <h4>🎯 Top 5 Results</h4>
                <p>Shows the five most likely European flag classes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature3:

        st.markdown(
            """
            <div class="glass-card">
                <h4>📊 Confidence</h4>
                <p>Displays the model's confidence for each prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div class="footer-caption">European Flags AI • TensorFlow / Keras • Streamlit</div>',
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except (
    UnidentifiedImageError,
    OSError,
):

    st.error(
        "The uploaded file is not a valid "
        "readable image."
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

with st.spinner(
    "🔎 Analyzing the European flag..."
):

    try:

        predictions = predict_flag(
            image
        )

    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.exception(exc)

        st.stop()


prediction = predictions[0]


# ============================================================
# IMAGE + MAIN RESULT (showcase cards)
# ============================================================

st.divider()

image_col, result_col = st.columns(
    [1.05, 1]
)


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------

with image_col:

    st.markdown(
        """
        <div class="showcase-card">
            <div class="showcase-header"><span class="dot"></span> 🖼️ Uploaded Image</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="image-frame">', unsafe_allow_html=True)

    st.image(
        image,
        use_container_width=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
            <span class="image-caption-pill">📄 {uploaded_file.name}</span>
            <span class="image-caption-pill">📐 {image.width} × {image.height}px</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# MAIN PREDICTION
# ------------------------------------------------------------

with result_col:

    confidence = (
        prediction["confidence"] / 100
    )

    if prediction["confidence"] >= 90:
        badge_html = '<div style="margin-top:0.7rem; color:#7CFFB2; font-weight:600;">🔥 Very high confidence</div>'
    elif prediction["confidence"] >= 75:
        badge_html = '<div style="margin-top:0.7rem; color:#7dd3fc; font-weight:600;">👍 Good confidence</div>'
    else:
        badge_html = '<div style="margin-top:0.7rem; color:#fbbf24; font-weight:600;">⚠️ Lower confidence — try a clearer image</div>'

    st.markdown(
        f"""
        <div class="showcase-card">
            <div class="showcase-header"><span class="dot"></span> 🎯 Prediction</div>
            <div class="winner-card">
                <div class="winner-label">Top Match</div>
                <div class="winner-flag">🇪🇺 {prediction['class_name']}</div>
                <div class="winner-confidence-badge">{prediction['confidence']:.2f}% confidence</div>
                {badge_html}
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.progress(
        min(max(confidence, 0.0), 1.0),
        text=(
            f"Confidence: "
            f"{prediction['confidence']:.2f}%"
        ),
    )

    st.markdown(
        '<p style="color:rgba(255,255,255,0.6); font-size:0.85rem; margin-top:0.4rem;">'
        'The prediction above is the model\'s highest-probability class.</p></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TOP 5
# ============================================================

st.divider()

st.header(
    "🏆 Top 5 Predictions"
)

st.caption(
    "These values come directly from the model's "
    "output probabilities, matching your Flask app."
)

MEDALS = ["🥇", "🥈", "🥉"]

for rank, item in enumerate(
    predictions,
    start=1,
):

    country = item["class_name"]

    confidence = item["confidence"]

    medal = MEDALS[rank - 1] if rank <= 3 else f"#{rank}"

    left, middle, right = st.columns(
        [0.5, 3.2, 1]
    )

    with left:

        st.markdown(
            f"<div style='font-size:1.5rem; text-align:center;'>{medal}</div>",
            unsafe_allow_html=True,
        )

    with middle:

        st.write(
            f"**{country}**"
        )

        st.progress(
            min(
                max(
                    confidence / 100,
                    0.0,
                ),
                1.0,
            )
        )

    with right:

        st.markdown(
            f"<div style='font-weight:700; color:#ffd166; text-align:right; padding-top:0.4rem;'>{confidence:.2f}%</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# DETAILED RESULTS
# ============================================================

st.divider()

with st.expander(
    "📊 View detailed prediction data"
):

    for rank, item in enumerate(
        predictions,
        start=1,
    ):

        st.write(
            f"**#{rank} — "
            f"{item['class_name']}**"
        )

        st.write(
            f"Confidence: "
            f"{item['confidence']:.6f}%"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "🧠 Model Information"
):

    st.markdown(
        """
        <div class="glass-card">
            <p><b>Architecture:</b> EfficientNetB2</p>
            <p><b>Input size:</b> 260 × 260 × 3</p>
            <p><b>Classes:</b> {n_classes}</p>
            <p><b>Inference framework:</b> TensorFlow / Keras</p>
            <p><b>Preprocessing:</b> RGB → 260 × 260 → float32 → batch</p>
            <p><b>Normalization:</b> None</p>
        </div>
        """.format(n_classes=len(CLASS_NAMES)),
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-caption">🇪🇺 European Flags AI &nbsp;•&nbsp; '
    'EfficientNetB2 &nbsp;•&nbsp; TensorFlow / Keras &nbsp;•&nbsp; Streamlit</div>',
    unsafe_allow_html=True,
)