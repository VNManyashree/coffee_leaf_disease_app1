import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "coffee_leaf_model.tflite"
CLASS_NAMES_PATH = "class_names.json"
ICON_PATH = "icon.png"

icon = Image.open(ICON_PATH)

DISEASE_INFO = {
    "Healthy": {
        "status": "Healthy leaf",
        "cause": "No visible disease symptoms detected.",
        "advice": "Maintain proper irrigation and nutrition."
    },
    "Rust": {
        "status": "Coffee Leaf Rust",
        "cause": "Usually caused by Hemileia vastatrix fungus.",
        "advice": "Remove infected leaves and improve airflow."
    },
    "Phoma": {
        "status": "Phoma leaf spot",
        "cause": "Usually associated with fungal infection under wet conditions.",
        "advice": "Improve field sanitation and remove infected leaves."
    },
    "Miner": {
        "status": "Leaf miner damage",
        "cause": "Caused by insects feeding inside the leaf tissue.",
        "advice": "Use pest monitoring and suitable control methods."
    },
    "Cerscospora": {
        "status": "Cercospora leaf spot",
        "cause": "Usually caused by Cercospora fungi.",
        "advice": "Reduce plant stress and improve nutrition."
    }
}

st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    page_icon=icon,
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(170, 196, 183, 0.18), transparent 25%),
        radial-gradient(circle at bottom right, rgba(214, 222, 216, 0.22), transparent 28%),
        linear-gradient(135deg, #f4f1eb 0%, #eef2ee 50%, #e7ece7 100%);
    color: #203128;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(250,252,250,0.80));
    border: 1px solid rgba(60, 85, 70, 0.08);
    border-radius: 28px;
    padding: 34px 32px;
    box-shadow: 0 18px 40px rgba(27, 39, 31, 0.08);
    margin-bottom: 24px;
}

.hero-kicker {
    color: #6f8a79;
    font-size: 0.85rem;
    letter-spacing: 0.16rem;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 12px;
    color: #22332b;
}

.hero-sub {
    color: #55685d;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 720px;
}

.card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(60, 85, 70, 0.08);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 32px rgba(27, 39, 31, 0.07);
    margin-bottom: 20px;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #24362d;
    margin-bottom: 10px;
}

.metric-pill {
    display: inline-block;
    padding: 10px 16px;
    border-radius: 999px;
    background: #edf3ee;
    border: 1px solid #d9e4dc;
    color: #2b4337;
    font-weight: 700;
    margin-bottom: 14px;
}

.result-name {
    font-size: 2rem;
    font-weight: 800;
    color: #22332b;
    margin: 0.4rem 0;
}

.result-status {
    color: #617569;
    font-size: 1.02rem;
    margin-bottom: 0.9rem;
}

.info-box {
    background: #f7faf7;
    border-left: 4px solid #a9c3b0;
    padding: 14px 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    color: #2d4035;
}

.note {
    color: #64766b;
    font-size: 0.96rem;
    line-height: 1.6;
}

[data-testid="stFileUploader"] {
    background: #f8fbf8;
    border: 1px dashed #bfd0c4;
    padding: 18px;
    border-radius: 20px;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 2.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-kicker">Coffee Leaf Analysis</div>
    <div class="hero-title">Diagnose Leaf Health With a Cleaner Interface</div>
    <div class="hero-sub">
        Upload a coffee leaf image and get the predicted disease, confidence score,
        likely cause, and practical treatment guidance in one focused screen.
    </div>
</div>
""", unsafe_allow_html=True)

for path in [MODEL_PATH, CLASS_NAMES_PATH, ICON_PATH]:
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

@st.cache_resource
def load_model_and_classes():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return interpreter, class_names

def preprocess_image(file_data):
    image = Image.open(file_data).convert("RGB")
    preview = image.copy()
    resized = image.resize(IMG_SIZE)
    image_array = np.array(resized).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)
    return preview, image_array

def predict_tflite(interpreter, input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:
        model_input = input_image
    else:
        model_input = input_image.astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output

def render_prediction(file_data, interpreter, class_names):
    preview, input_image = preprocess_image(file_data)
    prediction = predict_tflite(interpreter, input_image)

    pred_index = int(np.argmax(prediction))
    pred_class = class_names[pred_index]
    confidence = float(prediction[pred_index]) * 100

    info = DISEASE_INFO.get(pred_class, {
        "status": pred_class,
        "cause": "Cause information not available.",
        "advice": "Check class label spelling."
    })

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="card"><div class="section-title">Leaf Preview</div>', unsafe_allow_html=True)
        st.image(preview, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Diagnosis Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-pill">Confidence {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{pred_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-status">{info["status"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><b>Cause</b><br>{info["cause"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><b>Advice</b><br>{info["advice"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

interpreter, class_names = load_model_and_classes()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="note">Use one clear coffee leaf image with good lighting and a visible disease area.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    render_prediction(uploaded_file, interpreter, class_names)
else:
    st.markdown("""
    <div class="card">
        <div class="section-title">How This Screen Works</div>
        <div class="note">
            Upload a leaf image and the model will return the most likely class,
            confidence score, cause, and treatment guidance in a single result view.
        </div>
    </div>
    """, unsafe_allow_html=True)
