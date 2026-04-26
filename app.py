import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tflite_runtime.interpreter as tflite


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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp {
    background:
        radial-gradient(circle at top left, rgba(116, 170, 129, 0.18), transparent 25%),
        radial-gradient(circle at bottom right, rgba(196, 235, 181, 0.12), transparent 25%),
        linear-gradient(135deg, #17352b 0%, #21493b 45%, #0d1f1a 100%);
    color: white;
}
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem; }
.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    border-radius: 28px;
    padding: 34px 32px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.22);
    margin-bottom: 24px;
}
.hero-kicker {
    color: #c8e6c9; font-size: 0.9rem; letter-spacing: 0.14rem;
    font-weight: 700; text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
    font-size: 3rem; font-weight: 800; line-height: 1.05;
    margin-bottom: 12px; color: #ffffff;
}
.hero-sub {
    color: #e4f3ea; font-size: 1.08rem; line-height: 1.7; max-width: 700px;
}
.glass-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.20);
    margin-bottom: 20px;
}
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #effff3; margin-bottom: 10px;
}
.metric-pill {
    display: inline-block; padding: 10px 16px; border-radius: 999px;
    background: rgba(201, 255, 216, 0.12);
    border: 1px solid rgba(201, 255, 216, 0.18);
    color: #eafff0; font-weight: 600; margin-bottom: 14px;
}
.result-name {
    font-size: 2rem; font-weight: 800; color: #ffffff; margin: 0.4rem 0;
}
.result-status {
    color: #cfead7; font-size: 1.05rem; margin-bottom: 0.8rem;
}
.info-box {
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #9be7ab;
    padding: 14px 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    color: #eefaf1;
}
.small-note {
    color: #d7efe0; font-size: 0.95rem; line-height: 1.6;
}
[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.04);
    padding: 14px;
    border-radius: 18px;
}
.stRadio > div {
    background: rgba(255,255,255,0.05);
    padding: 12px 14px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-kicker">AI Powered Coffee Diagnostics</div>
    <div class="hero-title">Coffee Leaf Disease Detection</div>
    <div class="hero-sub">
        Capture a live leaf image or upload one from your gallery to identify the disease,
        view confidence, understand the likely cause, and get practical care guidance.
    </div>
</div>
""", unsafe_allow_html=True)

for path in [MODEL_PATH, CLASS_NAMES_PATH, ICON_PATH]:
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

@st.cache_resource
def load_model_and_classes():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return interpreter, class_names

def preprocess_image(file_data):
    image = Image.open(file_data).convert("RGB")
    display_image = image.copy()
    resized = image.resize(IMG_SIZE)
    image_array = np.array(resized).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)
    return display_image, image_array

def predict_tflite(interpreter, input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if input_details[0]["dtype"] == np.float32:
        model_input = input_image
    else:
        model_input = input_image.astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output

def render_prediction(file_data, interpreter, class_names):
    display_image, input_image = preprocess_image(file_data)
    prediction = predict_tflite(interpreter, input_image)

    pred_index = int(np.argmax(prediction))
    pred_class = class_names[pred_index]
    confidence = float(prediction[pred_index]) * 100

    info = DISEASE_INFO.get(pred_class, {
        "status": pred_class,
        "cause": "Cause information not available.",
        "advice": "Check class label spelling."
    })

    col1, col2 = st.columns([1.05, 1])

    with col1:
        st.markdown('<div class="glass-card"><div class="section-title">Leaf Preview</div>', unsafe_allow_html=True)
        st.image(display_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Diagnosis Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-pill">Confidence {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{pred_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-status">{info["status"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><b>Cause</b><br>{info["cause"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><b>Advice</b><br>{info["advice"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

interpreter, class_names = load_model_and_classes()

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Choose Image Source</div>', unsafe_allow_html=True)
    source = st.radio(
        "Select input mode",
        ["Upload from Gallery", "Take Photo"],
        label_visibility="collapsed"
    )
    st.markdown(
        '<div class="small-note">Use a clear coffee leaf image with proper light and visible disease area for better prediction.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

file_data = None

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Panel</div>', unsafe_allow_html=True)

    if source == "Upload from Gallery":
        file_data = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    else:
        file_data = st.camera_input(
            "Take a photo",
            label_visibility="collapsed"
        )

    st.markdown('</div>', unsafe_allow_html=True)

if file_data is not None:
    render_prediction(file_data, interpreter, class_names)
else:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">How It Works</div>
        <div class="small-note">
            1. Choose gallery or camera.<br>
            2. Add a coffee leaf image.<br>
            3. The trained model detects the disease class.<br>
            4. The app shows confidence, cause, and advice.
        </div>
    </div>
    """, unsafe_allow_html=True)
