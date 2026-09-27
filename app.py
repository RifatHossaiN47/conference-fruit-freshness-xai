"""
Streamlit Web Application
Fresh vs. Rotten Fruit Multi-Task Classifier with Explainable AI (Grad-CAM)
IEEE ICCIT 2025 Paper Implementation
"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src import FRUIT_CLASSES, FRESHNESS_CLASSES
from src.gradcam import MultiTaskGradCAM

# Page Configuration
st.set_page_config(
    page_title="Fresh vs Rotten Fruit Classifier",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling to match professional conference deployment
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .result-card-green {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-top: 10px;
    }
    .result-card-red {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-top: 10px;
    }
    .metric-value-green {
        font-size: 1.5rem;
        font-weight: 700;
        color: #065F46;
    }
    .metric-value-red {
        font-size: 1.5rem;
        font-weight: 700;
        color: #991B1B;
    }
    .metric-sub {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model(weights_path="models/ResNet152.keras"):
    """Loads and caches the multi-task model."""
    from tensorflow import keras
    if not os.path.exists(weights_path):
        return None
    return keras.models.load_model(weights_path)

# Header Section
st.markdown("<h1 class='main-title'>🍎 Fresh vs Rotten Fruit Classifier</h1>", unsafe_allow_html=True)
st.caption("IEEE ICCIT 2025 • Multi-Task ResNet152V2 Deep Learning with Grad-CAM Explainability")

# Mode Selection
mode = st.radio(
    "Choose classification mode:",
    options=["📁 Upload Image", "📷 Live Camera Feed"],
    horizontal=True
)

uploaded_file = None
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image to classify the fruit type and freshness",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("Take a photo of the fruit to classify")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    col_img, _ = st.columns([1, 0.01])
    with col_img:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Classification trigger
    classify_btn = st.button("Classify", type="primary")
    
    if classify_btn or mode == "📷 Live Camera Feed":
        with st.spinner("Analyzing fruit features and surface freshness..."):
            # Load model
            model = get_model()
            
            if model is None:
                st.error("⚠️ Pretrained weights `models/ResNet152.keras` not found. Please refer to `models/README.md` to download the weights file.")
            else:
                # Preprocess
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                batch_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                preds = model.predict(batch_array, verbose=0)
                fruit_probs = preds[0][0]
                fresh_prob = float(preds[1][0][0])
                
                fruit_idx = int(np.argmax(fruit_probs))
                fruit_name = FRUIT_CLASSES[fruit_idx] if fruit_idx < len(FRUIT_CLASSES) else f"Class_{fruit_idx}"
                fruit_conf = float(fruit_probs[fruit_idx]) * 100.0
                
                is_rotten = fresh_prob >= 0.5
                fresh_label = "Rotten" if is_rotten else "Fresh"
                rotten_pct = fresh_prob * 100.0
                
                st.write("")
                # Output cards side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🍓 Fruit Type")
                    st.markdown(f"""
                    <div class='result-card-green'>
                        <div class='metric-value-green'>{fruit_name}</div>
                        <div class='metric-sub'>Confidence: {fruit_conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("### ✨ Freshness")
                    card_class = "result-card-red" if is_rotten else "result-card-green"
                    text_class = "metric-value-red" if is_rotten else "metric-value-green"
                    st.markdown(f"""
                    <div class='{card_class}'>
                        <div class='{text_class}'>{fresh_label}</div>
                        <div class='metric-sub'>Rotten probability: {rotten_pct:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Grad-CAM Explainability Section
                st.write("")
                with st.expander("🔥 Explainable AI (Grad-CAM Visualizations)", expanded=True):
                    st.markdown("Visualizing which visual cues the neural network attended to for each task:")
                    
                    explainer = MultiTaskGradCAM(model)
                    fruit_hm = explainer.generate_heatmap(batch_array, task_output="fruit_output", class_index=fruit_idx)
                    fresh_hm = explainer.generate_heatmap(batch_array, task_output="fresh_output", class_index=0)
                    
                    fruit_overlay = explainer.overlay_heatmap(fruit_hm, np.array(img_resized))
                    fresh_overlay = explainer.overlay_heatmap(fresh_hm, np.array(img_resized))
                    
                    gcol1, gcol2 = st.columns(2)
                    with gcol1:
                        st.image(fruit_overlay, caption=f"Fruit Task Focus: {fruit_name} (Shape / Morphology)", use_container_width=True)
                    with gcol2:
                        st.image(fresh_overlay, caption=f"Freshness Task Focus: {fresh_label} (Texture / Surface Lesions)", use_container_width=True)

# Instructions and metadata accordion
with st.expander("ℹ️ How to use & Paper Details", expanded=False):
    st.markdown("""
    **How to use:**
    1. Upload an image of any supported fruit/vegetable or take a picture using your camera.
    2. Supported fruits: Apple, Banana, Bittergourd, Capsicum, Cucumber, Okra, Orange, Potato, Tomato.
    3. Click **Classify** to evaluate both fruit type and freshness simultaneously.
    4. View the **Grad-CAM Attention Maps** to understand the visual explanation behind each prediction.
    
    **Research Paper:**
    - *Title:* A Multi Task Deep Learning Model for Fruit Detection and Freshness Classification with GradCAM Explainability
    - *Conference:* IEEE 28th ICCIT 2025
    - *DOI:* [10.1109/ICCIT68739.2025.11491294](https://doi.org/10.1109/ICCIT68739.2025.11491294)
    """)
