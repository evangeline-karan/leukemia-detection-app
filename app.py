import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Leukemia Detection",
    page_icon="🧬",
    layout="wide"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: #ff4b4b;
    text-align: center;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: rgba(255,255,255,0.1);
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    color: white;
}

.explanation-box {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ================== SETTINGS ==================
MODEL_PATH = r"C:\Users\evang\OneDrive\Documents\Leukimia detection\mobilenet_leukemia_model.h5"
IMG_SIZE = (224, 224)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# ================== TITLE ==================
st.title("🧬 Leukemia Detection using Deep Learning")
st.markdown("""
<div class='explanation-box'>
This AI system analyzes blood smear images and predicts whether the sample shows signs of Leukemia.  
It also generates a <b>Grad-CAM Heatmap</b> to visualize which regions influenced the model's decision.
</div>
""", unsafe_allow_html=True)

st.write("")

# ================== IMAGE UPLOAD ==================
uploaded_file = st.file_uploader(
    "📤 Upload Blood Smear Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img_original, IMG_SIZE)

    # Preprocess
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ================== PREDICTION ==================
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        label = "🩸 Leukemia Detected"
        color = "red"
    else:
        label = "✅ Normal"
        color = "lightgreen"

    st.markdown(
        f"""
        <div class='prediction-box'>
        Prediction: <span style='color:{color}'>{label}</span><br>
        Confidence Score: {pred:.4f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ================== GRAD-CAM ==================
    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        last_conv_layer = "Conv_1"

    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
    heatmap = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        img_resized,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    # ================== DISPLAY ==================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_resized, caption="Original Image", channels="BGR")

    with col2:
        st.image(colored_heatmap, caption="Grad-CAM Heatmap", channels="BGR")

    with col3:
        st.image(superimposed_img, caption="Final AI Attention Map", channels="BGR")

    # ================== EXPLANATION ==================
    st.write("")

    st.subheader("🧠 Grad-CAM Explanation")

    st.markdown("""
    <div class='explanation-box'>

    <b>What does the Grad-CAM image show?</b><br><br>

    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the important regions
    in the blood smear image that influenced the AI model's prediction.

    🔴 <b>Red/Yellow Areas</b> → Regions the model focused on strongly.<br>
    🔵 <b>Blue Areas</b> → Less important regions.<br><br>

    If the highlighted regions are concentrated around abnormal white blood cells,
    the model predicts Leukemia.

    This helps doctors and researchers understand:
    - Why the AI made a prediction
    - Which cell regions were important
    - Whether the model is focusing on medically relevant areas

    </div>
    """, unsafe_allow_html=True)