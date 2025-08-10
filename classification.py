import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import os
from PIL import Image

# CONFIGURATION

IMG_SIZE = (224, 224)

# Google Drive direct download links
MODEL_URLS = {
    "Small CNN": "https://drive.google.com/uc?id=1qLGm7pg1Ltrk9239p-imtUbYWVQta1nd",
    "ResNet50 (Fine-tuned)": "https://drive.google.com/uc?id=1SmzFv-kZYJnWiQpJ2jkzNEqY8DtXRDYb"
}

# Replace this list with your actual dataset class names
CLASS_NAMES = ["Fish1", "Fish2", "Fish3", "Fish4", "Fish5"]

# DOWNLOAD FUNCTION
def download_model_from_drive(model_name):
    """Download model from Google Drive if not already present."""
    url = MODEL_URLS[model_name]
    output_path = f"{model_name.replace(' ', '_')}.h5"

    if not os.path.exists(output_path):
        st.write(f"⬇️ Downloading {model_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write(f"✅ {model_name} downloaded successfully.")

    return output_path

# LOAD MODELS
@st.cache_resource
def load_model_from_drive(model_name):
    file_path = download_model_from_drive(model_name)
    return tf.keras.models.load_model(file_path)

# IMAGE PREPROCESSING
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# STREAMLIT APP UI
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Auto Model Selection - Fish Image Classification")
st.write("Upload an image of a fish — the app will run both models and select the best one.")

# Load models only once (cached)
st.write("🔄 Loading models, please wait...")
cnn_model = load_model_from_drive("Small CNN")
resnet_model = load_model_from_drive("ResNet50 (Fine-tuned)")

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Predict with both models
    preds_cnn = cnn_model.predict(img_array)[0]
    preds_resnet = resnet_model.predict(img_array)[0]

    # Auto model selection
    max_conf_cnn = np.max(preds_cnn)
    max_conf_resnet = np.max(preds_resnet)

    if max_conf_cnn >= max_conf_resnet:
        best_model = "Small CNN"
        best_preds = preds_cnn
    else:
        best_model = "ResNet50 (Fine-tuned)"
        best_preds = preds_resnet

    # Adjust class names if mismatch
    if len(CLASS_NAMES) != len(best_preds):
        CLASS_NAMES = [f"Class{i+1}" for i in range(len(best_preds))]

    # Display results
    best_idx = np.argmax(best_preds)
    st.subheader(f"✅ Best Model Selected: {best_model}")
    st.write(f"**Prediction:** {CLASS_NAMES[best_idx]} ({best_preds[best_idx]:.2f})")

    # Top 3 predictions
    st.subheader("Top 3 Predictions:")
    top_indices = best_preds.argsort()[-3:][::-1]
    for idx in top_indices:
        st.write(f"- {CLASS_NAMES[idx]}: {best_preds[idx]:.2f}")
