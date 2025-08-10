import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# CONFIGURATION
IMG_SIZE = (224, 224)

# Google Drive file IDs for your models
MODEL_IDS = {
    "Small CNN": "1qLGm7pg1Ltrk9239p-imtUbYWVQta1nd",
    "ResNet50 (Fine-tuned)": "1SmzFv-kZYJnWiQpJ2jkzNEqY8DtXRDYb"
}

# Local filenames for saved models
MODEL_FILES = {
    "Small CNN": "small_cnn.h5",
    "ResNet50 (Fine-tuned)": "resnet50_finetuned.h5"
}

# Update this list with the actual class names from your dataset
CLASS_NAMES = ["Class1", "Class2", "Class3", "Class4"]

# Helper to download from Drive
def download_from_drive(file_id, dest_path):
    """Download file from Google Drive using HTTP streaming."""
    if os.path.exists(dest_path):
        return
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def load_model_from_drive(model_name):
    file_path = MODEL_FILES[model_name]
    download_from_drive(MODEL_IDS[model_name], file_path)
    return tf.keras.models.load_model(file_path)

# Streamlit interface
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image and choose a model to classify it.")

model_choice = st.selectbox("Choose a model:", list(MODEL_IDS.keys()))
model = load_model_from_drive(model_choice)

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]

    # Adjust CLASS_NAMES if its length doesn't match model output
    if len(CLASS_NAMES) != len(preds):
        CLASS_NAMES = [f"Class{i+1}" for i in range(len(preds))]

    # Get top predictions (max 3)
    top_n = min(3, len(preds))
    top_indices = preds.argsort()[-top_n:][::-1]

    # Display top prediction
    st.subheader(f"Prediction: **{CLASS_NAMES[top_indices[0]]}** ({preds[top_indices[0]]:.2f})")

    # Display all top predictions
    st.write("### Top Predictions:")
    for idx in top_indices:
        st.write(f"- {CLASS_NAMES[idx]}: {preds[idx]:.2f}")
