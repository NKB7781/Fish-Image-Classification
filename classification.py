import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# CONFIG
IMG_SIZE = (224, 224)

# Google Drive file IDs for your models
MODEL_IDS = {
    "Small CNN": "1qLGm7pg1Ltrk9239p-imtUbYWVQta1nd",
    "ResNet50 (Fine-tuned)": "1SmzFv-kZYJnWiQpJ2jkzNEqY8DtXRDYb"
}

MODEL_FILES = {
    "Small CNN": "small_cnn.h5",
    "ResNet50 (Fine-tuned)": "resnet50_finetuned.h5"
}

CLASS_NAMES = [
    "Class1", "Class2", "Class3", "Class4"
]

# STREAMLIT UI
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and select a model to classify it.")

# Model selection
model_choice = st.selectbox("Choose a model:", list(MODEL_IDS.keys()))

# Download model from Google Drive if not exists
@st.cache_resource
def load_model_from_drive(model_name):
    file_path = MODEL_FILES[model_name]
    if not os.path.exists(file_path):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_IDS[model_name]}", file_path, quiet=False)
    return tf.keras.models.load_model(file_path)

model = load_model_from_drive(model_choice)

# Prediction
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    # Top-3
    st.write("### Top-3 Predictions:")
    top_indices = predictions[0].argsort()[-3:][::-1]
    for idx in top_indices:
        st.write(f"{CLASS_NAMES[idx]}: {predictions[0][idx]:.2f}")


