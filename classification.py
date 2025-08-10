import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import os
from PIL import Image

# Google Drive File IDs
MODEL_IDS = {
    "Small CNN": "1qLGm7pg1Ltrk9239p-imtUbYWVQta1nd",
    "ResNet50 (Fine-tuned)": "1SmzFv-kZYJnWiQpJ2jkzNEqY8DtXRDYb"
}

# Class names (replace with your dataset's actual classes)
CLASS_NAMES = ['Fish1', 'Fish2', 'Fish3', 'Fish4', 'Fish5']

# Download file from Google Drive using requests
def download_model_from_drive(model_name):
    file_id = MODEL_IDS[model_name]
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = f"{model_name.replace(' ', '_')}.h5"

    if not os.path.exists(output_path):
        st.write(f"⬇️ Downloading {model_name}...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return output_path

@st.cache_resource
def load_model_from_drive(model_name):
    file_path = download_model_from_drive(model_name)
    return tf.keras.models.load_model(file_path)

# Load both models at start
st.write("🔄 Loading models...")
cnn_model = load_model_from_drive("Small CNN")
resnet_model = load_model_from_drive("ResNet50 (Fine-tuned)")

st.title("🐟 Auto Model Selection - Fish Image Classification")
st.write("Upload an image of a fish. The app will pick the best model automatically.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    img = image.resize((224, 224))  # Adjust to your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Predict with both models
    preds_cnn = cnn_model.predict(img_array)[0]
    preds_resnet = resnet_model.predict(img_array)[0]

    # Compare max confidence
    max_conf_cnn = np.max(preds_cnn)
    max_conf_resnet = np.max(preds_resnet)

    if max_conf_cnn >= max_conf_resnet:
        best_model = "Small CNN"
        best_preds = preds_cnn
    else:
        best_model = "ResNet50 (Fine-tuned)"
        best_preds = preds_resnet

    # Display best model prediction
    st.subheader(f"✅ Best Model Selected: {best_model}")
    best_idx = np.argmax(best_preds)
    st.write(f"Prediction: **{CLASS_NAMES[best_idx]}** ({best_preds[best_idx]:.2f})")

    # Top 3 predictions
    st.subheader("Top 3 Predictions:")
    top_indices = np.argsort(best_preds)[::-1][:3]
    for idx in top_indices:
        st.write(f"- {CLASS_NAMES[idx]}: {best_preds[idx]:.2f}")

