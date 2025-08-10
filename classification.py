import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image
from io import BytesIO

# ---------------- CONFIG ---------------- #
MODEL_IDS = {
    "Small CNN": "1qLGm7pg1Ltrk9239p-imtUbYWVQta1nd", 
    "ResNet50 (Fine-tuned)": "1SmzFv-kZYJnWiQpJ2jkzNEqY8DtXRDYb"  
}

CLASS_NAMES = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']  
IMG_SIZE = (224, 224)

# ---------------- FUNCTIONS ---------------- #
def download_large_file_from_drive(file_id, destination):
    """Download large file from Google Drive, handling confirmation token."""
    URL = "https://docs.google.com/uc?export=download"

    if os.path.exists(destination):
        return destination  # Skip download if already present

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    return destination


@st.cache_resource
def load_model_from_drive(model_name):
    file_path = download_large_file_from_drive(MODEL_IDS[model_name], f"{model_name.replace(' ', '_')}.h5")
    return tf.keras.models.load_model(file_path)


def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def predict_with_model(model, image):
    preds = model.predict(image)[0]
    return preds


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="🐟 Multiclass Fish Image Classification", layout="centered")

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and the app will choose the best model to classify it.")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Predict with both models
    predictions = {}
    for model_name in MODEL_IDS.keys():
        with st.spinner(f"Loading {model_name}..."):
            model = load_model_from_drive(model_name)
            preds = predict_with_model(model, img_array)
            predictions[model_name] = preds

    # Choose best model by highest confidence
    best_model = max(predictions, key=lambda m: np.max(predictions[m]))
    best_preds = predictions[best_model]
    top_class = CLASS_NAMES[np.argmax(best_preds)]
    confidence = np.max(best_preds)

    st.subheader(f"✅ Best Model: {best_model}")
    st.write(f"**Prediction:** {top_class} ({confidence:.2f})")

    # Show top 3 predictions
    st.subheader("Top 3 Predictions:")
    top_indices = np.argsort(best_preds)[::-1][:3]
    for idx in top_indices:
        st.write(f"- {CLASS_NAMES[idx]}: {best_preds[idx]:.2f}")
