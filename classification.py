import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
# Google Drive file IDs for models
MODEL_IDS = {
    "Small CNN": "1qLGm7pg1Ltrk9239p-imtUbYWVQta1nd",
    "ResNet50 (Fine-tuned)": "1SmzFv-kZYJnWiQpJ2jkzNEqY8DtXRDYb"
}

MODEL_FILES = {
    "Small CNN": "small_cnn.h5",
    "ResNet50 (Fine-tuned)": "resnet50_finetuned.h5"
}

# Replace with actual class names from training
CLASS_NAMES = [
    "Class1", "Class2", "Class3", "Class4"
]

# STREAMLIT UI
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and select a model to classify it.")

# Model selection
model_choice = st.selectbox("Choose a model:", list(MODEL_IDS.keys()))

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model = load_model(MODEL_PATHS[model_choice])

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    # Show top-3 predictions
    st.write("### Top-3 Predictions:")
    top_indices = predictions[0].argsort()[-3:][::-1]
    for idx in top_indices:
        st.write(f"{CLASS_NAMES[idx]}: {predictions[0][idx]:.2f}")


