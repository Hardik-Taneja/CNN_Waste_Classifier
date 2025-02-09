import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(page_title="Waste Classification using CNN", page_icon="♻️", layout="wide")

# Google Drive Model File ID (Replace this with your actual Google Drive file ID)
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE"
MODEL_PATH = "best_model.h5"

# Function to Download Model if Not Found
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... Please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Download model if not present
download_model()

# Load Model with Caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found! Please check the model link.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Define Class Labels
CLASS_NAMES = ["Organic", "Recyclable"]

# Sidebar - About Section
with st.sidebar:
    st.info(
        "### 📌 About This Project\n"
        "**Waste Classification using CNN** is an AI-powered solution that classifies waste into **Organic** and **Recyclable** categories.\n\n"
        "### 🔍 Project Overview\n"
        "- 🚀 **Technology Used:** Convolutional Neural Networks (CNN) for image classification.\n"
        "- 🗂 **Dataset Source:** Techsash (Kaggle)\n"
        "- 📊 **Model Training:** Built and trained using TensorFlow & Keras.\n"
        "- 🎯 **Goal:** Automate waste classification to promote effective recycling and waste management.\n\n"
        "### 🌟 Future Enhancements\n"
        "- Advanced classification into **plastic, glass, metal, and e-waste**.\n"
        "- Integration with **IoT-based waste monitoring systems**.\n"
        "- Deployment as a **mobile/web application for real-world use**.\n\n"
        "**♻️ Let's contribute to a greener planet! 🌍**"
    )

# Main Title
st.markdown("<h1 style='text-align: center;'>♻️ Waste Classification using CNN</h1>", unsafe_allow_html=True)

# Upload Section
st.subheader("📤 Upload an image for classification")
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

# Function to Process and Classify Image
def classify_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Adjust based on model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict using CNN Model
    if model:
        prediction = model.predict(img)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    else:
        return None, None

# Display Uploaded Image & Classify
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    predicted_class, confidence = classify_image(image)
    
    if predicted_class:
        st.success(f"### 🏷 Predicted Class: **{predicted_class}**")
        st.info(f"### 🎯 Confidence: **{confidence:.2f}%**")
    else:
        st.warning("⚠ Model not loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown("### Developed with ❤️ by Hardik Taneja")
st.markdown("### AICTE Internship Student Registration ID: STU6771907e413361735495806")
