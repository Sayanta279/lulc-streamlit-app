
import streamlit as st
import os
import numpy as np
import rasterio
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils.predictor import predict_tiff, classify_prediction, show_images
from tempfile import NamedTemporaryFile
import gdown

# App Title
st.set_page_config(layout="wide")
st.title("🌍 Land Use Land Cover (LULC) Classification with Deep Learning")

# Load Model from Google Drive
@st.cache_resource
def load_model():
    model_path = "model/model.h5"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        url = "https://drive.google.com/uc?id=1vkeZmAIzop8K5MdsIK8o70xpEHN7YVH6"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
st.success("✅ Model Loaded Successfully")

# File uploader
uploaded_files = st.file_uploader("📂 Upload TIFF files (at least 2)", type=["tif", "tiff"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    input_images = []
    predicted_images = []
    classified_images = []

    st.info("⏳ Processing uploaded files...")

    for file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".tif") as temp_input:
            temp_input.write(file.read())
            temp_input_path = temp_input.name

        predicted_path = temp_input_path.replace(".tif", "_predicted.tif")
        classified_path = temp_input_path.replace(".tif", "_classified.tif")

        input_img, pred_img = predict_tiff(temp_input_path, predicted_path, model)
        class_img = classify_prediction(pred_img, classified_path, temp_input_path)

        input_images.append(input_img)
        predicted_images.append(pred_img)
        classified_images.append(class_img)

    st.success("✅ Prediction and Classification Completed")
    show_images(input_images, predicted_images, classified_images)
else:
    st.warning("⚠️ Please upload at least two TIFF files to proceed.")
