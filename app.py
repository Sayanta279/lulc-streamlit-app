import streamlit as st
import numpy as np
import rasterio
import gdown
import os
import tensorflow as tf
import cv2
from tempfile import NamedTemporaryFile
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt

st.set_page_config(page_title="LULC Prediction App", layout="centered")
st.title("🌍 LULC Prediction Web App (U-Net)")

MODEL_PATH = "unet_lulc_model.keras"
MODEL_DRIVE_LINK = "https://drive.google.com/uc?id=1yaCVYa4Cw0D9QEi1-EbuMtuyhPXRH-PI"

# Step 1: Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Google Drive..."):
            gdown.download(MODEL_DRIVE_LINK, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully.")

# Step 2: Load model
@st.cache_resource
def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Step 3: Preprocess TIFF image
def preprocess_tiff(tiff_file, target_shape=(256, 256)):
    with rasterio.open(tiff_file) as src:
        image = src.read(1)
        original_shape = image.shape
        image_resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_NEAREST)
        image_resized = image_resized.astype(np.float32)
        image_resized /= np.max(image_resized)
        input_tensor = np.expand_dims(image_resized, axis=(0, -1))
    return input_tensor, original_shape, image

# Step 4: Postprocess prediction
def postprocess_prediction(prediction, original_shape):
    pred_resized = cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_NEAREST)
    classified = np.digitize(pred_resized, bins=[0.2, 0.4, 0.6, 0.8]) + 1
    return classified

# Step 5: Display maps
def display_results(original, prediction):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("🛰️ Original TIFF")
    ax1.imshow(original, cmap='gray')
    ax2.set_title("🧾 Predicted LULC Classes")
    ax2.imshow(prediction, cmap='tab10')
    st.pyplot(fig)

# Load model
model = load_model()

# Upload TIFF
uploaded_file = st.file_uploader("📂 Upload a TIFF file for prediction", type=["tif", "tiff"])
if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("📄 File uploaded successfully.")
    
    input_tensor, original_shape, original = preprocess_tiff(tmp_path)
    prediction = model.predict(input_tensor)[0, :, :, 0]
    classified = postprocess_prediction(prediction, original_shape)
    
    display_results(original, classified)
