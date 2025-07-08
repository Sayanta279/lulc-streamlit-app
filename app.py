import streamlit as st
import numpy as np
import rasterio
import os
import cv2
import gdown
import tensorflow as tf
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from rasterio.plot import reshape_as_image

# ------------------- CONFIGURATION -----------------------
MODEL_FOLDER_ID = "1z2Nm7bkE6zxq1_ro0B9JGETkTVk4sjwd"  # Your GDrive model folder ID
MODEL_DIR = "unet_lulc_model_tf"  # Local folder name after download

# ------------------- LOAD MODEL --------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        # Download the entire folder
        gdown.download_folder(id=MODEL_FOLDER_ID, quiet=False, use_cookies=False)
    return tf.keras.models.load_model(MODEL_DIR)

model = load_model()

# ------------------- STREAMLIT UI ------------------------
st.set_page_config(page_title="LULC Predictor", layout="wide")
st.title("🛰️ Land Use Land Cover (LULC) Prediction")

uploaded_file = st.file_uploader("📂 Upload a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with rasterio.open(tmp_path) as src:
        input_data = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        original_shape = input_data.shape

    # Resize to model input size (256x256)
    input_resized = cv2.resize(input_data, (256, 256), interpolation=cv2.INTER_LINEAR)
    input_resized = np.expand_dims(input_resized, axis=(0, -1))  # (1, 256, 256, 1)

    # Predict
    prediction_resized = model.predict(input_resized)[0, :, :, 0]  # (256, 256)
    prediction = cv2.resize(prediction_resized, original_shape[::-1], interpolation=cv2.INTER_LINEAR)

    # Threshold (optional): binarize or classify
    prediction_classified = np.zeros_like(prediction, dtype=np.uint8)
    prediction_classified[prediction > 0.5] = 1  # Binary mask

    # Show visualizations
    st.subheader("Original Raster")
    fig1, ax1 = plt.subplots()
    ax1.imshow(input_data, cmap='gray')
    ax1.set_title("Input Image")
    st.pyplot(fig1)

    st.subheader("Predicted LULC Probability")
    fig2, ax2 = plt.subplots()
    ax2.imshow(prediction, cmap='viridis')
    ax2.set_title("Predicted (Continuous)")
    st.pyplot(fig2)

    st.subheader("Classified LULC Mask")
    fig3, ax3 = plt.subplots()
    ax3.imshow(prediction_classified, cmap='tab10')
    ax3.set_title("Predicted (Classified)")
    st.pyplot(fig3)

    # Option to download prediction
    save_btn = st.button("📥 Download Predicted TIFF")
    if save_btn:
        output_path = tmp_path.replace(".tif", "_predicted.tif")
        profile.update(dtype=rasterio.uint8, count=1, transform=transform, crs=crs)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction_classified.astype(rasterio.uint8), 1)

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Classified Prediction",
                data=f,
                file_name="lulc_predicted_classified.tif",
                mime="application/octet-stream"
            )
