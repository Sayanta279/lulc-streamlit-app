import os
import cv2
import gdown
import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from rasterio.plot import reshape_as_image

# Model folder ID on Google Drive (update if changed)
FOLDER_ID = "1z2Nm7bkE6zxq1_ro0B9JGETkTVk4sjwd"
MODEL_DIR = "unet_lulc_model_tf"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        gdown.download_folder(id=FOLDER_ID, quiet=False, use_cookies=False)
    return tf.keras.models.load_model(MODEL_DIR)

model = load_model()

st.set_page_config(page_title="LULC Predictor", layout="wide")
st.title("🛰️ Land Use Land Cover (LULC) Prediction")

uploaded_file = st.file_uploader("Upload a single-band GeoTIFF image", type=["tif", "tiff"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    with rasterio.open(input_path) as src:
        img = src.read(1)
        original_shape = img.shape
        input_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        input_resized = input_resized.astype(np.float32) / np.max(input_resized)
        input_tensor = np.expand_dims(input_resized, axis=(0, -1))

    prediction = model.predict(input_tensor)[0, :, :, 0]
    prediction_resized = cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_NEAREST)

    st.subheader("🧾 Prediction Output")
    st.image(prediction_resized, caption="Predicted LULC", use_column_width=True, clamp=True)

    st.subheader("📥 Download")
    with NamedTemporaryFile(delete=False, suffix=".tif") as out_file:
        with rasterio.open(input_path) as src:
            profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_file.name, 'w', **profile) as dst:
            dst.write(prediction_resized.astype(np.float32), 1)
        with open(out_file.name, "rb") as f:
            st.download_button("Download predicted TIFF", f.read(), file_name="lulc_prediction.tif", mime="image/tiff")
