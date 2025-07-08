import streamlit as st
import numpy as np
import tensorflow as tf
import rasterio
import cv2
import os
import matplotlib.pyplot as plt
import gdown
from tempfile import NamedTemporaryFile
from rasterio.plot import reshape_as_image

# Download model from Google Drive if not exists
MODEL_FILE = "unet_lulc_model.h5"
MODEL_ID = "1vkeZmAIzop8K5MdsIK8o70xpEHN7YVH6"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_FILE, quiet=False)
    return tf.keras.models.load_model(MODEL_FILE)

model = load_model()

st.set_page_config(page_title="LULC Predictor", layout="wide")
st.title("🛰️ Land Use Land Cover (LULC) Prediction")

uploaded_files = st.file_uploader("Upload one or more .tif raster files", type=["tif", "tiff"], accept_multiple_files=True)

def classify_prediction(prediction_data, thresholds=[0.2, 0.4, 0.6, 0.8], class_values=[1, 2, 3, 4, 5]):
    classified = np.zeros_like(prediction_data, dtype=np.int32)
    for i, thresh in enumerate(thresholds):
        if i == 0:
            classified[prediction_data <= thresh] = class_values[i]
        else:
            classified[(prediction_data > thresholds[i-1]) & (prediction_data <= thresh)] = class_values[i]
    classified[prediction_data > thresholds[-1]] = class_values[-1]
    return classified

if uploaded_files:
    input_images, pred_images, class_images = [], [], []

    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with rasterio.open(tmp_path) as src:
            input_data = src.read(1)
            original_shape = input_data.shape
            profile = src.profile

        input_resized = cv2.resize(input_data, (256, 256), interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(input_resized, axis=(0, -1))

        prediction = model.predict(input_tensor)[0, ..., 0]
        prediction_resized = cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_LINEAR)

        classified = classify_prediction(prediction_resized)

        input_images.append(input_data)
        pred_images.append(prediction_resized)
        class_images.append(classified)

        # Save classified output
        classified_path = tmp_path.replace(".tif", "_classified.tif")
        profile.update(dtype=rasterio.int32, count=1)
        with rasterio.open(classified_path, 'w', **profile) as dst:
            dst.write(classified, 1)

        with open(classified_path, "rb") as f:
            st.download_button(f"📥 Download {uploaded_file.name}_classified.tif", f, file_name=os.path.basename(classified_path))

    # Visualization
    st.subheader("🔍 Comparison")
    for i in range(len(input_images)):
        st.markdown(f"### File {i+1}")
        col1, col2, col3 = st.columns(3)
        col1.image(input_images[i], caption="Input", use_column_width=True, clamp=True)
        col2.image(pred_images[i], caption="Prediction (probability)", use_column_width=True, clamp=True)
        col3.image(class_images[i], caption="Classified Output", use_column_width=True, clamp=True)