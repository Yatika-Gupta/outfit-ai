import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("outfit_model.h5")

class_names = ['STREET', 'casual']

st.title("👗 Outfit AI Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    st.success(f"Predicted Outfit: {result}")