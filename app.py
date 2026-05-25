import streamlit as st
from PIL import Image
import numpy as np

st.title("👗 Outfit AI Classifier")

class_names = ['STREET', 'casual']

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    result = "casual"

    st.success(f"Predicted Outfit: {result}")
