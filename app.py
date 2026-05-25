import streamlit as st
from PIL import Image
import numpy as np

# Load model
model = load_model("outfit_model.h5")

class_names = ['STREET', 'casual']

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # dummy prediction (temporary)
    result = "casual"

    st.success(f"Predicted Outfit: {result}")