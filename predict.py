import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("outfit_model.keras")

# Class names (must match folders)
class_names = ['STREET', 'casual']

# Load image (CHANGE THIS NAME)
img = Image.open("outfit.jpg").resize((224, 224))

# Convert image
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
result = class_names[np.argmax(prediction)]

print("Predicted Outfit:", result)