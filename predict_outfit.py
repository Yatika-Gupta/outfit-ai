print("AI script started")
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fashion labels
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=5)

# Load your custom image
img = Image.open("outfit.jpg").convert('L')

# Resize image
img = img.resize((28, 28))

# Convert to array
img_array = np.array(img)

# Normalize
img_array = img_array / 255.0

# Invert colors
img_array = 1 - img_array

# Reshape
img_array = img_array.reshape(1, 28, 28)

# Predict
prediction = model.predict(img_array)

predicted_label = np.argmax(prediction)

print("Predicted Outfit:", class_names[predicted_label])

# Show image
plt.imshow(img_array[0], cmap=plt.cm.binary)
plt.title(class_names[predicted_label])
plt.show()
