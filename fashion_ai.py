import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Labels
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

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(train_images, train_labels, epochs=5)

# Prediction model
probability_model = keras.Sequential([
    model,
    keras.layers.Softmax()
])

# Predict first image
predictions = probability_model.predict(test_images)

# Print prediction
predicted_label = np.argmax(predictions[0])

print("AI Prediction:", class_names[predicted_label])
print("Actual Label:", class_names[test_labels[0]])

# Show image
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.title(class_names[predicted_label])
plt.colorbar()
plt.show()