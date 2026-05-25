import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset from folders
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=16
)

class_names = dataset.class_names
print("Classes:", class_names)

# Normalize images
dataset = dataset.map(lambda x, y: (x / 255.0, y))

# Build CNN model (real image AI)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(dataset, epochs=5)

# Save model
model.save("outfit_model.keras")

print("REAL MODEL TRAINED & SAVED 🚀")