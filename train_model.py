import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Paths
DATASET_PATH = "/Users/ramya/Documents/RAMYA/Projects/waste_classifier_app/archive/Garbage classification"
MODEL_PATH = "model/waste_classifier.h5"
CLASS_NAMES_PATH = "model/class_names.txt"

# Parameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 30  # Train longer for better performance

# Create output folder if it doesn’t exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Data generators
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Print class index mapping
print("Class indices:", train_data.class_indices)

# Save class names in the correct order
class_names = list(train_data.class_indices.keys())
with open(CLASS_NAMES_PATH, 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print(f"✅ Class names saved to {CLASS_NAMES_PATH}")

# Define the CNN model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model((150, 150, 3), train_data.num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Optional: test prediction before saving
x_batch, y_batch = next(val_data)
preds = model.predict(x_batch)
print("Sample prediction (raw probabilities):", preds[0])
print("Predicted class:", class_names[np.argmax(preds[0])])
print("Classes found:", train_data.class_indices)
print("Number of classes:", train_data.num_classes)

# Save the trained model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
