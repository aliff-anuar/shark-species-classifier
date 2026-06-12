# fix_mobilenet.py
# Run this once to resave MobileNetV3 in compatible format

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import layers

RESULTS_DIR = "shark_results"
NUM_CLASSES = 8

print("Rebuilding MobileNetV3 architecture...")

# Rebuild exact same architecture used in training
base = MobileNetV3Large(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

inputs  = keras.Input(shape=(224, 224, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="MobileNetV3")

# Load weights only (avoids architecture compatibility issue)
h5_path = os.path.join(RESULTS_DIR, "MobileNetV3_final.h5")
print(f"Loading weights from {h5_path}...")

try:
    model.load_weights(h5_path)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Trying best model instead...")
    model.load_weights(os.path.join(RESULTS_DIR, "MobileNetV3_best.h5"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Save in new .keras format
new_path = os.path.join(RESULTS_DIR, "MobileNetV3_final.keras")
model.save(new_path)
print(f"Saved to: {new_path}")
print("Done! Now run step5_visualize.py again.")