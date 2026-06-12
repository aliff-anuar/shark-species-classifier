# =============================================================
# STEP 4: MODEL TRAINING
# Role: Data Scientist
# Description: Train ResNet50, DenseNet121, MobileNetV3
#              with transfer learning, 50 epochs each
#              Metrics: Accuracy + mAP
# =============================================================

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import json

# ── Configuration ────────────────────────────────────────────
SPLIT_DIR   = "shark_dataset/split"
RESULTS_DIR = "shark_results"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 50
NUM_CLASSES = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = [
    "blacktip_reef_shark",
    "bull_shark",
    "great_white_shark",
    "hammerhead_shark",
    "mako_shark",
    "nurse_shark",
    "tiger_shark",
    "whale_shark"
]

# ── Data Generators ──────────────────────────────────────────
# Training: with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2]
)

# Validation & Test: only rescale
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ── mAP Helper ───────────────────────────────────────────────
def compute_map(model, generator):
    """Compute mean Average Precision (mAP) on a generator."""
    generator.reset()
    y_true = []
    y_pred = []

    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        preds = model.predict(X_batch, verbose=0)
        y_true.append(y_batch)
        y_pred.append(preds)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # Compute AP for each class then average
    APs = []
    for c in range(NUM_CLASSES):
        ap = average_precision_score(y_true[:, c], y_pred[:, c])
        APs.append(ap)

    return np.mean(APs), APs

# ── Build Model Helper ────────────────────────────────────────
def build_model(base_model_fn, model_name):
    """Build transfer learning model with frozen base + custom head."""
    print(f"\n Building {model_name}...")

    base = base_model_fn(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Phase 1: Freeze base — train only the new head
    base.trainable = False

    inputs  = keras.Input(shape=(224, 224, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=model_name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base

# ── Train One Model ───────────────────────────────────────────
def train_model(base_model_fn, model_name):
    model, base = build_model(base_model_fn, model_name)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(RESULTS_DIR, f"{model_name}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, f"{model_name}_log.csv")
        )
    ]

    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    start_time = time.time()

    # ── Phase 1: Train head only (10 epochs warm-up) ──────────
    print("\n Phase 1: Warming up head (10 epochs)...")
    history_phase1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: Fine-tune — unfreeze top layers of base ──────
    print("\n Phase 2: Fine-tuning (remaining epochs)...")
    base.trainable = True

    # Unfreeze only the last 30 layers of the base model
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    remaining_epochs = EPOCHS - 10
    history_phase2 = model.fit(
        train_gen,
        epochs=remaining_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    elapsed = time.time() - start_time
    training_time = round(elapsed / 60, 2)  # in minutes

    # ── Combine histories ──────────────────────────────────────
    combined_history = {
        "accuracy"    : history_phase1.history["accuracy"]     + history_phase2.history["accuracy"],
        "val_accuracy": history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"],
        "loss"        : history_phase1.history["loss"]         + history_phase2.history["loss"],
        "val_loss"    : history_phase1.history["val_loss"]     + history_phase2.history["val_loss"],
    }

    # ── Evaluate on test set ───────────────────────────────────
    print(f"\n Evaluating {model_name} on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    test_map, per_class_ap   = compute_map(model, test_gen)

    print(f"\n {model_name} Results:")
    print(f"   Test Accuracy  : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Test mAP       : {test_map:.4f}")
    print(f"   Training Time  : {training_time} minutes")

    # ── Save results ───────────────────────────────────────────
    results = {
        "model"         : model_name,
        "test_accuracy" : round(float(test_accuracy), 4),
        "test_map"      : round(float(test_map), 4),
        "training_time" : training_time,
        "per_class_ap"  : {CLASS_NAMES[i]: round(float(per_class_ap[i]), 4) for i in range(NUM_CLASSES)},
        "history"       : {k: [round(float(v), 4) for v in vals] for k, vals in combined_history.items()}
    }

    with open(os.path.join(RESULTS_DIR, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save final model
    model.save(os.path.join(RESULTS_DIR, f"{model_name}_final.h5"))
    print(f" Model saved to: {RESULTS_DIR}/{model_name}_final.h5")

    return results

# ── Run All 3 Models ─────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    models_to_train = [
        (ResNet50,       "ResNet50"),
        (DenseNet121,    "DenseNet121"),
        (MobileNetV3Large, "MobileNetV3"),
    ]

    for base_fn, name in models_to_train:
        result = train_model(base_fn, name)
        all_results[name] = result

    # Save combined summary
    with open(os.path.join(RESULTS_DIR, "all_results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Final Comparison Table ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Accuracy':>10} {'mAP':>10} {'Time (min)':>12}")
    print(f"{'-'*50}")
    for name, r in all_results.items():
        print(f"{name:<15} {r['test_accuracy']:>10.4f} {r['test_map']:>10.4f} {r['training_time']:>12}")
    print(f"{'='*60}")
    print(f"\n All results saved to: {RESULTS_DIR}/")
