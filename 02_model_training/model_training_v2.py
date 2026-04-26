"""
Improved training script for the Medicinal Leaf Classifier.

Fixes all critical bugs from the original Model_Creation.ipynb:
1. Proper train/val split (no data leakage)
2. Correct MobileNetV2 preprocessing ([-1, 1] normalization)
3. Dropout regularization
4. EarlyStopping + ReduceLROnPlateau callbacks
5. Runtime data augmentation layers
6. Two-phase training: frozen base → fine-tune top layers

Imports from: tensorflow, numpy, pathlib
Imported by: standalone script (run directly)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ─── Configuration ────────────────────────────────────────────────────────────

# VERIFY: update this path to point to your clean augmented dataset
DATASET_PATH: str = str(Path(__file__).parent / ".." / "01_data_augmentation" / "Medicinal_Leaves")
OUTPUT_DIR: Path = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
INITIAL_EPOCHS: int = 30
FINE_TUNE_EPOCHS: int = 20
FINE_TUNE_AT_LAYER: int = 100  # Unfreeze layers from this index onwards


# ─── Data Loading ─────────────────────────────────────────────────────────────

print(f"Loading dataset from: {DATASET_PATH}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, Batch size: {BATCH_SIZE}")

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode="int",
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode="int",
)

class_names = train_data.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

assert num_classes == 13, f"Expected 13 classes, got {num_classes}"


# ─── Preprocessing ────────────────────────────────────────────────────────────
# MobileNetV2 expects inputs in [-1, 1], not [0, 1]

def apply_preprocessing(image: tf.Tensor, label: tf.Tensor) -> tuple:
    """Apply MobileNetV2-specific preprocessing (scales to [-1, 1])."""
    image = preprocess_input(image)
    return image, label


train_data = train_data.map(apply_preprocessing).cache().prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(apply_preprocessing).cache().prefetch(tf.data.AUTOTUNE)


# ─── Data Augmentation (in-model, runtime) ────────────────────────────────────

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
], name="data_augmentation")


# ─── Model Architecture ──────────────────────────────────────────────────────

base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze for Phase 1

print(f"Base model: MobileNetV2 ({len(base_model.layers)} layers)")
print(f"Trainable parameters (Phase 1, frozen base): ", end="")

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

trainable_params = sum(p.numpy().size for p in model.trainable_weights)
print(f"{trainable_params:,}")
model.summary()


# ─── Callbacks ────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "mobilenetv2_leaf_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]


# ─── Phase 1: Train head only (frozen base) ──────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1: Training classification head (frozen base)")
print("=" * 60)

history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
)


# ─── Phase 2: Fine-tune top layers of base model ─────────────────────────────

print("\n" + "=" * 60)
print(f"PHASE 2: Fine-tuning (unfreezing layers {FINE_TUNE_AT_LAYER}+)")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
    layer.trainable = False

fine_tune_trainable = sum(p.numpy().size for p in model.trainable_weights)
print(f"Trainable parameters (Phase 2): {fine_tune_trainable:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Much lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

history_phase2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=total_epochs,
    initial_epoch=len(history_phase1.epoch),
    callbacks=callbacks,
)


# ─── Save final model ────────────────────────────────────────────────────────

final_path = OUTPUT_DIR / "mobilenetv2_leaf_classifier.h5"
model.save(str(final_path))
print(f"\nFinal model saved to: {final_path}")


# ─── Print training summary ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

# Combine histories
all_train_acc = history_phase1.history["accuracy"] + history_phase2.history.get("accuracy", [])
all_val_acc = history_phase1.history["val_accuracy"] + history_phase2.history.get("val_accuracy", [])
all_train_loss = history_phase1.history["loss"] + history_phase2.history.get("loss", [])
all_val_loss = history_phase1.history["val_loss"] + history_phase2.history.get("val_loss", [])

print("\nEpoch | Train Acc | Val Acc | Train Loss | Val Loss")
print("-" * 55)
for i in range(len(all_train_acc)):
    marker = " ← fine-tune" if i >= len(history_phase1.epoch) else ""
    print(f"{i+1:5d} | {all_train_acc[i]:.4f}    | {all_val_acc[i]:.4f}  | {all_train_loss[i]:.4f}     | {all_val_loss[i]:.4f}{marker}")

# Overfitting check
final_train = all_train_acc[-1]
final_val = all_val_acc[-1]
gap = final_train - final_val

print(f"\nFinal Train Accuracy: {final_train:.4f}")
print(f"Final Val Accuracy:   {final_val:.4f}")
print(f"Gap (train - val):    {gap:.4f}")

if gap > 0.15:
    print("⚠️  WARNING: Large train-val gap (>15%). Model is likely OVERFITTING.")
    print("   Suggestions: increase Dropout, add L2 regularization, get more real data.")
elif gap > 0.05:
    print("ℹ️  NOTICE: Moderate train-val gap (5-15%). Some overfitting, but acceptable.")
else:
    print("✅ Model generalizes well (gap < 5%).")


# ── Module self-test
# Run: python model_training_v2.py
# Expected output: Training runs for up to 50 epochs, prints summary table
