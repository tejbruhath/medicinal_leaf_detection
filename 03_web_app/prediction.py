"""
Prediction module for the Medicinal Leaf Classifier web application.

This module handles image preprocessing, model loading, and inference
using MobileNetV2 (raw image). The segmented model is intentionally
bypassed due to an unfixable dual-input architecture incompatibility
with the current Keras loader.

Imports from: config
Imported by: app
"""

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import keras
import keras.layers as keras_layers
from keras.models import load_model

from config import (
    IMAGE_SIZE,
    LABEL_NAMES,
    LOW_CONFIDENCE_MESSAGE,
    LOW_CONFIDENCE_RAW_THRESHOLD,
    RAW_MODEL_PATH,
)

# ── Custom layer wrappers to handle Keras 3 → tf-keras serialization quirks ──


class _SafeDepthwiseConv2D(keras_layers.DepthwiseConv2D):
    """DepthwiseConv2D that ignores the 'groups' kwarg added by Keras 3."""

    @classmethod
    def from_config(cls, config: dict) -> "_SafeDepthwiseConv2D":
        """Build from config, stripping the unknown 'groups' argument.

        Args:
            config: Layer config dict from the saved model.

        Returns:
            A new _SafeDepthwiseConv2D instance.
        """
        config.pop("groups", None)
        return super().from_config(config)


class _SafeRandomFlip(keras_layers.RandomFlip):
    """RandomFlip that ignores the 'data_format' kwarg added by Keras 3."""

    @classmethod
    def from_config(cls, config: dict) -> "_SafeRandomFlip":
        """Build from config, stripping the unknown 'data_format' argument.

        Args:
            config: Layer config dict from the saved model.

        Returns:
            A new _SafeRandomFlip instance.
        """
        config.pop("data_format", None)
        return super().from_config(config)


class _SafeInputLayer(keras_layers.InputLayer):
    """InputLayer that ignores the 'optional' kwarg added by Keras 3."""

    @classmethod
    def from_config(cls, config: dict) -> "_SafeInputLayer":
        """Build from config, stripping the unknown 'optional' argument.

        Args:
            config: Layer config dict from the saved model.

        Returns:
            A new _SafeInputLayer instance.
        """
        config.pop("optional", None)
        return super().from_config(config)


_CUSTOM_OBJECTS: dict[str, type] = {
    "DepthwiseConv2D": _SafeDepthwiseConv2D,
    "RandomFlip": _SafeRandomFlip,
    "InputLayer": _SafeInputLayer,
}


def preprocess_image(image_path: str, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Load, resize, and normalize an image for model inference.

    Args:
        image_path: Path to the image file.
        target_size: Target height and width in pixels.

    Returns:
        A 4D numpy array of shape (1, target_size, target_size, 3).

    Raises:
        FileNotFoundError: If image_path does not exist.
    """
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(target_size, target_size)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)
    
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image_array = preprocess_input(image_array)

    assert image_array.shape == (1, target_size, target_size, 3), (
        f"Expected shape (1, {target_size}, {target_size}, 3), got {image_array.shape}"
    )

    return image_array


def predict_leaf(image_path: str) -> tuple[str, float]:
    """
    Run prediction on a leaf image using the raw MobileNetV2 model.

    Args:
        image_path: Path to the uploaded leaf image.

    Returns:
        A tuple of (class_name, confidence) where confidence is 0.0–1.0.
        Returns (LOW_CONFIDENCE_MESSAGE, confidence) when confidence is too low.

    Raises:
        Exception: Propagated from model loading or image processing failures.
    """
    try:
        raw_model = load_model(str(RAW_MODEL_PATH), custom_objects=_CUSTOM_OBJECTS)
        features = preprocess_image(image_path)
        predictions = raw_model.predict(features)
        class_index = int(np.argmax(predictions))
        confidence = float(predictions[0][class_index])
        class_name = LABEL_NAMES[class_index]
        print(f"Raw prediction: {class_name} ({confidence:.2%})")

        if confidence < LOW_CONFIDENCE_RAW_THRESHOLD:
            print(f"Low confidence: {confidence:.2f}")
            return LOW_CONFIDENCE_MESSAGE, confidence

        return class_name, confidence

    except Exception as e:
        print(f"Prediction error: {e}")
        raise


# ── Module self-test
# Run: python -m prediction
# Expected output: prints model paths and checks they exist
if __name__ == "__main__":
    print(f"Raw model path: {RAW_MODEL_PATH} (exists: {RAW_MODEL_PATH.exists()})")
    print(f"Labels: {LABEL_NAMES}")
