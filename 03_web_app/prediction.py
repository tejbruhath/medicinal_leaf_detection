"""
Prediction module for the Medicinal Leaf Classifier web application.

This module handles image preprocessing, model loading, HSV-based leaf
segmentation, and inference using MobileNetV2 models.

Imports from: config
Imported by: app
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import (
    HSV_LOWER_BOUND,
    HSV_UPPER_BOUND,
    IMAGE_SIZE,
    LABEL_NAMES,
    LOW_CONFIDENCE_MESSAGE,
    LOW_CONFIDENCE_RAW_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    RAW_MODEL_PATH,
    SEGMENTED_MODEL_PATH,
)


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
    image_array = image_array / 255.0
    image_array = tf.expand_dims(image_array, axis=0)

    assert image_array.shape == (1, target_size, target_size, 3), (
        f"Expected shape (1, {target_size}, {target_size}, 3), got {image_array.shape}"
    )

    return image_array


def segment_leaf(image_path: str, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Perform HSV-based green leaf segmentation on an image.

    Args:
        image_path: Path to the image file.
        target_size: Target height and width in pixels.

    Returns:
        A 4D numpy array of the segmented leaf, shape (1, target_size, target_size, 3).

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be read by OpenCV.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (target_size, target_size))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array(HSV_LOWER_BOUND)
    upper_bound = np.array(HSV_UPPER_BOUND)

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    segmented = segmented.astype(np.float32) / 255.0
    segmented = segmented[np.newaxis, ...]

    assert segmented.shape == (1, target_size, target_size, 3), (
        f"Expected shape (1, {target_size}, {target_size}, 3), got {segmented.shape}"
    )

    return segmented


def predict_leaf(image_path: str) -> str:
    """
    Run dual-model prediction on a leaf image.

    Loads both the raw and segmented models, runs inference, and
    returns the predicted leaf class name or a low-confidence warning.

    Args:
        image_path: Path to the uploaded leaf image.

    Returns:
        The predicted class name string, or a low-confidence warning message.

    Raises:
        Exception: Propagated from model loading or image processing failures.
    """
    try:
        # --- Raw image prediction ---
        raw_model = load_model(str(RAW_MODEL_PATH))
        raw_features = preprocess_image(image_path)
        raw_predictions = raw_model.predict(raw_features)
        raw_class_index = int(np.argmax(raw_predictions))
        raw_confidence = float(raw_predictions[0][raw_class_index])
        raw_class_name = LABEL_NAMES[raw_class_index]
        print(f"Raw prediction: {raw_class_name} ({raw_confidence:.2%})")

        # --- Segmented image prediction ---
        seg_model = load_model(str(SEGMENTED_MODEL_PATH))
        seg_features = segment_leaf(image_path)
        seg_predictions = seg_model.predict(seg_features)
        seg_class_index = int(np.argmax(seg_predictions))
        seg_confidence = float(seg_predictions[0][seg_class_index])
        seg_class_name = LABEL_NAMES[seg_class_index]
        print(f"Segmented prediction: {seg_class_name} ({seg_confidence:.2%})")

        # --- Combine results ---
        result = seg_class_name

        if seg_confidence < LOW_CONFIDENCE_THRESHOLD or raw_confidence < LOW_CONFIDENCE_RAW_THRESHOLD:
            print(f"Low confidence: seg={seg_confidence:.2f}, raw={raw_confidence:.2f}")
            result = LOW_CONFIDENCE_MESSAGE

        return result

    except Exception as e:
        print(f"Prediction error: {e}")
        raise


# ── Module self-test
# Run: python -m prediction
# Expected output: prints model paths and checks they exist
if __name__ == "__main__":
    print(f"Raw model path: {RAW_MODEL_PATH} (exists: {RAW_MODEL_PATH.exists()})")
    print(f"Seg model path: {SEGMENTED_MODEL_PATH} (exists: {SEGMENTED_MODEL_PATH.exists()})")
    print(f"Labels: {LABEL_NAMES}")
