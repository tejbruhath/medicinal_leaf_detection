"""
Configuration constants for the Medicinal Leaf Classifier web application.

This module centralizes all application constants, model paths, and label
definitions. All other modules import from here — never hardcode these values.
"""

from pathlib import Path

# Application
SECRET_KEY: str = "supersecretkey"
DEBUG: bool = False

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_DIR: Path = BASE_DIR / "models"
UPLOAD_DIR: Path = BASE_DIR / "static" / "uploads"
DATABASE_PATH: Path = BASE_DIR / "mydb.db"

# Model files
RAW_MODEL_PATH: Path = MODEL_DIR / "mobilenetv2_leaf_classifier.h5"
SEGMENTED_MODEL_PATH: Path = MODEL_DIR / "model_inceptionv2_withaug500andsegmented.h5"

# Image preprocessing
IMAGE_SIZE: int = 224

# Classification labels (alphabetical order matching training data)
LABEL_NAMES: list[str] = [
    "Aloevera",
    "Amla",
    "Bhrami",
    "Bringaraja",
    "Coriender",
    "Curry",
    "Ekka",
    "Hibiscus",
    "Lemon",
    "Mint",
    "Neem",
    "Papaya",
    "Tulsi",
]
NUM_CLASSES: int = len(LABEL_NAMES)

# HSV segmentation bounds (green leaf extraction)
HSV_LOWER_BOUND: tuple[int, int, int] = (30, 40, 40)
HSV_UPPER_BOUND: tuple[int, int, int] = (90, 255, 255)

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD: float = 0.50
LOW_CONFIDENCE_RAW_THRESHOLD: float = 0.40
LOW_CONFIDENCE_MESSAGE: str = (
    "Confidence score is very low so you can provide a good quality leaf image only"
)


# ── Module self-test
# Run: python -m config
# Expected output: prints all config values
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"RAW_MODEL_PATH: {RAW_MODEL_PATH}")
    print(f"SEGMENTED_MODEL_PATH: {SEGMENTED_MODEL_PATH}")
    print(f"LABEL_NAMES ({NUM_CLASSES} classes): {LABEL_NAMES}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"HSV bounds: {HSV_LOWER_BOUND} -> {HSV_UPPER_BOUND}")
