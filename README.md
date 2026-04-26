# Medicinal Leaf Classifier

A machine learning pipeline for classifying 13 types of medicinal plant leaves using MobileNetV2 transfer learning.

## Project Structure

```
├── 01_data_augmentation/    Image augmentation to balance dataset (500 images/class)
├── 02_model_training/       MobileNetV2 transfer learning notebook
├── 03_web_app/              Flask web application for inference
│   ├── app.py               Main Flask application
│   ├── config.py            Centralized constants
│   ├── database.py          SQLite user authentication
│   ├── prediction.py        Model inference (raw + segmented)
│   ├── models/              Trained .h5 model files
│   ├── static/              CSS, JS, images
│   └── templates/           Jinja2 HTML templates
└── docs/                    Architecture diagrams & documentation
```

## Supported Leaf Classes

Aloevera, Amla, Bhrami, Bringaraja, Coriender, Curry, Ekka, Hibiscus, Lemon, Mint, Neem, Papaya, Tulsi

## Quick Start

```bash
# 1. Set up Python 3.11 environment
pyenv local 3.11.9
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r 03_web_app/requirements.txt

# 3. Run the web app
cd 03_web_app && python app.py
```

## Pipeline

1. **Data Augmentation** (`01_data_augmentation/`): Balances dataset to 500 images per class using `ImageDataGenerator`.
2. **Model Training** (`02_model_training/`): Trains MobileNetV2 on augmented dataset with `image_dataset_from_directory`.
3. **Web App** (`03_web_app/`): Flask app with user auth, image upload, dual-model prediction (raw + HSV-segmented).

## Tech Stack

- **Python 3.11** | **TensorFlow 2.21** | **Flask** | **SQLite** | **OpenCV**
