# Deployment & Operations Guide

This guide provides instructions for deploying the web application and operating the machine learning pipeline, including how to retrain the model.

## 1. Running the Web Application

The application requires Python 3.11.9 and local installation of dependencies. The heavy ML frameworks (TensorFlow) make Docker deployment sub-optimal for quick local tests, so venv usage is recommended.

```bash
# Ensure Python 3.11 is used
pyenv local 3.11.9

# Create and activate environment
cd 03_web_app
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```
*The app will start on `http://127.0.0.1:5000`.*

---

## 2. Retraining the ML Model

If you find that the model is misidentifying leaves, or if you want to add a 14th class, you need to execute the training pipeline.

### Prerequisites: Environment Setup

The ML training scripts reside in `01_data_augmentation/venv` as they require TensorFlow and OpenCV.

```bash
cd 01_data_augmentation
source venv/bin/activate
# If the venv breaks due to OS changes, recreate it:
# rm -rf venv && python3.11 -m venv venv && pip install -r requirements.txt
```

### Step 2.1: Clean the Dataset 
If the `Medicinal_Leaves/` directory was contaminated with old augmented images (images starting with `aug_`), clean them out before running anything.
```bash
find Medicinal_Leaves/ -name "aug_*" -type f -delete
```

### Step 2.2: Execute Training Script
We use the updated CLI python script rather than a Jupyter Notebook, allowing it to easily crash out on failure and providing clean console logs.

```bash
cd ../02_model_training
python model_training_v2.py
```

### Step 2.3: Re-deploy Model
When training completes successfully, it saves weights to `02_model_training/models/mobilenetv2_leaf_classifier.h5`.

You must copy this new model over to the Web Application and update the active configuration:

1. Copy the `.h5` file from `02_model_training/models/` into `03_web_app/models/`.
2. Open `03_web_app/config.py`.
3. Locate `RAW_MODEL_PATH` and make sure it points to the new `.h5` file name.
```python
RAW_MODEL_PATH: Path = MODEL_DIR / "mobilenetv2_leaf_classifier.h5"
```
4. Restart the Flask Server.

---

## 3. Operations Troubleshooting

### Missing PIL Error
> `Could not import PIL.Image. The use of load_img requires PIL.`
You forgot to install Pillow. Run `pip install Pillow` internally within the active venv.

### SQLite Database Locks
If the SQLite database locks up (you cannot login), verify that the python instance isn't still actively queried via threaded requests. Delete `mydb.db` to fully wipe the DB state and try again.

### Flask Port 5000 in Use
If the system complains `Address Already In Use`, find the python process occupying port 5000 and kill it:
```bash
lsof -i :5000
kill -9 <PID>
```
