# System Architecture

This document describes the high-level system architecture of the Medicinal Leaf Classifier following the massive codebase refactoring in April 2026.

## Context Architecture

The system enables users to upload photos of medicinal leaves through a web interface, which are then processed by a dual-model machine learning pipeline (MobileNetV2 classification and HSV segmentation) to provide botanical identification and medical treatments.

```mermaid
flowchart TD
    User([User])
    System[Medicinal Leaf Classifier Web App]
    Database[(SQLite Database)]
    Model[(MobileNetV2 Model)]

    User -->|Uploads leaf images| System
    System -->|Authenticates users| Database
    System -->|Classifies images| Model
    Model -->|Returns predictions| System
    System -->|Displays plant info| User
```

## Container Architecture

The project is structured into three primary logical zones, representing the machine learning lifecycle: Data Augmentation, Model Training, and Web Application deployment.

```mermaid
flowchart TB
    subgraph M1 [01_data_augmentation]
        nb1[augmentation.ipynb]
        data_raw[(Medicinal_Leaves/)]
        data_aug[(result/)]
        nb1 -->|reads| data_raw
        nb1 -->|generates augmented images| data_aug
    end

    subgraph M2 [02_model_training]
        nb2[model_training_v2.py]
        nb2 -->|loads| data_aug
        nb2 -->|trains and saves| model_out[(mobilenetv2_leaf_classifier.h5)]
    end

    subgraph M3 [03_web_app]
        flask[Flask App]
        db[(mydb.db)]
        flask -->|loads for inference| model_out
        flask -->|auth| db
    end
```

## Component Architecture: Web App (`03_web_app/`)

The monolithic `web_app.py` was decomposed into modular components, isolating configuration, database logic, inference logic, and routing.

```mermaid
flowchart TB
    Client[Web Browser]
    
    subgraph Flask App
        app(app.py - Routing & Controllers)
        config(config.py - Globals & Paths)
        db_mod(database.py - SQLite logic)
        pred_mod(prediction.py - ML logic)
    end
    
    Client <-->|HTTP GET/POST| app
    app -->|Imports keys| config
    app -->|Validates/Creates user| db_mod
    app -->|predict_leaf(image)| pred_mod
    
    pred_mod -->|loads bounds| config
    pred_mod -->|loads paths| config
```

### Module Responsibilities
*   **`config.py`**: Centralizes all configuration, constants, and Paths (e.g., `LABEL_NAMES`, `HSV_LOWER_BOUND`, `MODEL_DIR`).
*   **`database.py`**: Handles SQLite logic, user creation, and basic authentication queries.
*   **`prediction.py`**: Prepares the image `preprocess_image()`, runs HSV segmentation `segment_leaf()`, and computes confidence scores.
*   **`app.py`**: Sets up Flask routes, template rendering, and file upload form handling.
