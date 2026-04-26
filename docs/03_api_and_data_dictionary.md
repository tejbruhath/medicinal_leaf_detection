# API & Data Dictionary

This document covers the application-level data models, SQLite schema, and internal API routing mechanism used by the Flask application.

## Flask HTTP Routing

The Flask application (`03_web_app/app.py`) serves standard HTML and handles multipart form data for image uploads.

| Method | Route | Description | Requires Auth | Data Submitted |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Renders index/landing page. | No | None |
| `GET` | `/loginpage` | Renders login page. | No | None |
| `POST` | `/login` | Authenticates User, sets session cookies. | No | `username`, `password` |
| `GET` | `/register` | Renders user registration page. | No | None |
| `POST` | `/add_user` | Creates new user in the DB. | No | `name`, `email`, `password` |
| `GET` | `/prediction` | Renders the leaf upload/prediction UI. | **Yes** | None |
| `POST` | `/upload` | Receives image, triggers ML inf., saves to `/static/uploads`. | No (Should be Yes) | `imageFile` (Multipart) |
| `POST` | `/predict` | Evaluates the latest uploaded image manually. | No | None |
| `GET` | `/logout` | Clears Flask session and redirects home. | No | None |

## Data Dictionary: Database Schema

The web app uses SQLite3 (`mydb.db`).

### Table: `users`
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | PRIMARY KEY, AUTOINCREMENT | Internal system ID. |
| `username` | `TEXT` | NOT NULL, UNIQUE | Desired username for login. |
| `email` | `TEXT` | NOT NULL, UNIQUE | User email address. |
| `password` | `TEXT` | NOT NULL | **Security Note:** Currently stored as plaintext. Should transition to `bcrypt`. |
| `datetime` | `TEXT` | NOT NULL | ISO-8601 formatted creation timestamp. |

## Machine Learning Constants (`config.py`)

The ML pipeline operates on exactly 13 predefined classes. Modifying the dataset requires appending to `LABEL_NAMES` and updating `NUM_CLASSES`.

```python
LABEL_NAMES = [
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

# Dual Model Verification
LOW_CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_RAW_THRESHOLD = 0.40
```
