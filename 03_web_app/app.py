"""
Flask web application for the Medicinal Leaf Classifier.

This module defines all HTTP routes for the web interface including
authentication (login, register, logout), image upload, and leaf prediction.

Imports from: config, database, prediction
"""

import shutil
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import DEBUG, SECRET_KEY, UPLOAD_DIR
from database import create_user, get_user_id, validate_user
from prediction import predict_leaf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the single-page application frontend."""
    return render_template("app_ui.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    """JSON endpoint for user login."""
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"status": "error", "message": "Missing credentials"}), 400

    username = data["username"]
    password = data["password"]

    is_valid, user_name_db = validate_user(username, password)
    if is_valid:
        session["user"] = user_name_db
        return jsonify({"status": "success", "message": "Logged in successfully", "user": user_name_db})
    else:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401


@app.route("/api/register", methods=["POST"])
def api_register():
    """JSON endpoint for user registration."""
    data = request.get_json()
    if not data or "name" not in data or "email" not in data or "password" not in data:
        return jsonify({"status": "error", "message": "Missing fields"}), 400
    
    # Normally check if exists, but database.py create_user returns False if it fails (e.g. duplicate)
    success = create_user(data["name"], data["email"], data["password"])
    if success:
        return jsonify({"status": "success", "message": "User registered successfully"})
    else:
        return jsonify({"status": "error", "message": "User already exists or DB error"}), 409


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """JSON endpoint for logout."""
    session.pop("user", None)
    return jsonify({"status": "success", "message": "Logged out"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint to upload image and get predictions."""
    if "user" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if "imageFile" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    f = request.files["imageFile"]
    if f.filename == "":
        return jsonify({"status": "error", "message": "Empty file name"}), 400

    # Save the file cleanly
    filename = secure_filename(f.filename)
    filepath = UPLOAD_DIR / filename
    f.save(filepath)

    try:
        # Get prediction — returns (class_name, confidence) or (LOW_CONFIDENCE_MESSAGE, conf)
        pred_class, pred_confidence = predict_leaf(str(filepath))

        # Determine if this is a low-confidence result
        from config import LOW_CONFIDENCE_MESSAGE
        is_low_confidence = pred_class == LOW_CONFIDENCE_MESSAGE

        # Plant information lookup
        plant_info_db = {
            "Tulsi": {
                "botanical": "Ocimum tenuiflorum",
                "uses": "Used for treating colds, flu, and asthma. Sacred in Ayurveda.",
                "side_effects": "May cause nausea or diarrhea in high dosages. Avoid before surgery.",
                "remedies": "Chew raw leaves or brew into a tea with ginger and honey.",
            },
            "Mint": {
                "botanical": "Mentha",
                "uses": "Aids digestion, relieves headaches.",
                "side_effects": "Heartburn, allergic reactions (rare).",
                "remedies": "Crush leaves and inhale, or brew tea.",
            },
            "Neem": {
                "botanical": "Azadirachta indica",
                "uses": "Antibacterial, antifungal. Used for skin conditions and dental care.",
                "side_effects": "Toxic in large doses. Not for pregnant women.",
                "remedies": "Boil leaves and use water for bathing skin.",
            },
            "Coriender": {
                "botanical": "Coriandrum sativum",
                "uses": "Digestive aid, anti-inflammatory, lowers blood sugar.",
                "side_effects": "May cause allergic reactions in sensitive individuals.",
                "remedies": "Brew seeds into tea for digestion. Use fresh leaves in food.",
            },
            "Aloe Vera": {
                "botanical": "Aloe barbadensis miller",
                "uses": "Treats burns, skin conditions, and digestive issues.",
                "side_effects": "Oral consumption in high doses can cause diarrhea.",
                "remedies": "Apply gel directly to skin for burns or irritation.",
            },
            "default": {
                "botanical": "Botanical Name Unknown",
                "uses": "Information not available for this specific plant.",
                "side_effects": "Unknown.",
                "remedies": "Consult an ayurvedic practitioner.",
            }
        }

        info = plant_info_db.get(pred_class, plant_info_db["default"])

        return jsonify({
            "status": "success",
            "prediction": {
                "plant_name": pred_class,
                "confidence": float(pred_confidence),
                "botanical_name": info["botanical"],
                "uses": info["uses"],
                "side_effects": info["side_effects"],
                "remedies": info["remedies"],
                "warning": pred_class if is_low_confidence else None
            }
        })

    except Exception as e:

        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Prediction failed: {e}"}), 500


# ── Module self-test
# Run: python app.py
# Expected output: Flask development server starts on http://127.0.0.1:5000
if __name__ == "__main__":
    app.run(debug=DEBUG)
