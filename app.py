import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "FINAL_European_Flags_EfficientNetB2.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMG_SIZE = (260, 260)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load class names.
if CLASS_NAMES_PATH.exists():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Holland", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
        "Luxembourg", "Malta", "Slovakia", "Slovenia", "South Cyprus",
        "Spain", "Sweden"
    ]

model = None
model_error = None

if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = f"Model not found: {MODEL_PATH}"


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(path: Path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_flag(path: Path):
    if model is None:
        raise RuntimeError(model_error or "Model could not be loaded.")

    batch = prepare_image(path)
    probabilities = model.predict(batch, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:5]

    predictions = [
        {
            "class_name": CLASS_NAMES[int(i)],
            "confidence": float(probabilities[i] * 100),
        }
        for i in top_indices
    ]

    return predictions


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    top_predictions = []
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "Please choose a flag image first."
        elif not allowed_file(file.filename):
            error = "Unsupported image type. Use JPG, JPEG, PNG, or WEBP."
        else:
            filename = secure_filename(file.filename)
            saved_path = UPLOAD_DIR / filename

            # Avoid overwriting a previous upload.
            if saved_path.exists():
                stem = saved_path.stem
                suffix = saved_path.suffix
                counter = 1
                while saved_path.exists():
                    saved_path = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                file.save(saved_path)
                top_predictions = predict_flag(saved_path)
                prediction = top_predictions[0]
                image_url = "/" + saved_path.relative_to(BASE_DIR).as_posix()
            except (UnidentifiedImageError, OSError):
                error = "The uploaded file is not a valid readable image."
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        top_predictions=top_predictions,
        image_url=image_url,
        error=error,
        model_ready=model is not None,
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if model is not None else "model_error",
        "model": MODEL_PATH.name,
        "classes": len(CLASS_NAMES),
        "model_error": model_error,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("file")

    if not file or not file.filename:
        return jsonify({"error": "No image supplied."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image type."}), 400

    filename = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / filename

    try:
        file.save(saved_path)
        predictions = predict_flag(saved_path)
        return jsonify({
            "prediction": predictions[0],
            "top_5": predictions,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
