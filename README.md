# 🇪🇺 EuroFlag AI — European Flag Recognition

A polished Flask web application that recognizes **24 European flags** using a trained **EfficientNetB2** image-classification model.

## Model performance

The final model was evaluated on an unseen test set:

| Metric          |           Result |
| --------------- | ---------------: |
| Test Accuracy   | **88.00%** |
| Macro Precision | **89.08%** |
| Macro Recall    | **86.42%** |
| Macro F1        | **86.18%** |
| Weighted F1     | **87.15%** |
| Test Loss       | **0.4292** |

Validation performance of the selected checkpoint:

- Validation Accuracy: **91.28%**
- Validation Loss: **0.3785**

> The test set is the final generalization benchmark. The application reports model predictions, not guaranteed identification.

## Features

- Modern responsive dark UI
- Drag-and-drop image upload
- JPG, JPEG, PNG and WEBP support
- EfficientNetB2 inference at 260×260
- Top-1 prediction with confidence
- Top-5 predictions
- REST-style `/api/predict` endpoint
- `/api/health` model health endpoint
- Upload size limit of 10 MB
- Mobile-friendly design

## Supported classes

Austria, Belgium, Bulgaria, Croatia, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Holland, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Slovakia, Slovenia, South Cyprus, Spain, Sweden.

## Project structure

```text
european_flags_flask_app/
├── app.py
├── class_names.json
├── requirements.txt
├── .gitignore
├── README.md
├── models/
│   └── FINAL_European_Flags_EfficientNetB2.keras
├── static/
│   ├── style.css
│   └── uploads/
└── templates/
    └── index.html
```

## 1. Python version

Recommended:

```text
Python 3.12.0
```

Check:

```bash
python --version
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow can take some time to install.

## 4. Add the trained model

Place your downloaded model here:

```text
models/FINAL_European_Flags_EfficientNetB2.keras
```

The included `class_names.json` contains the class order used by the application.

## 5. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API

### Health

```text
GET /api/health
```

### Prediction

```text
POST /api/predict
```

Form field:

```text
file=<image>
```

Example using curl:

```bash
curl -X POST -F "file=@flag.jpg" http://127.0.0.1:5000/api/predict
```

Example response:

```json
{
  "prediction": {
    "class_name": "France",
    "confidence": 97.42
  },
  "top_5": [
    {"class_name": "France", "confidence": 97.42},
    {"class_name": "Italy", "confidence": 1.21},
    {"class_name": "Ireland", "confidence": 0.64},
    {"class_name": "Belgium", "confidence": 0.31},
    {"class_name": "Romania", "confidence": 0.18}
  ]
}
```

The actual classes are limited to the 24 classes listed above.

## GitHub

Initialize:

```bash
git init
git add .
git commit -m "Initial European Flags Flask application"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

### Important model note

The `.gitignore` intentionally ignores `.keras` model files because trained models can be large. For a GitHub repository, either:

1. Keep the model locally and document where to obtain it.
2. Use Git LFS.
3. Use a model-hosting service.

If you want the model inside the Git repository and it is small enough for your hosting limits, remove this line from `.gitignore`:

```text
models/*.keras
```

## Troubleshooting

### Model not found

Make sure the exact path is:

```text
models/FINAL_European_Flags_EfficientNetB2.keras
```

### Shape mismatch

This model expects:

```text
260 × 260 × 3
```

The Flask app already resizes uploaded images to 260×260.

### TensorFlow installation problems

Confirm:

```bash
python --version
```

and use Python 3.12.x with a fresh virtual environment.

## Tech stack

- Python 3.12
- Flask
- TensorFlow / Keras
- EfficientNetB2
- NumPy
- Pillow
- HTML5
- CSS3
- JavaScript
