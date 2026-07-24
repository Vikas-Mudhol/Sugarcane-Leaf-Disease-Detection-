# 🌱 Sugarcane Leaf Disease Detection System

An end-to-end deep learning system that detects diseases in sugarcane leaves from a single photo. It validates that the uploaded image is genuinely a sugarcane leaf before classifying it, then returns a treatment recommendation — all through a simple web interface.

Built with **EfficientNet-B4** for classification and **OpenAI CLIP** for zero-shot leaf validation, wrapped in a **Flask** web app.

---

## ✨ Features

- 🔍 **3-gate input validation** — rejects non-leaf and wrong-crop images before running the disease model, instead of confidently guessing on garbage input
- 🧠 **EfficientNet-B4** classifier fine-tuned via 3-phase progressive unfreezing
- 🎯 **Test-Time Augmentation (TTA)** — averages predictions across 4 image views for more stable results
- 🌾 **5-class detection**: Healthy, Mosaic, RedRot, Rust, Yellow
- 💊 **Treatment recommendations** — fertilizer, soil care, cultivation practices, and pesticide guidance per disease
- 🖥️ **Web interface** — drag-and-drop upload with confidence scores and results display

---

## 🏗️ How It Works

```
Upload Image
      │
      ▼
① Green-Pixel Check  ──── fails ───▶  Reject: "No plant detected"
      │ passes
      ▼
② CLIP Leaf Validation ── fails ───▶  Reject: "Not a sugarcane leaf"
      │ passes
      ▼
③ EfficientNet-B4 + TTA (4-view averaged prediction)
      │
      ▼
Confidence ≥ 0.55?  ── no ───▶  Flagged as low-confidence
      │ yes
      ▼
Disease class + Treatment recommendation returned
```

1. **Green-pixel heuristic** — a fast, rule-based check confirming the image plausibly contains a plant
2. **CLIP zero-shot validation** — [OpenAI CLIP](https://github.com/openai/CLIP) (`clip-vit-base-patch32`) checks the image against sugarcane-leaf vs. non-leaf text prompts
3. **Disease classification** — EfficientNet-B4 (ImageNet-pretrained, fine-tuned) with 4-view TTA predicts the disease class and confidence

---

## 📁 Project Structure

```
sugarcane_v2/
├── config.py              # Central configuration (paths, hyperparameters, classes)
├── dataset.py              # Dataset loading, Albumentations augmentation, TTA transforms
├── model.py                 # EfficientNet-B4 architecture + staged unfreezing
├── utils.py                  # Seeding, Mixup/CutMix helpers
├── train.py                   # 3-phase training loop
├── main.py                     # Training entry point
├── evaluate.py                  # Model evaluation — accuracy, confusion matrix, classification report
├── predict.py                    # CLI single-image inference (green-pixel + CLIP + model)
├── leaf_validator.py              # CLIP-based sugarcane leaf validator
├── app.py                          # Flask web application
├── templates/
│   └── index.html                   # Frontend UI
├── static/
│   └── uploads/                       # Uploaded images saved here
├── outputs/                             # Trained model + evaluation artifacts (generated)
└── requirements.txt                       # Python dependencies
```

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet-pretrained) |
| Classifier head | Dropout → Linear(1792→512) → BatchNorm → SiLU → Dropout → Linear(512→5) |
| Input size | 224 × 224 |
| Classes | Healthy, Mosaic, RedRot, Rust, Yellow |
| Training strategy | 3-phase progressive fine-tuning (head-only → last 3 blocks → full backbone) |
| Regularization | Mixup, CutMix, label smoothing, dropout |
| Optimizer | AdamW with warmup + cosine annealing LR schedule |
| Inference | 4-view Test-Time Augmentation (original, h-flip, v-flip, 90° rotation) |

---

## ⚙️ Requirements

- Python 3.9+
- CUDA-capable GPU recommended (tested on RTX 4060 8GB); CPU fallback supported
- Windows or Linux

Dependencies (see `requirements.txt`):

```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
Pillow>=9.0.0
scikit-learn>=1.2.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-image>=0.20.0
transformers>=4.35.0
flask>=3.0.0
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sugarcane-leaf-disease-detection.git
cd sugarcane-leaf-disease-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch with CUDA (recommended)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Check your CUDA version with `nvidia-smi` and adjust the `cu121` suffix if needed. For CPU-only, skip this step — the next step installs a CPU build.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare your dataset

Organize your dataset folder like this:

```
dataset/
├── Healthy/
│   ├── img001.jpg
│   └── ...
├── Mosaic/
├── RedRot/
├── Rust/
└── Yellow/
```

Then update `DATA_DIR` in `config.py`:

```python
DATA_DIR = r'path/to/your/dataset'
```

### 6. Train the model

```bash
python main.py
```

This runs all 3 training phases and saves:
- `outputs/best_model.pth` — best model weights (by validation accuracy)
- `outputs/training_curves.png` — accuracy/loss plots

### 7. Evaluate the model

```bash
python evaluate.py
```

Generates:
- Overall and per-class accuracy, precision, recall, F1-score
- `outputs/confusion_matrix.png`
- Full classification report

### 8. Run inference on a single image (CLI)

```bash
python predict.py --image path/to/leaf.jpg
```

Add `--no-tta` to skip test-time augmentation for faster (slightly less accurate) predictions.

### 9. Launch the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser, then drag and drop a leaf image to get a diagnosis and treatment plan.

---

## ⚡ Configuration

All key settings live in `config.py`:

| Setting | Description | Default |
|---|---|---|
| `DATA_DIR` | Path to dataset folder | *(must be set)* |
| `IMG_SIZE` | Input image resolution | `224` |
| `BATCH_SIZE` | Training batch size | `32` |
| `EPOCHS` | Max training epochs | `60` |
| `LR` | Initial learning rate | `1e-4` |
| `EARLY_STOP` | Early stopping patience | `12` |
| `NUM_WORKERS` | DataLoader worker processes | `0` (required on Windows) |

---

## 🧪 Testing

The project includes a layered validation pipeline that should be tested against:

| Scenario | Expected Behavior |
|---|---|
| Valid sugarcane leaf image | Correct disease class + confidence ≥ 0.55 |
| Non-leaf image (object, face, etc.) | Rejected at the green-pixel or CLIP gate |
| Different plant's leaf (mango, banana, etc.) | Rejected at the CLIP gate |
| Blurry / low-confidence image | Flagged as low-confidence rather than a false diagnosis |
| Unsupported file type via API | `400` error with a clear message |

---

## 📊 Sample Output

| Confusion Matrix | Training Curves |
|---|---|
| Per-class prediction accuracy heatmap | Accuracy/loss across all 3 training phases |

*(Generated automatically in `outputs/` after running `train.py`/`main.py` and `evaluate.py`.)*

---

## ⚠️ Known Limitations

- Model accuracy depends heavily on dataset quality and size — results should be validated on your own held-out test set before production use
- The CLIP leaf-validation threshold (0.40) and confidence threshold (0.55) are heuristically set and may need tuning for your specific data
- `app.py` currently runs Flask's built-in development server — use a production WSGI server (e.g., Gunicorn + Nginx) for real deployment
- Not a substitute for professional agricultural diagnosis — treatment recommendations are general guidance, not certified agronomic advice

---

## 🛠️ Tech Stack

- **PyTorch** / **Torchvision** — model training and inference
- **EfficientNet-B4** — disease classification backbone
- **Hugging Face Transformers** — CLIP model for leaf validation
- **Albumentations** — image augmentation
- **Flask** — web application backend
- **scikit-learn** — evaluation metrics

---

## 📄 License

*(Add your chosen license here — e.g., MIT, Apache 2.0. If unsure, [choosealicense.com](https://choosealicense.com/) can help you pick one.)*

---

## 🙏 Acknowledgements

- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, Google AI (2019)
- [CLIP](https://github.com/openai/CLIP) — OpenAI (2021)
- Built as part of an MCA academic project

---

## 📬 Contact

*(Add your name/GitHub/email here for others to reach you.)*
