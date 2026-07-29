<div align="center">

# 🏥 MedicalVision-AI

**A unified platform for 16 CNN-based medical imaging classification projects with Explainable AI (XAI)**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Upload a medical scan → Get instant predictions with 7 XAI visual explanations → Download PDF reports**

[Features](#-key-features) • [Projects](#-projects) • [Quick Start](#-quick-start) • [API](#-api-endpoints) • [XAI Methods](#-explainable-ai-methods) • [Architecture](#-architecture)

</div>

---

## 📋 Overview

MedicalVision-AI is a **unified web platform** that brings together **16 deep learning projects** covering a wide spectrum of medical imaging modalities — brain MRI, chest X-ray, lung CT, retinal fundus, skin lesion, bone X-ray, histopathology, and more. Each project provides:

- **Classification** using pre-trained CNN architectures (EfficientNet-B0, ResNet, DenseNet, Custom CNN)
- **Explainability** via 7 state-of-the-art XAI methods
- **Downloadable PDF reports** with prediction results and heatmap galleries
- **Kaggle dataset integration** — each project includes a `download_data.py` script

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **🧠 16 Medical Imaging Projects** | Brain tumor, pneumonia, COVID-19, TB, Alzheimer's, skin cancer, diabetic retinopathy, malaria, breast cancer, bone fracture, lung cancer, MedScan-AI (multi-modality), retina, GI tract, oral scan, hema vision |
| **🔬 7 XAI Explainers** | Grad-CAM, Grad-CAM++, Score-CAM, Saliency Maps, Guided Backpropagation, Integrated Gradients, Occlusion Sensitivity |
| **🌐 Web Dashboard** | FastAPI-powered interactive UI with project grid, search, and per-project analysis pages |
| **🚀 GPU/CPU Auto-Detect** | Seamless CUDA support with automatic fallback to CPU |
| **📄 PDF Reports** | Professional downloadable reports with prediction, confidence, probability bars, and XAI heatmaps |
| **📦 Kaggle Integration** | Automated dataset download scripts for every project |
| **🔌 REST API** | Full API for predictions, explanations, and project metadata |
| **🏆 MedScan-AI Flagship** | Multi-modality project supporting 6 imaging types (Brain MRI, Chest X-ray, Lung CT, Retinal Fundus, Skin Lesion, Breast Histopathology) |

---

## 📂 Projects

| # | Project | Modality | Classes | Model |
|---|---------|----------|---------|-------|
| 01 | **Brain Tumor MRI Classification** | Brain MRI | Glioma, Meningioma, Pituitary, Normal | EfficientNet-B0 |
| 02 | **Pneumonia Detection** | Chest X-ray | Normal, Pneumonia | EfficientNet-B0 |
| 03 | **COVID-19 Detection** | Chest X-ray | Normal, COVID-19, Viral Pneumonia, Bacterial Pneumonia | EfficientNet-B0 |
| 04 | **Tuberculosis Detection** | Chest X-ray | Normal, Tuberculosis | EfficientNet-B0 |
| 05 | **Alzheimer's Detection** | Brain MRI | Mild Demented, Moderate Demented, Non Demented, Very Mild Demented | EfficientNet-B0 |
| 06 | **Skin Cancer Classification** | Dermoscopic | Benign, Malignant | EfficientNet-B0 |
| 07 | **Diabetic Retinopathy Detection** | Retinal Fundus | Mild, Moderate, No DR, Proliferative DR, Severe | EfficientNet-B0 |
| 08 | **Malaria Detection** | Blood Smear | Parasitized, Uninfected | EfficientNet-B0 |
| 09 | **Breast Cancer Classification** | Histopathology | Benign, Malignant | EfficientNet-B0 |
| 10 | **Bone Fracture Detection** | X-ray | Fracture, No Fracture | EfficientNet-B0 |
| 11 | **Lung Cancer CT Scan Classification** | Lung CT | Benign, Malignant, Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma | EfficientNet-B0 |
| 12 | **MedScan-AI — Multi-Modality** | 6 modalities | Normal, Abnormal (per modality) | Custom CNN |
| 13 | **RetinaVision — Retinal Disease** | Retinal Fundus | 8 eye disease classes | EfficientNet-B0 |
| 14 | **GI Tract Classification** | Endoscopy | Normal, Ulcer, Polyp, Bleeding | EfficientNet-B0 |
| 15 | **OralScan — Oral Cancer Detection** | Oral Cavity | Normal, Oral Squamous Cell Carcinoma, Leukoplakia, Lichen Planus | EfficientNet-B0 |
| 16 | **HemaVision — Blood Cell Analysis** | Blood Smear | Normal, Leukemia, Lymphoma, Myeloma | EfficientNet-B0 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch (CUDA optional but recommended for performance)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nakib33/MedicalVision-AI.git
cd MedicalVision-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (optional — run per project)
python projects/01_brain_tumor/download_data.py
# Repeat for other projects as needed

# 4. Start the server
python run.py
```

The server starts at **http://0.0.0.0:8000** by default. Open your browser and explore the project grid.

### Custom Server Options

```bash
python run.py --port 8080            # Custom port
python run.py --host 127.0.0.1       # Localhost only
python run.py --port 8080 --reload   # Hot reload for development
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page — project grid |
| `GET` | `/api/projects` | JSON list of all projects |
| `GET` | `/{project_id}/` | Project-specific analysis UI |
| `POST` | `/{project_id}/predict` | Upload image → get prediction |
| `POST` | `/{project_id}/explain` | Upload image → get prediction + all XAI explanations |
| `GET` | `/{project_id}/report` | Download PDF report (after prediction) |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |
| `GET` | `/redoc` | Alternative API docs (ReDoc) |
| `GET` | `/health` | Health check — device, projects loaded |

### Example: Prediction Request

```bash
curl -X POST http://localhost:8000/01_brain_tumor/predict \
  -F "file=@scan.jpg"
```

```json
{
  "predicted_class": "Glioma",
  "confidence": 0.956,
  "probabilities": {
    "Glioma": 0.956,
    "Meningioma": 0.021,
    "Pituitary": 0.018,
    "Normal": 0.005
  }
}
```

### Example: XAI Explanation Request

```bash
curl -X POST http://localhost:8000/01_brain_tumor/explain \
  -F "file=@scan.jpg"
```

Returns prediction + 7 XAI methods, each with base64-encoded heatmap and overlay images.

---

## 🔬 Explainable AI Methods

MedicalVision-AI integrates **7 XAI techniques** to help interpret model decisions:

| Method | Type | Description |
|---|---|---|
| **Grad-CAM** | Activation-based | Gradient-weighted Class Activation Mapping — highlights regions from the final convolutional layer |
| **Grad-CAM++** | Activation-based | Improved Grad-CAM with better localisation for multiple object instances |
| **Score-CAM** | Activation-based | Score-weighted activation maps — uses forward-pass confidence instead of gradients |
| **Saliency Map** | Gradient-based | Vanilla saliency — gradient of output with respect to input pixels |
| **Guided Backpropagation** | Gradient-based | Modified backpropagation that produces sharper feature visualisations |
| **Integrated Gradients** | Axiomatic | Path-integral attribution from baseline to input (satisifies sensitivity & implementation invariance axioms) |
| **Occlusion Sensitivity** | Perturbation-based | Sliding-window occlusion map — measures prediction drop per image patch |

All explainers produce overlay heatmaps visualised in the web UI and included in downloadable PDF reports.

---

## 🏛️ Architecture

```
MedicalVision-AI/
├── run.py                     # Application launcher (FastAPI + Uvicorn)
├── main.py                    # FastAPI app — route registration, upload handling
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
│
├── projects/                  # 16 individual medical imaging projects
│   ├── __init__.py            # Project discovery & registry
│   ├── 01_brain_tumor/        # Each project: config.py, model_loader.py,
│   ├── 02_pneumonia/          #   download_data.py, model.ipynb, static/
│   ├── 03_covid19/
│   ├── 04_tuberculosis/
│   ├── 05_alzheimers/
│   ├── 06_skin_cancer/
│   ├── 07_diabetic_retinopathy/
│   ├── 08_malaria/
│   ├── 09_breast_cancer/
│   ├── 10_bone_fracture/
│   ├── 11_lung_cancer/
│   ├── 12_medscan_ai/         # Flagship multi-modality project
│   ├── 13_retinavision/
│   ├── 14_gi_tract/
│   ├── 15_oralscan/
│   └── 16_hemavision/
│
├── shared/                    # Shared modules used across all projects
│   ├── config.py              # Central config — device, paths, defaults
│   ├── models/                # Model architectures
│   │   ├── efficientnet.py    # EfficientNet-B0
│   │   ├── resnet.py          # ResNet-18, ResNet-50
│   │   ├── densenet.py        # DenseNet-121
│   │   └── custom_cnn.py      # Custom CNN implementation
│   ├── pipelines/             # Training & inference pipelines
│   │   ├── inference.py       # Single & batch prediction
│   │   ├── train.py           # Training loop with early stopping
│   │   ├── dataset.py         # Dataset & DataLoader utilities
│   │   └── transforms.py      # Image transforms & augmentation
│   ├── explainers/            # XAI explainers
│   │   ├── gradcam.py
│   │   ├── gradcam_pp.py
│   │   ├── scorecam.py
│   │   ├── saliency.py
│   │   ├── guided_backprop.py
│   │   ├── integrated_gradients.py
│   │   ├── occlusion.py
│   │   └── xai_factory.py     # Orchestrator — runs all explainers
│   ├── utils/                 # Utilities
│   │   ├── metrics.py         # Classification metrics
│   │   ├── report.py          # PDF report generator
│   │   └── visualization.py   # Heatmap overlays, charts, base64 encoding
│   ├── templates/             # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── about.html
│   │   └── project_ui.html
│   └── static/                # Static assets (CSS, JS)
│       ├── css/main.css
│       └── js/main.js
│
└── scripts/                   # Utility scripts for project generation
```

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework & pre-trained models |
| `fastapi` / `uvicorn` | Web server & API framework |
| `jinja2` | HTML template rendering |
| `python-multipart` | File upload handling |
| `pillow` / `opencv-python` | Image processing |
| `numpy` / `pandas` | Data manipulation |
| `matplotlib` | Heatmap visualisation & PDF generation |
| `scikit-learn` | Evaluation metrics |
| `kagglehub` | Kaggle dataset download |
| `tqdm` | Progress bars |

Full list in [requirements.txt](requirements.txt).

---

## 📄 PDF Reports

The PDF report generator creates professional medical AI analysis reports including:

- **Cover page** with uploaded scan thumbnail, prediction result, confidence score, and class probability bar chart
- **XAI heatmap gallery** showing all 7 explainer overlays
- **Technical summary** with model info, class names, and methodology descriptions
- **Medical disclaimer** for research and educational purposes

---

## ⚠️ Disclaimer

This project is **for research and educational purposes only**. It does **NOT** constitute a medical diagnosis or professional medical advice. Always consult a qualified healthcare provider for medical decisions. The predictions from these models should not be used as the sole basis for any clinical decision.

---

## 📚 Datasets

Each project includes a `download_data.py` script that fetches the dataset from Kaggle. You'll need a [Kaggle](https://www.kaggle.com) account and API token configured on your machine. Datasets range from public Kaggle competitions and repositories covering chest X-rays, brain MRI, retinal fundus, dermoscopic, histopathology, and other medical imaging modalities.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:
- New medical imaging projects
- Additional XAI methods
- Model architecture improvements
- UI/UX enhancements
- Dataset integrations

---

## 📬 Contact

**Project Maintainer:** Nakib Uddin Ahmed  
**GitHub:** [nakib33](https://github.com/nakib33)  
**Repository:** [github.com/nakib33/MedicalVision-AI](https://github.com/nakib33/MedicalVision-AI)

---

<div align="center">

**MedicalVision-AI** — *AI-Powered Medical Imaging Analysis for Research and Education*

</div>
