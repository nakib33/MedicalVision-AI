"""MedScan-AI — Flagship Project Configuration.

Supports multiple imaging modalities:
- Brain MRI
- Chest X-ray
- Lung CT
- Retinal Fundus
- Skin Lesion
- Breast Histopathology
"""

PROJECT_ID = "12_medscan_ai"
PROJECT_NAME = "MedScan-AI - Explainable Medical Imaging"
SHORT_NAME = "MedScan-AI"
DESCRIPTION = "Unified multi-modality medical imaging analysis with comprehensive Explainable AI (XAI)"

CLASSES = [
    "Normal",
    "Abnormal",
]
IMG_SIZE = 224
MODEL_NAME = "custom_cnn"  # Uses Custom CNN + EfficientNet comparison

DATASET_URL = "https://www.kaggle.com/datasets"
KAGGLE_DATASET = None  # Multi-dataset; each modality loaded separately

MODALITIES = [
    {
        "id": "brain_mri",
        "name": "Brain MRI",
        "icon": "🧠",
        "description": "T1-weighted brain MRI scans for structural analysis",
        "classes": ["Normal", "Glioma", "Meningioma", "Pituitary"],
    },
    {
        "id": "chest_xray",
        "name": "Chest X-ray",
        "icon": "🫁",
        "description": "Posterior-anterior chest radiographs",
        "classes": ["Normal", "Pneumonia", "COVID", "Tuberculosis"],
    },
    {
        "id": "lung_ct",
        "name": "Lung CT",
        "icon": "🫁",
        "description": "CT scans of lung tissue for nodule analysis",
        "classes": ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma"],
    },
    {
        "id": "retinal_fundus",
        "name": "Retinal Fundus",
        "icon": "👁️",
        "description": "Fundus photography for retinal disease screening",
        "classes": ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"],
    },
    {
        "id": "skin_lesion",
        "name": "Skin Lesion",
        "icon": "🔬",
        "description": "Dermoscopic images of skin lesions",
        "classes": ["Benign", "Malignant"],
    },
    {
        "id": "breast_histopathology",
        "name": "Breast Histopathology",
        "icon": "🔬",
        "description": "Histopathology slides for breast cancer screening",
        "classes": ["IDC Negative", "IDC Positive"],
    },
]

CLASS_COLORS = {
    "Normal": "#38a169",
    "Abnormal": "#e53e3e",
}
