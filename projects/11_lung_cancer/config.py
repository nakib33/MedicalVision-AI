"""Lung Cancer CT Scan Classification - Project Configuration.

Dataset: dishantrathi20/ct-scan-images-for-lung-cancer
6 classes: Benign, Malignant, Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma
"""

PROJECT_ID = "11_lung_cancer"
PROJECT_NAME = "Lung Cancer CT Scan Classification"
SHORT_NAME = "Lung Cancer"
DESCRIPTION = "6-class classification of lung CT scans: Benign, Malignant, Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma & Normal"

CLASSES = [
    "Benign",
    "Malignant",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/dishantrathi20/ct-scan-images-for-lung-cancer"
KAGGLE_DATASET = "dishantrathi20/ct-scan-images-for-lung-cancer"

CLASS_COLORS = {
    "Benign": "#38a169",                 # green — non-cancerous
    "Malignant": "#e53e3e",              # red — cancerous
    "Adenocarcinoma": "#6b46c1",         # purple
    "Large Cell Carcinoma": "#dd6b20",   # orange
    "Normal": "#3182ce",                 # blue
    "Squamous Cell Carcinoma": "#319795",# teal
}

# Number of classes — used by training and inference
NUM_CLASSES = len(CLASSES)
