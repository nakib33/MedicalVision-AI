"""Skin Cancer Classification - Project Configuration."""

PROJECT_ID = "06_skin_cancer"
PROJECT_NAME = "Skin Cancer Classification"
SHORT_NAME = "Skin Cancer"
DESCRIPTION = "Binary - Benign vs Malignant skin lesions (HAM10000)"

CLASSES = ["Benign", "Malignant"]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

CLASS_COLORS = {
    "Benign": "#d53f8c",
    "Malignant": "#d53f8c",
}
