"""Breast Cancer Histopathology - Project Configuration."""

PROJECT_ID = "09_breast_cancer"
PROJECT_NAME = "Breast Cancer Histopathology"
SHORT_NAME = "Breast Cancer"
DESCRIPTION = "Binary - IDC Positive vs IDC Negative histopathology"

CLASSES = ["IDC Negative", "IDC Positive"]
IMG_SIZE = 224
MODEL_NAME = "densenet121"

DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images"
KAGGLE_DATASET = "paultimothymooney/breast-histopathology-images"

CLASS_COLORS = {
    "IDC Negative": "#e53e3e",
    "IDC Positive": "#e53e3e",
}
