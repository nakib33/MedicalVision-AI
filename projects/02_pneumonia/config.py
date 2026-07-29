"""Pneumonia Detection - Project Configuration."""

PROJECT_ID = "02_pneumonia"
PROJECT_NAME = "Pneumonia Detection"
SHORT_NAME = "Pneumonia"
DESCRIPTION = "Binary classification of chest X-rays - Normal vs Pneumonia"

CLASSES = ["Normal", "Pneumonia"]
IMG_SIZE = 224
MODEL_NAME = "densenet121"

DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

CLASS_COLORS = {
    "Normal": "#e53e3e",
    "Pneumonia": "#e53e3e",
}
