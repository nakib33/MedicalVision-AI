"""Tuberculosis Detection - Project Configuration."""

PROJECT_ID = "04_tuberculosis"
PROJECT_NAME = "Tuberculosis Detection"
SHORT_NAME = "TB Detection"
DESCRIPTION = "Binary classification - Tuberculosis vs Normal from chest X-rays"

CLASSES = ["Normal", "Tuberculosis"]
IMG_SIZE = 224
MODEL_NAME = "resnet50"

DATASET_URL = "https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
KAGGLE_DATASET = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"

CLASS_COLORS = {
    "Normal": "#dd6b20",
    "Tuberculosis": "#dd6b20",
}
