"""Malaria Cell Classification - Project Configuration."""

PROJECT_ID = "08_malaria"
PROJECT_NAME = "Malaria Cell Classification"
SHORT_NAME = "Malaria"
DESCRIPTION = "Binary - Parasitized vs Uninfected blood cells"

CLASSES = ["Parasitized", "Uninfected"]
IMG_SIZE = 224
MODEL_NAME = "resnet18"

DATASET_URL = "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria"
KAGGLE_DATASET = "iarunava/cell-images-for-detecting-malaria"

CLASS_COLORS = {
    "Parasitized": "#38a169",
    "Uninfected": "#38a169",
}
