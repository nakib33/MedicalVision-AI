"""Alzheimer MRI Classification - Project Configuration."""

PROJECT_ID = "05_alzheimers"
PROJECT_NAME = "Alzheimer MRI Classification"
SHORT_NAME = "Alzheimer"
DESCRIPTION = "Multi-class - Non Demented, Very Mild, Mild, Moderate"

CLASSES = ["Non Demented", "Very Mild", "Mild", "Moderate"]
IMG_SIZE = 224
MODEL_NAME = "densenet121"

DATASET_URL = "https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
KAGGLE_DATASET = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"

CLASS_COLORS = {
    "Non Demented": "#805ad5",
    "Very Mild": "#805ad5",
    "Mild": "#805ad5",
    "Moderate": "#805ad5",
}
