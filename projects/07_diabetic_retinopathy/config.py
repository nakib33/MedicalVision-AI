"""Diabetic Retinopathy Detection - Project Configuration."""

PROJECT_ID = "07_diabetic_retinopathy"
PROJECT_NAME = "Diabetic Retinopathy Detection"
SHORT_NAME = "Retinopathy"
DESCRIPTION = "Multi-class - 5 stages (APTOS 2019)"

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/mariaherrerot/aptos2019"
KAGGLE_DATASET = "mariaherrerot/aptos2019"

CLASS_COLORS = {
    "No DR": "#319795",
    "Mild": "#319795",
    "Moderate": "#319795",
    "Severe": "#319795",
    "Proliferative DR": "#319795",
}
