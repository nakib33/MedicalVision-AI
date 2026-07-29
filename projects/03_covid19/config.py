"""COVID-19 Chest X-ray Classification - Project Configuration."""

PROJECT_ID = "03_covid19"
PROJECT_NAME = "COVID-19 Chest X-ray Classification"
SHORT_NAME = "COVID-19"
DESCRIPTION = "Multi-class classification - COVID, Viral Pneumonia, Normal"

CLASSES = ["COVID", "Viral Pneumonia", "Normal"]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset"
KAGGLE_DATASET = "pranavraikokte/covid19-image-dataset"

CLASS_COLORS = {
    "COVID": "#3182ce",
    "Viral Pneumonia": "#3182ce",
    "Normal": "#3182ce",
}
