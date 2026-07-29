"""OralScan AI - Oral Disease Classification."""

PROJECT_ID = "15_oralscan"
PROJECT_NAME = "OralScan AI - Oral Disease Detection"
SHORT_NAME = "OralScan"
DESCRIPTION = "Multi-class oral disease classification - 5 oral health conditions"

CLASSES = [
    "Healthy",
    "Dental Caries",
    "Gingivitis",
    "Oral Ulcer",
    "Leukoplakia",
]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/nourelhoda2020/oral-diseases-dataset"
KAGGLE_DATASET = "nourelhoda2020/oral-diseases-dataset"

CLASS_COLORS = {
    "Healthy": "#38a169",
    "Dental Caries": "#e53e3e",
    "Gingivitis": "#dd6b20",
    "Oral Ulcer": "#d53f8c",
    "Leukoplakia": "#805ad5",
}
