"""GastroVision AI - Gastrointestinal Disease Classification."""

PROJECT_ID = "14_gi_tract"
PROJECT_NAME = "GastroVision AI - GI Tract Disease Classification"
SHORT_NAME = "GastroVision"
DESCRIPTION = "Multi-class endoscopic image classification - 8 GI tract conditions"

CLASSES = [
    "Esophagitis",
    "Polyps",
    "Ulcerative Colitis",
    "Dyed Lifted Polyps",
    "Dyed Resection Margins",
    "Cecum",
    "Pylorus",
    "Z-Line",
]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset"
KAGGLE_DATASET = "plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset"

CLASS_COLORS = {
    "Esophagitis": "#e53e3e",
    "Polyps": "#dd6b20",
    "Ulcerative Colitis": "#d53f8c",
    "Dyed Lifted Polyps": "#3182ce",
    "Dyed Resection Margins": "#805ad5",
    "Cecum": "#38a169",
    "Pylorus": "#319795",
    "Z-Line": "#718096",
}
