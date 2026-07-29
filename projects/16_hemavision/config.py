"""HemaVision AI - Blood Cell Disease Classification."""

PROJECT_ID = "16_hemavision"
PROJECT_NAME = "HemaVision AI - Blood Cell Classification"
SHORT_NAME = "HemaVision"
DESCRIPTION = "Multi-class blood cell classification - 4 white blood cell types"

CLASSES = [
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
]
IMG_SIZE = 224
MODEL_NAME = "resnet18"

DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/blood-cells"
KAGGLE_DATASET = "paultimothymooney/blood-cells"

CLASS_COLORS = {
    "Eosinophil": "#d53f8c",
    "Lymphocyte": "#3182ce",
    "Monocyte": "#dd6b20",
    "Neutrophil": "#805ad5",
}
