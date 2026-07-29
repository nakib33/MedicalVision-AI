"""Bone Fracture Detection - Project Configuration."""

PROJECT_ID = "10_bone_fracture"
PROJECT_NAME = "Bone Fracture Detection"
SHORT_NAME = "Bone Fracture"
DESCRIPTION = "Binary - Fractured vs Normal bone X-rays"

CLASSES = ["Normal", "Fractured"]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project"
KAGGLE_DATASET = "pkdarabi/bone-fracture-detection-computer-vision-project"

CLASS_COLORS = {
    "Normal": "#2b6cb0",
    "Fractured": "#2b6cb0",
}
