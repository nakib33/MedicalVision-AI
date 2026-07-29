"""Brain Tumor MRI Classification - Project Configuration."""

PROJECT_ID = "01_brain_tumor"
PROJECT_NAME = "Brain Tumor MRI Classification"
SHORT_NAME = "Brain Tumor"
DESCRIPTION = "Multi-class classification of brain MRI scans into Glioma, Meningioma, Pituitary, or Normal"

CLASSES = ["Glioma", "Meningioma", "Pituitary", "Normal"]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri"
KAGGLE_DATASET = "sartajbhuvaji/brain-tumor-classification-mri"

CLASS_COLORS = {
    "Glioma": "#e53e3e",
    "Meningioma": "#dd6b20",
    "Pituitary": "#3182ce",
    "Normal": "#38a169",
}
