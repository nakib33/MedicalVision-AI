"""RetinaVision AI - Multi-Disease Retinal Disease Classification."""

PROJECT_ID = "13_retinavision"
PROJECT_NAME = "RetinaVision AI - Retinal Disease Classification"
SHORT_NAME = "RetinaVision"
DESCRIPTION = "Multi-class retinal fundus image analysis - 8 eye disease classes"

CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy",
    "Myopia",
    "Other",
]
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"

DATASET_URL = "https://www.kaggle.com/datasets/sshikamaru/retinal-fundus-multi-disease-image-dataset-rfmid"
KAGGLE_DATASET = "sshikamaru/retinal-fundus-multi-disease-image-dataset-rfmid"

CLASS_COLORS = {
    "Normal": "#38a169",
    "Diabetic Retinopathy": "#e53e3e",
    "Glaucoma": "#3182ce",
    "Cataract": "#dd6b20",
    "Age-related Macular Degeneration": "#805ad5",
    "Hypertensive Retinopathy": "#d53f8c",
    "Myopia": "#319795",
    "Other": "#718096",
}
