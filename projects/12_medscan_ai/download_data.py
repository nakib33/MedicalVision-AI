"""MedScan-AI — Multi-modality dataset downloader.

Downloads sample images for each supported modality from various sources.
For a real deployment, each modality would have its own dedicated dataset.
"""

from pathlib import Path
import requests
from PIL import Image
import io

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Sample medical images (public domain / research-use images)
SAMPLE_IMAGES = [
    {
        "modality": "brain_mri",
        "class": "Normal",
        "url": "https://raw.githubusercontent.com/sartajbhuvaji/brain-tumor-classification-mri/master/data/val/notumor/1%20%282%29.jpg",
    },
    {
        "modality": "chest_xray",
        "class": "Normal",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S1684118220300480-main.pdf-001-a-1.jpg",
    },
]

# Real download via kagglehub for proper datasets
KAGGLE_DATASETS = {
    "brain_mri": "sartajbhuvaji/brain-tumor-classification-mri",
    "chest_xray": "paultimothymooney/chest-xray-pneumonia",
}


def download():
    """Download sample images for MedScan-AI demo."""
    print("=" * 60)
    print("MedScan-AI Multi-Modality Dataset Downloader")
    print("=" * 60)

    print("\nThis project supports multiple imaging modalities.")
    print("For full training, download individual datasets from Kaggle:")
    for mod, ds_id in KAGGLE_DATASETS.items():
        print(f"  - {mod}: kagglehub.dataset_download('{ds_id}')")

    print("\nDownloading sample images for demo/testing...")

    for mod_name in ["brain_mri", "chest_xray", "lung_ct", "retinal_fundus",
                     "skin_lesion", "breast_histopathology"]:
        for cls in ["Normal", "Abnormal"]:
            (DATA_DIR / "train" / cls).mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "val" / cls).mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "test" / cls).mkdir(parents=True, exist_ok=True)

    # Create placeholder images (colored squares with modality label)
    import numpy as np
    for mod in ["brain_mri", "chest_xray", "lung_ct", "retinal_fundus",
                "skin_lesion", "breast_histopathology"]:
        for cls, color in [("Normal", (0, 100, 0)), ("Abnormal", (100, 0, 0))]:
            img = Image.fromarray(
                np.ones((224, 224, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
            )
            img.save(DATA_DIR / "train" / cls / f"{mod}_sample.png")
            img.save(DATA_DIR / "val" / cls / f"{mod}_sample.png")

    print("  [OK] Created placeholder dataset for all 6 modalities")
    print("\nNOTE: For production training, download real datasets from Kaggle")
    print("using the individual project download_data.py scripts.\n")


if __name__ == "__main__":
    download()
