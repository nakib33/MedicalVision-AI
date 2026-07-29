"""Download Breast Cancer Histopathology dataset from Kaggle.

Simple dataset: patient folders with 0/ (IDC Negative) and 1/ (IDC Positive) subfolders.
Splits by PATIENT to prevent data leakage.
"""

import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["IDC Negative", "IDC Positive"]
CLASS_MAP = {"0": "IDC Negative", "1": "IDC Positive"}
DATASET_ID = "paultimothymooney/breast-histopathology-images"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def organise_images(source_dir):
    print("")
    print("Scanning patient folders...")

    # Find patient folders (numeric names with 0/ and 1/ subdirs)
    patient_folders = []
    for f in sorted(source_dir.iterdir()):
        if f.is_dir() and f.name.isdigit():
            has_0 = (f / "0").exists()
            has_1 = (f / "1").exists()
            if has_0 or has_1:
                patient_folders.append(f)

    print("  Found " + str(len(patient_folders)) + " patient folders")

    if len(patient_folders) == 0:
        print("  ! No patient folders found! Debugging...")
        for f in sorted(source_dir.iterdir())[:10]:
            if f.is_dir():
                print("    Folder: " + f.name)
        return

    # Shuffle and split by patient
    random.shuffle(patient_folders)
    total = len(patient_folders)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = [
        ("train", patient_folders[:train_end]),
        ("val", patient_folders[train_end:val_end]),
        ("test", patient_folders[val_end:]),
    ]

    # Create dirs and copy
    total_copied = 0
    for split_name, patients in splits:
        for cls_name in CLASSES:
            (DATA_DIR / split_name / cls_name).mkdir(parents=True, exist_ok=True)

        copied = 0
        for patient in patients:
            for class_id, cls_name in CLASS_MAP.items():
                src_dir = patient / class_id
                if not src_dir.exists():
                    continue
                for img_path in src_dir.iterdir():
                    if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        shutil.copy2(img_path, DATA_DIR / split_name / cls_name / img_path.name)
                        copied += 1
        print("  [" + split_name + "] " + str(copied) + " images from "
              + str(len(patients)) + " patients")
        total_copied += copied

    print("")
    print("  [OK] Total: " + str(total_copied) + " images")
    print("")
    print("Summary:")
    total_all = 0
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            split_total = 0
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    n = len(list(cls_dir.glob("*")))
                    if n > 0:
                        print("    " + split + "/" + cls_dir.name + ": " + str(n) + " images")
                        split_total += n
            print("    " + split + " total: " + str(split_total) + " images")
            total_all += split_total
    print("")
    print("  [OK] Dataset ready at: " + str(DATA_DIR))


def auto_download():
    try:
        import kagglehub
        print("")
        print("Downloading from Kaggle...")
        path = kagglehub.dataset_download(DATASET_ID)
        print("  Downloaded to: " + str(path))
        organise_images(Path(path))
        return True
    except Exception as e:
        print("  Download failed: " + str(e))
        return False


def download():
    print("=" * 60)
    print("Breast Cancer Histopathology Dataset Downloader")
    print("=" * 60)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    if not auto_download():
        print("")
        print("! Manual download:")
        print("  https://www.kaggle.com/datasets/" + DATASET_ID)


if __name__ == "__main__":
    download()
