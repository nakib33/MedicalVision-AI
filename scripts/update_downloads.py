"""Rewrite all remaining download scripts with manual-download support."""
import os

PROJECTS_DIR = "projects"

# Read template from external file
TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "download_template.txt")

TEMPLATE = r'''"""Download __PROJECT_NAME__ dataset from Kaggle.

USAGE:
  Option 1 - Automatic (requires Kaggle API key):
      python download_data.py

  Option 2 - Manual (if API fails):
      1. Go to: __KAGGLE_URL__
      2. Click "Download" to get the ZIP
      3. Place the ZIP in this folder
      4. Run: python download_data.py manual
"""

import sys
import random
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = __CLASSES_LIST__
DATASET_ID = "__DATASET_ID__"

FOLDER_MAP = {
    cls.lower(): cls for cls in CLASSES
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def organise_images(source_dir):
    print("\nScanning " + str(source_dir) + " for images...")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    image_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    class_images = {cls: [] for cls in CLASSES}
    for file_path in source_dir.rglob("*"):
        if file_path.suffix.lower() not in image_ext:
            continue
        if "__MACOSX" in str(file_path):
            continue
        parent = file_path.parent.name.lower()
        for key, val in FOLDER_MAP.items():
            if key in parent:
                class_images[val].append(file_path)
                break
    total = sum(len(v) for v in class_images.values())
    if total == 0:
        print("  ! No images found! Check folder structure.")
        print("  ! Looking for folders: " + str(list(FOLDER_MAP.keys())))
        # Debug: show actual folders
        for f in sorted(source_dir.rglob("*")):
            if f.is_dir() and f.parent != source_dir:
                print("    Found: " + str(f.relative_to(source_dir)))
        return
    for cls_name, images in class_images.items():
        random.shuffle(images)
        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        for split_name, subset in [
            ("train", images[:train_end]),
            ("val", images[train_end:val_end]),
            ("test", images[val_end:]),
        ]:
            for src_path in subset:
                shutil.copy2(src_path, DATA_DIR / split_name / cls_name / src_path.name)
    print("  [OK] Organised " + str(total) + " images")
    print("\nSummary:")
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
    print("\n  [OK] Total: " + str(total_all) + " images ready at: " + str(DATA_DIR))


def try_kagglehub():
    try:
        import kagglehub
        print("\nTrying automatic download via kagglehub...")
        path = kagglehub.dataset_download(DATASET_ID)
        print("  Downloaded to: " + str(path))
        organise_images(Path(path))
        return True
    except Exception as e:
        print("  Automatic download failed: " + str(e))
        return False


def manual_mode():
    zips = list(PROJECT_DIR.glob("*.zip"))
    if zips:
        zip_path = zips[0]
        print("\nFound ZIP: " + zip_path.name)
        extract_dir = PROJECT_DIR / "temp_extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        organise_images(extract_dir)
        shutil.rmtree(extract_dir)
        return True
    folders = [d for d in PROJECT_DIR.iterdir()
               if d.is_dir() and d.name not in ["data", "static", "__pycache__", "temp_extracted"]]
    if folders:
        print("\nFound folder: " + folders[0].name)
        organise_images(folders[0])
        return True
    print("\n" + "!" * 60)
    print("  No ZIP or dataset folder found!")
    print("  Go to: __KAGGLE_URL__")
    print("  Download the ZIP and place it in: " + str(PROJECT_DIR))
    print("  Then run: python download_data.py manual")
    print("!" * 60)
    return False


def download():
    print("=" * 60)
    print("PROJECT___PROJECT_NAME__")
    print("=" * 60)
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if mode == "manual":
        manual_mode()
    else:
        if not try_kagglehub():
            print("\nSwitching to manual mode...")
            manual_mode()


if __name__ == "__main__":
    download()
'''


PROJECTS = {
    "05_alzheimers": {
        "name": "Alzheimer MRI Classification",
        "classes": '["Non Demented", "Very Mild", "Mild", "Moderate"]',
        "dataset_id": "priyaltandel/alzheimer-mri-classification",
        "kaggle_url": "https://www.kaggle.com/datasets/priyaltandel/alzheimer-mri-classification",
    },
    "06_skin_cancer": {
        "name": "Skin Cancer Classification",
        "classes": '["Benign", "Malignant"]',
        "dataset_id": "kmader/skin-cancer-mnist-ham10000",
        "kaggle_url": "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
    },
    "07_diabetic_retinopathy": {
        "name": "Diabetic Retinopathy Detection",
        "classes": '["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]',
        "dataset_id": "mariaherrerot/aptos2019",
        "kaggle_url": "https://www.kaggle.com/datasets/mariaherrerot/aptos2019",
    },
    "08_malaria": {
        "name": "Malaria Cell Classification",
        "classes": '["Parasitized", "Uninfected"]',
        "dataset_id": "iarunava/cell-images-for-detecting-malaria",
        "kaggle_url": "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
    },
    "09_breast_cancer": {
        "name": "Breast Cancer Histopathology",
        "classes": '["IDC Negative", "IDC Positive"]',
        "dataset_id": "paultimothymooney/breast-histopathology-images",
        "kaggle_url": "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images",
    },
    "10_bone_fracture": {
        "name": "Bone Fracture Detection",
        "classes": '["Normal", "Fractured"]',
        "dataset_id": "pkdarabi/bone-fracture-detection-computer-vision-project",
        "kaggle_url": "https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project",
    },
    "11_lung_cancer": {
        "name": "Lung Cancer CT Scan Classification",
        "classes": '["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"]',
        "dataset_id": "andrewmvd/lung-cancer-subtype-classification",
        "kaggle_url": "https://www.kaggle.com/datasets/andrewmvd/lung-cancer-subtype-classification",
    },
}


def make_script(pid, info):
    """Generate download script by replacing placeholders in template."""
    script = TEMPLATE
    script = script.replace("PROJECT___PROJECT_NAME__", info["name"])
    script = script.replace("CLASSES_LIST", info["classes"])
    script = script.replace("DATASET_ID", info["dataset_id"])
    script = script.replace("KAGGLE_URL", info["kaggle_url"])
    return script


for pid, info in PROJECTS.items():
    path = os.path.join(PROJECTS_DIR, pid, "download_data.py")
    content = make_script(pid, info)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Updated: {path}")

print(f"\nDone! {len(PROJECTS)} scripts updated")
