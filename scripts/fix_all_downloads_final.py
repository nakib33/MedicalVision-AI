"""Rewrite all remaining download scripts with clean format."""
import os

PROJECTS_DIR = "projects"

TEMPLATE = '''"""Download PROJECT_NAME dataset from Kaggle."""

import random
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = CLASSES_LIST
DATASET_ID = "DATASET_ID_VALUE"

FOLDER_MAP = {
    "class1_lower": "Class1",
    "class2_lower": "Class2",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def organise_images(source_dir):
    print(f"\\nScanning {source_dir} for images...")
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
        print("  ! No images found! Debugging folder structure:")
        for f in sorted(source_dir.rglob("*")):
            if f.is_dir() and f.parent != source_dir:
                depth = len(f.relative_to(source_dir).parts)
                if depth <= 2:
                    print(f"    Folder: {f.relative_to(source_dir)}")
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
    print(f"  [OK] Organised {total} images")
    print("\\nSummary:")
    total_all = 0
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            split_total = 0
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    n = len(list(cls_dir.glob("*")))
                    if n > 0:
                        print(f"    {split}/{cls_dir.name}: {n} images")
                        split_total += n
            print(f"    {split} total: {split_total} images")
            total_all += split_total
    print(f"\\n  [OK] Total: {total_all} images ready at: {DATA_DIR}")


def process_zip(zip_path):
    print(f"\\nFound ZIP: {zip_path.name}")
    extract_dir = PROJECT_DIR / "temp_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    organise_images(extract_dir)
    shutil.rmtree(extract_dir)


def download():
    print("=" * 60)
    print("PROJECT_NAME")
    print("=" * 60)
    zips = list(PROJECT_DIR.glob("*.zip"))
    if zips:
        process_zip(zips[0])
        return
    folders = [d for d in PROJECT_DIR.iterdir()
               if d.is_dir() and d.name not in ["data", "static", "__pycache__", "temp_extracted"]]
    if folders:
        print(f"\\nFound folder: {folders[0].name}")
        organise_images(folders[0])
        return
    print("\\n" + "!" * 60)
    print("  No dataset found!")
    print(f"  Download from: https://www.kaggle.com/datasets/DATASET_ID_VALUE")
    print("  Place the ZIP in this folder and run again.")
    print("!" * 60)


if __name__ == "__main__":
    download()
'''

PROJECTS = {
    "06_skin_cancer": {
        "name": "Skin Cancer Classification",
        "classes": '["Benign", "Malignant"]',
        "dataset_id": "kmader/skin-cancer-mnist-ham10000",
        "folder_map": '''{
    "benign": "Benign",
    "malignant": "Malignant",
}''',
    },
    "07_diabetic_retinopathy": {
        "name": "Diabetic Retinopathy Detection",
        "classes": '["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]',
        "dataset_id": "mariaherrerot/aptos2019",
        "folder_map": '''{
    "no_dr": "No DR",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
    "proliferative_dr": "Proliferative DR",
}''',
    },
    "09_breast_cancer": {
        "name": "Breast Cancer Histopathology",
        "classes": '["IDC Negative", "IDC Positive"]',
        "dataset_id": "paultimothymooney/breast-histopathology-images",
        "folder_map": '''{
    "class0": "IDC Negative",
    "class1": "IDC Positive",
}''',
    },
    "10_bone_fracture": {
        "name": "Bone Fracture Detection",
        "classes": '["Normal", "Fractured"]',
        "dataset_id": "pkdarabi/bone-fracture-detection-computer-vision-project",
        "folder_map": '''{
    "normal": "Normal",
    "fractured": "Fractured",
    "fracture": "Fractured",
}''',
    },
    "11_lung_cancer": {
        "name": "Lung Cancer CT Scan Classification",
        "classes": '["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"]',
        "dataset_id": "andrewmvd/lung-cancer-subtype-classification",
        "folder_map": '''{
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large Cell Carcinoma",
    "largecellcarcinoma": "Large Cell Carcinoma",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
    "squamouscellcarcinoma": "Squamous Cell Carcinoma",
    "normal": "Normal",
}''',
    },
}


for pid, info in PROJECTS.items():
    script = TEMPLATE
    script = script.replace("PROJECT_NAME", info["name"])
    script = script.replace("CLASSES_LIST", info["classes"])
    script = script.replace("DATASET_ID_VALUE", info["dataset_id"])
    script = script.replace('''{
    "class1_lower": "Class1",
    "class2_lower": "Class2",
}''', info["folder_map"])

    path = os.path.join(PROJECTS_DIR, pid, "download_data.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Fixed: {path}")

print(f"\nDone! {len(PROJECTS)} scripts fixed")
