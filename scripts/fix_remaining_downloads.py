"""Fix all remaining download scripts with proper random-split approach."""
import os

PROJECTS_DIR = "projects"
# Projects 04 was already fixed manually
to_fix = ["05_alzheimers", "06_skin_cancer", "07_diabetic_retinopathy",
          "08_malaria", "09_breast_cancer", "10_bone_fracture", "11_lung_cancer"]

TEMPLATE = '''"""Download {name} dataset from Kaggle."""
import random
import shutil
from pathlib import Path
import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = {classes}
DATASET_ID = "{dataset_id}"

# Map actual folder names to standard class names
FOLDER_MAP = {{
    cls.lower(): cls for cls in CLASSES
}}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def download():
    print("=" * 60)
    print("{name} Dataset Downloader")
    print("=" * 60)

    print("\\n[1/3] Downloading from Kaggle...")
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    print(f"  [OK] Downloaded to: {{dataset_path}}")

    print("\\n[2/3] Organising into train/val/test...")

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(dataset_path)
    image_ext = {{".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}}

    # Collect images by class from the dataset
    class_images = {{cls: [] for cls in CLASSES}}
    for file_path in downloaded_path.rglob("*"):
        if file_path.suffix.lower() not in image_ext:
            continue
        if "__MACOSX" in str(file_path):
            continue
        parent = file_path.parent.name.lower()
        for key, val in FOLDER_MAP.items():
            if key in parent:
                class_images[val].append(file_path)
                break

    # Check if dataset already has train/val/test folders
    path_str = str(downloaded_path).lower()
    has_splits = any(s in path_str for s in ["/train", "/test", "/val"])

    if has_splits:
        # Use existing split detection
        for split in ["train", "val", "test"]:
            split_class_images = {{cls: [] for cls in CLASSES}}
            for file_path in downloaded_path.rglob("*"):
                if file_path.suffix.lower() not in image_ext:
                    continue
                if "__MACOSX" in str(file_path):
                    continue
                p = str(file_path).lower()
                s = None
                if "\\\\train\\\\" in p or "/train/" in p: s = "train"
                elif "\\\\val\\\\" in p or "/val/" in p: s = "val"
                elif "\\\\test\\\\" in p or "/test/" in p: s = "test"
                if s != split: continue
                parent = file_path.parent.name.lower()
                for key, val in FOLDER_MAP.items():
                    if key in parent:
                        shutil.copy2(file_path, DATA_DIR / split / val / file_path.name)
                        break
    else:
        # Random split from flat class folders
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

    total = sum(len(v) for v in class_images.values())
    print(f"  [OK] Organised {{total}} images")

    print("\\n[3/3] Summary:")
    total_all = 0
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            split_total = 0
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    n = len(list(cls_dir.glob("*")))
                    if n > 0:
                        print(f"    {{split}}/{{cls_dir.name}}: {{n}} images")
                        split_total += n
            print(f"    {{split}} total: {{split_total}} images")
            total_all += split_total

    print(f"\\n  [OK] Total: {{total_all}} images ready at: {{DATA_DIR}}")


if __name__ == "__main__":
    download()
'''

# Project metadata for each fix
PROJECT_INFO = {
    "05_alzheimers": {
        "name": "Alzheimer MRI Classification",
        "classes": '["Non Demented", "Very Mild", "Mild", "Moderate"]',
        "dataset_id": "tourist55/alzheimers-dataset-4-class-of-images",
    },
    "06_skin_cancer": {
        "name": "Skin Cancer Classification",
        "classes": '["Benign", "Malignant"]',
        "dataset_id": "kmader/skin-cancer-mnist-ham10000",
    },
    "07_diabetic_retinopathy": {
        "name": "Diabetic Retinopathy Detection",
        "classes": '["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]',
        "dataset_id": "mariaherrerot/aptos2019",
    },
    "08_malaria": {
        "name": "Malaria Cell Classification",
        "classes": '["Parasitized", "Uninfected"]',
        "dataset_id": "iarunava/cell-images-for-detecting-malaria",
    },
    "09_breast_cancer": {
        "name": "Breast Cancer Histopathology",
        "classes": '["IDC Negative", "IDC Positive"]',
        "dataset_id": "paultimothymooney/breast-histopathology-images",
    },
    "10_bone_fracture": {
        "name": "Bone Fracture Detection",
        "classes": '["Normal", "Fractured"]',
        "dataset_id": "pkdarabi/bone-fracture-detection-computer-vision-project",
    },
    "11_lung_cancer": {
        "name": "Lung Cancer CT Scan Classification",
        "classes": '["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"]',
        "dataset_id": "andrewmvd/lung-cancer-subtype-classification",
    },
}

for pid, info in PROJECT_INFO.items():
    path = os.path.join(PROJECTS_DIR, pid, "download_data.py")
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        continue
    content = TEMPLATE.format(
        name=info["name"],
        classes=info["classes"],
        dataset_id=info["dataset_id"],
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Rewritten: {path}")

print(f"\nComplete! {len(to_fix)} scripts rewritten")
