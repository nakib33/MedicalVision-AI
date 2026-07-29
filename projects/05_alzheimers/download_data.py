"""Download Alzheimer MRI Classification dataset from Kaggle."""

import random
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Non Demented", "Very Mild", "Mild", "Moderate"]
DATASET_ID = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"

# Map folder names in ZIP to standard class names
FOLDER_MAP = {
    "nondemented": "Non Demented",
    "verymilddemented": "Very Mild",
    "milddemented": "Mild",
    "moderatedemented": "Moderate",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def organise_images(source_dir):
    """Organise images from extracted folder into train/val/test."""
    print(f"\nScanning {source_dir} for images...")

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

    # Split per class into train/val/test
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
                dest = DATA_DIR / split_name / cls_name / src_path.name
                shutil.copy2(src_path, dest)

    print(f"  [OK] Organised {total} images")
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
                        print(f"    {split}/{cls_dir.name}: {n} images")
                        split_total += n
            print(f"    {split} total: {split_total} images")
            total_all += split_total
    print(f"\n  [OK] Total: {total_all} images ready at: {DATA_DIR}")


def process_zip(zip_path):
    """Extract ZIP and organise images."""
    print(f"\nFound ZIP: {zip_path.name}")
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
    print("Alzheimer MRI Classification Dataset Downloader")
    print("=" * 60)

    # Look for ZIP in project folder
    zips = list(PROJECT_DIR.glob("*.zip"))
    if zips:
        process_zip(zips[0])
        return

    # Look for extracted folder
    folders = [d for d in PROJECT_DIR.iterdir()
               if d.is_dir() and d.name not in ["data", "static", "__pycache__", "temp_extracted"]]
    if folders:
        print(f"\nFound folder: {folders[0].name}")
        organise_images(folders[0])
        return

    print("\n" + "!" * 60)
    print("  No dataset found!")
    print(f"  Download from: https://www.kaggle.com/datasets/{DATASET_ID}")
    print("  Place the ZIP file in this folder and run again.")
    print("!" * 60)


if __name__ == "__main__":
    download()
