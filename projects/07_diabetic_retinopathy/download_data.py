"""Download Diabetic Retinopathy Detection dataset from Kaggle.

The APTOS 2019 dataset uses CSV files to store labels (0-4 severity).
Images are pre-split into train/test/val sets.
"""

import csv
import random
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DIAGNOSIS_MAP = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
DATASET_ID = "mariaherrerot/aptos2019"

# Map each CSV file to its image subdirectory and output split name
SPLIT_CONFIG = [
    ("train_1.csv", "train_images", "train"),
    ("test.csv", "test_images", "test"),
    ("valid.csv", "val_images", "val"),
]


def organise_images(source_dir):
    print(f"\nScanning {source_dir} for images...")

    # Create output directories for all splits and classes
    for _, _, output_name in SPLIT_CONFIG:
        for cls in CLASSES:
            (DATA_DIR / output_name / cls).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for csv_file, image_subdir, split_name in SPLIT_CONFIG:
        csv_path = source_dir / csv_file
        image_dir = source_dir / image_subdir / image_subdir  # nested: e.g. train_images/train_images

        if not csv_path.exists():
            print(f"  ! CSV not found: {csv_path}")
            continue
        if not image_dir.exists():
            print(f"  ! Image dir not found: {image_dir}")
            continue

        # Read CSV to get id -> diagnosis mapping
        label_map = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_map[row["id_code"]] = int(row["diagnosis"])

        # Find and copy images
        copied = 0
        missed = 0
        for img_path in sorted(image_dir.glob("*.png")):
            img_id = img_path.stem  # filename without extension
            if img_id in label_map:
                cls_name = DIAGNOSIS_MAP[label_map[img_id]]
                dest = DATA_DIR / split_name / cls_name / img_path.name
                shutil.copy2(img_path, dest)
                copied += 1
            else:
                missed += 1

        print(f"  [{split_name}] Copied {copied} images"
              + (f", {missed} unmatched" if missed else ""))
        total_copied += copied

    print(f"\n  [OK] Organised {total_copied} images")

    # Print summary
    print("\nSummary:")
    total_all = 0
    for _, _, split_name in SPLIT_CONFIG:
        split_dir = DATA_DIR / split_name
        if not split_dir.exists():
            continue
        split_total = 0
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*")))
                if n:
                    print(f"    {split_name}/{cls_dir.name}: {n} images")
                    split_total += n
        print(f"    {split_name} total: {split_total} images")
        total_all += split_total
    print(f"\n  [OK] Total: {total_all} images ready at: {DATA_DIR}")


def auto_download():
    """Try downloading via kagglehub."""
    try:
        import kagglehub
        print("\nAttempting automatic download from Kaggle...")
        path = kagglehub.dataset_download(DATASET_ID)
        print(f"  Downloaded to: {path}")
        organise_images(Path(path))
        return True
    except Exception as e:
        print(f"  Automatic download failed: {e}")
        return False


def process_zip(zip_path):
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
    print("Diabetic Retinopathy Detection Dataset Downloader")
    print("=" * 60)

    zips = list(PROJECT_DIR.glob("*.zip"))
    if zips:
        process_zip(zips[0])
        return

    folders = [d for d in PROJECT_DIR.iterdir()
               if d.is_dir() and d.name not in ["data", "static", "__pycache__", "temp_extracted"]]
    if folders:
        print(f"\nFound folder: {folders[0].name}")
        organise_images(folders[0])
        return

    if auto_download():
        return

    print("\n" + "!" * 60)
    print("  No dataset found!")
    print(f"  Download from: https://www.kaggle.com/datasets/{DATASET_ID}")
    print("  Place the ZIP in this folder and run again.")
    print("!" * 60)


if __name__ == "__main__":
    download()
