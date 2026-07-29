"""Download Lung Cancer CT Scan dataset from Kaggle (dishantrathi20).

Dataset: https://www.kaggle.com/datasets/dishantrathi20/ct-scan-images-for-lung-cancer

The dataset comes with pre-organised train/valid/test splits but class names
are inconsistent across splits and squamous cell carcinoma lacks training data.
This script consolidates ALL images, re-splits them evenly (70/15/15), and
standardises class folder names to match config.py.
"""

import random
import shutil
from pathlib import Path

import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Standard target classes (must match config.py CLASSES)
TARGET_CLASSES = [
    "Benign",
    "Malignant",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
]

# Maps any substring found in the dataset folder name → standard class name
FOLDER_MAP = {
    "benign": "Benign",
    "bengin": "Benign",              # original Kaggle typo
    "malignant": "Malignant",
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large Cell Carcinoma",
    "largecellcarcinoma": "Large Cell Carcinoma",
    "normal": "Normal",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
    "squamouscellcarcinoma": "Squamous Cell Carcinoma",
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def organise_images(source_dir: Path):
    """Scan source_dir for images, map to standard classes, and re-split."""
    print(f"\nScanning {source_dir} for images...")

    # ── Collect all images by class ──────────────────────────────────
    class_images: dict[str, list[Path]] = {c: [] for c in TARGET_CLASSES}
    skipped = 0

    for file_path in source_dir.rglob("*"):
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "__MACOSX" in str(file_path):
            continue

        parent = file_path.parent.name.lower()
        matched = False
        for key, std_class in FOLDER_MAP.items():
            if key in parent:
                class_images[std_class].append(file_path)
                matched = True
                break
        if not matched:
            skipped += 1

    total = sum(len(v) for v in class_images.values())
    if total == 0:
        print("  ! No images found! Debugging folder structure:")
        for f in sorted(source_dir.rglob("*")):
            if f.is_dir() and f.parent != source_dir:
                depth = len(f.relative_to(source_dir).parts)
                if depth <= 2:
                    print(f"    Folder: {f.relative_to(source_dir)}")
        return

    print(f"  Found {total} images ({skipped} skipped — no class match)")

    # ── Report per-class counts before split ─────────────────────────
    print("\nPer-class totals:")
    for cls_name in TARGET_CLASSES:
        print(f"    {cls_name}: {len(class_images[cls_name])}")

    # ── Create target directories ────────────────────────────────────
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # ── Shuffle + split per class ────────────────────────────────────
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
                # Handle duplicate filenames
                if dest.exists():
                    stem = src_path.stem
                    dest = dest.with_name(f"{stem}_{random.randint(1000,9999)}{src_path.suffix}")
                shutil.copy2(src_path, dest)

    # ── Print summary ────────────────────────────────────────────────
    print("\nFinal dataset summary:")
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
    print(f"\n  [OK] Total: {total_all} images ready for training at: {DATA_DIR}")
    print(f"  Classes ({len(TARGET_CLASSES)}): {', '.join(TARGET_CLASSES)}")


def download():
    """Download dataset from Kaggle, then organise."""
    print("=" * 60)
    print("Lung Cancer CT Scan Dataset Downloader")
    print("=" * 60)

    print("\n[1/2] Downloading dataset from Kaggle...")
    print(f"  Dataset: {TARGET_CLASSES}")
    dataset_path = kagglehub.dataset_download(
        "dishantrathi20/ct-scan-images-for-lung-cancer"
    )
    print(f"  [OK] Downloaded to: {dataset_path}")

    print("\n[2/2] Organising into train/val/test with standardised class names...")
    organise_images(Path(dataset_path))

    print("\n  Done! Run the model.ipynb notebook to train.")


if __name__ == "__main__":
    download()
