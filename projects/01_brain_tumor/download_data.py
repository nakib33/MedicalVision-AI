"""Download Brain Tumor MRI dataset from Kaggle."""
import os
import shutil
import random
from pathlib import Path
import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Target class mapping from folder names to standard labels
CLASS_MAP = {
    "glioma_tumor": "Glioma",
    "meningioma_tumor": "Meningioma",
    "pituitary_tumor": "Pituitary",
    "no_tumor": "Normal",
}

TARGET_CLASSES = ["Glioma", "Meningioma", "Pituitary", "Normal"]

# Split ratios: 70% train, 15% val, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def download():
    print("=" * 60)
    print("Brain Tumor MRI Dataset Downloader")
    print("=" * 60)

    print("\n[1/3] Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(
        "sartajbhuvaji/brain-tumor-classification-mri"
    )
    print(f"  [OK] Downloaded to: {dataset_path}")

    print("\n[2/3] Organising into train/val/test structure...")

    # Create target directories
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # The Kaggle dataset has: Training/ and Testing/ folders
    # Each has: glioma_tumor/, meningioma_tumor/, no_tumor/, pituitary_tumor/
    downloaded_path = Path(dataset_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    # ─── Step 1: Copy all images with correct class names ───
    all_images = []  # (source_path, target_class_name)

    for split_source in ["Training", "Testing"]:
        split_path = downloaded_path / split_source
        if not split_path.exists():
            print(f"  ! Folder {split_source} not found, skipping")
            continue

        for folder_name, class_name in CLASS_MAP.items():
            class_path = split_path / folder_name
            if not class_path.exists():
                print(f"  ! {split_source}/{folder_name} not found, skipping")
                continue

            for img_file in class_path.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    all_images.append((img_file, class_name))

    random.shuffle(all_images)
    total = len(all_images)
    print(f"  Found {total} images total")

    # ─── Step 2: Split into train/val/test ───
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = [
        ("train", all_images[:train_end]),
        ("val", all_images[train_end:val_end]),
        ("test", all_images[val_end:]),
    ]

    # ─── Step 3: Copy files ───
    for split_name, image_list in splits:
        for src_path, class_name in image_list:
            dest = DATA_DIR / split_name / class_name / src_path.name
            # Handle duplicate filenames
            if dest.exists():
                dest = DATA_DIR / split_name / class_name / f"{src_path.stem}_{random.randint(1000,9999)}{src_path.suffix}"
            shutil.copy2(src_path, dest)

    # ─── Step 4: Summary ───
    print("\n[3/3] Summary:")
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
    print(f"  Classes: {', '.join(TARGET_CLASSES)}")


if __name__ == "__main__":
    download()
