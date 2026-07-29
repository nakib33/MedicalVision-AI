"""Download COVID-19 Chest X-ray dataset from Kaggle."""
import random
import shutil
from pathlib import Path
import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["COVID", "Viral Pneumonia", "Normal"]
DATASET_ID = "pranavraikokte/covid19-image-dataset"

# Map actual folder names to standard class names
FOLDER_MAP = {
    "covid": "COVID",
    "normal": "Normal",
    "viral pneumonia": "Viral Pneumonia",
}

VAL_RATIO = 0.20  # Split 80/20 from train folder


def download():
    print("=" * 60)
    print("COVID-19 Chest X-ray Dataset Downloader")
    print("=" * 60)

    print("\n[1/3] Downloading from Kaggle...")
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    print(f"  [OK] Downloaded to: {dataset_path}")

    print("\n[2/3] Organising into train/val/test...")

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(dataset_path)
    image_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    train_images = []
    test_images = []

    for file_path in downloaded_path.rglob("*"):
        if file_path.suffix.lower() not in image_ext:
            continue
        if "__MACOSX" in str(file_path):
            continue

        path_str = str(file_path)
        parent = file_path.parent.name

        target = None
        for key, val in FOLDER_MAP.items():
            if key in parent.lower():
                target = val
                break
        if target is None:
            continue

        if "\\test\\" in path_str or "/test/" in path_str:
            test_images.append((file_path, target))
        elif "\\train\\" in path_str or "/train/" in path_str:
            train_images.append((file_path, target))

    # Split train into train/val
    random.shuffle(train_images)
    val_count = int(len(train_images) * VAL_RATIO)
    val_images = train_images[:val_count]
    train_images = train_images[val_count:]

    for split_name, image_list in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        for src_path, class_name in image_list:
            shutil.copy2(src_path, DATA_DIR / split_name / class_name / src_path.name)

    total = len(train_images) + len(val_images) + len(test_images)
    print(f"  [OK] Organised {total} images")
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


if __name__ == "__main__":
    download()
