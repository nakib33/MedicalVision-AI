"""Download Tuberculosis Detection dataset from Kaggle."""
import random
import shutil
from pathlib import Path
import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Normal", "Tuberculosis"]
DATASET_ID = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"

# Map folder names to standard class names
FOLDER_MAP = {
    "normal": "Normal",
    "tuberculosis": "Tuberculosis",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def download():
    print("=" * 60)
    print("Tuberculosis Detection Dataset Downloader")
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

    # Collect all images by class (dataset has flat class folders)
    class_images = {cls: [] for cls in CLASSES}
    for file_path in downloaded_path.rglob("*"):
        if file_path.suffix.lower() not in image_ext:
            continue
        if "__MACOSX" in str(file_path):
            continue
        parent = file_path.parent.name
        for key, val in FOLDER_MAP.items():
            if key == parent.lower():
                class_images[val].append(file_path)
                break

    # Split each class into train/val/test
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
