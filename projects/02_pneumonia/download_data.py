"""Download Pneumonia Detection dataset from Kaggle."""
import os
import shutil
from pathlib import Path
import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Normal", "Pneumonia"]
DATASET_ID = "paultimothymooney/chest-xray-pneumonia"

# Map folder names to standard class names
FOLDER_MAP = {
    "normal": "Normal",
    "pneumonia": "Pneumonia",
}


def download():
    print("=" * 60)
    print("Pneumonia Detection Dataset Downloader")
    print("=" * 60)

    print("\n[1/3] Downloading from Kaggle...")
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    print(f"  [OK] Downloaded to: {dataset_path}")

    print("\n[2/3] Organising into train/val/test...")

    # Create target directories
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(dataset_path)
    image_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    file_count = 0

    # The dataset structure is:
    #   chest_xray/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpg
    # We rglob and detect splits from the path
    for file_path in downloaded_path.rglob("*"):
        if file_path.suffix.lower() not in image_ext:
            continue
        if "__MACOSX" in str(file_path):
            continue

        path_str = str(file_path)

        # Detect split from path (handle both / and \ separators)
        if "\\train\\" in path_str or "/train/" in path_str:
            split = "train"
        elif "\\val\\" in path_str or "/val/" in path_str:
            split = "val"
        elif "\\test\\" in path_str or "/test/" in path_str:
            split = "test"
        else:
            split = "train"

        # Detect class from parent folder name
        parent = file_path.parent.name
        target = FOLDER_MAP.get(parent.lower())

        if target is None:
            continue

        shutil.copy2(file_path, DATA_DIR / split / target / file_path.name)
        file_count += 1

    print(f"  [OK] Organised {file_count} images")

    # Summary
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
