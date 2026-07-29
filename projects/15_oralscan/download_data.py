"""Download OralScan AI - Oral Disease Detection dataset from Kaggle."""

import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Healthy", "Dental Caries", "Gingivitis", "Oral Ulcer", "Leukoplakia"]
DATASET_ID = "nourelhoda2020/oral-diseases-dataset"

FOLDER_MAP = {
    "healthy": "Healthy",
    "dental caries": "Dental Caries",
    "gingivitis": "Gingivitis",
    "oral ulcer": "Oral Ulcer",
    "leukoplakia": "Leukoplakia",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def organise_images(source_dir):
    print("")
    print("Scanning " + str(source_dir) + " for images...")
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
                    print("    Folder: " + str(f.relative_to(source_dir)))
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
    print("")
    print("Summary:")
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
    print("")
    print("  [OK] Total: " + str(total_all) + " images ready at: " + str(DATA_DIR))


def auto_download():
    try:
        import kagglehub
        print("")
        print("Downloading from Kaggle...")
        path = kagglehub.dataset_download(DATASET_ID)
        print("  Downloaded to: " + str(path))
        organise_images(Path(path))
        return True
    except Exception as e:
        print("  Download failed: " + str(e))
        return False


def download():
    print("=" * 60)
    print("OralScan AI - Oral Disease Detection Dataset Downloader")
    print("=" * 60)
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if not auto_download():
        print("")
        print("! Manual download: https://www.kaggle.com/datasets/" + DATASET_ID)


if __name__ == "__main__":
    download()
