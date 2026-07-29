"""Download Malaria Cell Classification dataset from Kaggle."""

import random
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Parasitized", "Uninfected"]
DATASET_ID = "iarunava/cell-images-for-detecting-malaria"

FOLDER_MAP = {
    "parasitized": "Parasitized",
    "uninfected": "Uninfected",
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


def process_zip(zip_path):
    extract_dir = PROJECT_DIR / "temp_extracted"

    # Skip extraction if already done
    if extract_dir.exists() and len(list(extract_dir.rglob("*.jpg"))) > 0:
        print("")
        print("Already extracted, using existing temp folder.")
        organise_images(extract_dir)
        return

    print("")
    print("Extracting ZIP (this may take a few minutes)...")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Extraction complete.")
    organise_images(extract_dir)
    shutil.rmtree(extract_dir)


def auto_download():
    try:
        import kagglehub
        print("")
        print("Attempting automatic download from Kaggle...")
        path = kagglehub.dataset_download(DATASET_ID)
        print("  Downloaded to: " + str(path))
        organise_images(Path(path))
        return True
    except Exception as e:
        print("  Automatic download failed: " + str(e))
        return False


def download():
    print("=" * 60)
    print("Malaria Cell Classification Dataset Downloader")
    print("=" * 60)

    # First check if data already exists
    if (DATA_DIR / "train" / "Parasitized").exists():
        count = len(list((DATA_DIR / "train" / "Parasitized").glob("*.jpg")))
        if count > 0:
            print("")
            print("Data already organised in: " + str(DATA_DIR))
            print("Total: " + str(count) + " images in train/Parasitized")
            print("To re-download, delete the data folder and run again.")
            return

    zips = list(PROJECT_DIR.glob("*.zip"))
    if zips:
        process_zip(zips[0])
        return

    folders = [d for d in PROJECT_DIR.iterdir()
               if d.is_dir() and d.name not in ["data", "static", "__pycache__", "temp_extracted"]]
    if folders:
        print("")
        print("Found folder: " + folders[0].name)
        organise_images(folders[0])
        return

    if auto_download():
        return

    print("")
    print("!" * 60)
    print("  No dataset found!")
    print("  Download from: https://www.kaggle.com/datasets/" + DATASET_ID)
    print("  Place the ZIP in this folder and run again.")
    print("!" * 60)


if __name__ == "__main__":
    download()
