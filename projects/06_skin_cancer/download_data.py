"""Download Skin Cancer Classification dataset from Kaggle (HAM10000)."""

import random
import shutil
import zipfile
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Benign", "Malignant"]
DATASET_ID = "kmader/skin-cancer-mnist-ham10000"

# Map diagnosis codes to Benign/Malignant
# nv=nevus(benign), bkl=benign keratosis, vasc=vascular, df=dermatofibroma
# mel=melanoma, bcc=basal cell carcinoma, akiec=actinic keratosis
DX_MAP = {
    "nv": "Benign",
    "bkl": "Benign",
    "vasc": "Benign",
    "df": "Benign",
    "mel": "Malignant",
    "bcc": "Malignant",
    "akiec": "Malignant",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def organise_images(source_dir):
    print(f"\nScanning {source_dir} for images...")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Find the metadata CSV
    csv_path = list(Path(source_dir).rglob("HAM10000_metadata.csv"))
    if not csv_path:
        print("  ! HAM10000_metadata.csv not found!")
        return
    df = pd.read_csv(csv_path[0])
    print(f"  Found metadata: {len(df)} images")

    # Find image files in part folders
    image_ext = {".jpg", ".jpeg", ".png"}
    image_files = {}
    for f in Path(source_dir).rglob("*"):
        if f.suffix.lower() in image_ext and "HAM10000_images" in str(f):
            image_files[f.stem] = f  # key = image_id without extension

    print(f"  Found {len(image_files)} image files on disk")

    # Match images to diagnosis and organize
    class_images = {cls: [] for cls in CLASSES}
    matched = 0
    for _, row in df.iterrows():
        img_id = row["image_id"]
        dx = row["dx"]
        if dx not in DX_MAP:
            continue
        if img_id in image_files:
            class_images[DX_MAP[dx]].append(image_files[img_id])
            matched += 1

    print(f"  Matched {matched} images to diagnosis")

    total = sum(len(v) for v in class_images.values())
    if total == 0:
        print("  ! No images matched! Check folder paths.")
        return

    # Split into train/val/test
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


def auto_download():
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
    print("Skin Cancer Classification Dataset Downloader (HAM10000)")
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
