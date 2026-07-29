"""Download Bone Fracture Detection dataset from Kaggle."""

import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = ["Normal", "Fractured"]
DATASET_ID = "pkdarabi/bone-fracture-detection-computer-vision-project"


def organise_from_yolo(source_dir):
    """Organise images from a YOLO-format dataset where labels are .txt files.

    An image is "Fractured" if its label file has content (bounding-box annotations),
    and "Normal" if the label file is empty. Respects existing train/valid/test splits.
    """
    print(f"\nScanning {source_dir} for YOLO-format data...")
    for split_out in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split_out / cls).mkdir(parents=True, exist_ok=True)

    image_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    split_map = {"train": "train", "valid": "val", "test": "test"}
    total_organised = 0

    for yolo_split_in, yolo_split_out in split_map.items():
        img_dir = source_dir / yolo_split_in / "images"
        lbl_dir = source_dir / yolo_split_in / "labels"
        if not img_dir.is_dir():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in image_ext:
                continue

            # Corresponding label file
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists() and lbl_path.read_text().strip():
                cls_name = "Fractured"
            else:
                cls_name = "Normal"

            dst = DATA_DIR / yolo_split_out / cls_name / img_path.name
            shutil.copy2(img_path, dst)
            total_organised += 1

    print(f"\nSummary:")
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
        # For Kaggle downloads, the structure is unknown; fall back to YOLO detection
        yolo_dir = _find_yolo_dir(Path(path))
        if yolo_dir == Path(path) and not any(
            (Path(path) / s / "images").is_dir() for s in ["train", "valid", "test"]
        ):
            print("  Downloaded data doesn't match YOLO format — skipping.")
            return False
        organise_from_yolo(yolo_dir)
        return True
    except Exception as e:
        print(f"  Automatic download failed: {e}")
        return False


def _find_yolo_dir(extract_dir):
    """Find the YOLO-format subdirectory inside a zip extract."""
    # Look for a child folder that has train/valid/test with images/ and labels/
    for child in sorted(extract_dir.iterdir()):
        if child.is_dir():
            for split in ["train", "valid", "test"]:
                if (child / split / "images").is_dir() and (child / split / "labels").is_dir():
                    return child
    return extract_dir


def process_zip(zip_path):
    print(f"\nFound ZIP: {zip_path.name}")
    extract_dir = PROJECT_DIR / "temp_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    yolo_dir = _find_yolo_dir(extract_dir)
    print(f"  YOLO dataset root: {yolo_dir.relative_to(PROJECT_DIR)}")
    organise_from_yolo(yolo_dir)
    shutil.rmtree(extract_dir)


def download():
    print("=" * 60)
    print("Bone Fracture Detection")
    print("=" * 60)

    # 1) Try a local ZIP archive
    zips = list(PROJECT_DIR.glob("*.zip"))
    if zips:
        process_zip(zips[0])
        return

    # 2) Try a pre-extracted YOLO folder
    for candidate in PROJECT_DIR.iterdir():
        if candidate.is_dir() and candidate.name not in ["data", "static", "__pycache__", "temp_extracted"]:
            # Check if it looks like a YOLO root
            if candidate.name == "BoneFractureYolo8" or any(
                (candidate / s / "images").is_dir() for s in ["train", "valid", "test"]
            ):
                print(f"\nFound YOLO folder: {candidate.name}")
                organise_from_yolo(candidate)
                return

    # 3) Try auto-download from Kaggle
    if auto_download():
        return

    print("\n" + "!" * 60)
    print("  No dataset found!")
    print(f"  Download from: https://www.kaggle.com/datasets/{DATASET_ID}")
    print("  Place the ZIP in this folder and run again.")
    print("!" * 60)


if __name__ == "__main__":
    download()
