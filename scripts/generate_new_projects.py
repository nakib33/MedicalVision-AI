"""Generate all files for projects 13-16."""
import os
import json

BASE = "projects"

PROJECTS = {
    "13_retinavision": {
        "name": "RetinaVision AI - Retinal Disease Classification",
        "classes": ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract",
                     "Age-related Macular Degeneration", "Hypertensive Retinopathy",
                     "Myopia", "Other"],
        "model": "efficientnet_b0",
        "dataset_id": "sshikamaru/retinal-fundus-multi-disease-image-dataset-rfmid",
        "color": "#667eea",
    },
    "14_gi_tract": {
        "name": "GastroVision AI - GI Tract Disease Classification",
        "classes": ["Esophagitis", "Polyps", "Ulcerative Colitis", "Dyed Lifted Polyps",
                     "Dyed Resection Margins", "Cecum", "Pylorus", "Z-Line"],
        "model": "efficientnet_b0",
        "dataset_id": "plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset",
        "color": "#38a169",
    },
    "15_oralscan": {
        "name": "OralScan AI - Oral Disease Detection",
        "classes": ["Healthy", "Dental Caries", "Gingivitis", "Oral Ulcer", "Leukoplakia"],
        "model": "efficientnet_b0",
        "dataset_id": "nourelhoda2020/oral-diseases-dataset",
        "color": "#dd6b20",
    },
    "16_hemavision": {
        "name": "HemaVision AI - Blood Cell Classification",
        "classes": ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
        "model": "resnet18",
        "dataset_id": "paultimothymooney/blood-cells",
        "color": "#d53f8c",
    },
}

TEMPLATE_LOADER = '''"""PROJECT_NAME - Model loader for inference."""

import torch
from shared.config import get_trained_model_path, DEVICE
from shared.models import create_model
from .config import PROJECT_ID, MODEL_NAME, CLASSES, IMG_SIZE

_model_instance = None


def get_model() -> torch.nn.Module:
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    model_path = get_trained_model_path(PROJECT_ID)
    model = create_model(MODEL_NAME, num_classes=len(CLASSES))
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  [OK] {PROJECT_NAME} model loaded")
    else:
        print(f"  ! No trained model at {model_path}")
    model.to(DEVICE)
    model.eval()
    _model_instance = model
    return model


def get_model_info() -> dict:
    return {
        "project_id": PROJECT_ID,
        "model": MODEL_NAME,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "trained": get_trained_model_path(PROJECT_ID).exists(),
    }
'''

TEMPLATE_DOWNLOAD = '''"""Download PROJECT_NAME dataset from Kaggle."""

import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CLASSES = CLASSES_LIST
DATASET_ID = "DS_ID"

FOLDER_MAP = FOLDER_MAP_ENTRY

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
    print("PROJECT_NAME Dataset Downloader")
    print("=" * 60)
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if not auto_download():
        print("")
        print("! Manual download: https://www.kaggle.com/datasets/" + DATASET_ID)


if __name__ == "__main__":
    download()
'''

NOTEBOOK_TEMPLATE = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PROJECT_NAME\\n",
    "\\n",
    "**Task:** Multi-class classification\\n",
    "\\n",
    "**Model:** MODEL_NAME\\n",
    "\\n",
    "**Dataset:** Kaggle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\nsys.path.insert(0, '../..')\\n\\nimport torch\\nfrom shared.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE\\nfrom shared.models import create_model\\nfrom shared.pipelines.dataset import create_dataloaders\\nfrom shared.pipelines.train import train_model\\nfrom shared.utils.metrics import compute_metrics\\n\\nprint(f'Device: {DEVICE}')\\nprint(f'PyTorch: {torch.__version__}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from config import PROJECT_ID, MODEL_NAME, CLASSES, IMG_SIZE\\nfrom shared.config import get_trained_model_path\\ncheckpoint_path = str(get_trained_model_path(PROJECT_ID))\\nprint(f'Project: {PROJECT_ID}')\\nprint(f'Model: {MODEL_NAME}')\\nprint(f'Classes: {len(CLASSES)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loader, val_loader, test_loader, ds_classes = create_dataloaders(\\ndata_root='data', class_names=CLASSES, img_size=IMG_SIZE, batch_size=BATCH_SIZE)\\nprint(f'Train: {len(train_loader.dataset)} images')\\nprint(f'Val: {len(val_loader.dataset)} images')\\nprint(f'Test: {len(test_loader.dataset)} images')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = create_model(MODEL_NAME, num_classes=len(CLASSES))\\ntotal = sum(p.numel() for p in model.parameters())\\nprint(f'Parameters: {total:,}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = train_model(model=model, train_loader=train_loader, val_loader=val_loader, device=DEVICE, num_epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE, checkpoint_path=checkpoint_path)\\nprint('Training complete!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))\\nmodel.eval()\\nall_p, all_l, all_pr = [], [], []\\nwith torch.no_grad():\\n    for images, labels in test_loader:\\n        images, labels = images.to(DEVICE), labels.to(DEVICE)\\n        outputs = model(images)\\n        probs = torch.softmax(outputs, dim=1)\\n        preds = probs.argmax(dim=1)\\n        all_p.extend(preds.cpu().numpy())\\n        all_l.extend(labels.cpu().numpy())\\n        all_pr.extend(probs.cpu().numpy())\\nm = compute_metrics(all_l, all_p, all_pr)\\nprint(f'Accuracy: {m[\"accuracy\"]:.4f}')\\nprint(f'F1-Score: {m[\"f1_score\"]:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from shared.explainers.xai_factory import run_all_explainers\\nfrom shared.pipelines.transforms import get_inference_transform\\nfrom PIL import Image\\nimport os, random\\ntest_dir = 'data/test'\\nif os.path.isdir(test_dir):\\n    cls_dir = random.choice([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])\\n    img_name = random.choice(os.listdir(os.path.join(test_dir, cls_dir)))\\n    img_path = os.path.join(test_dir, cls_dir, img_name)\\n    print(f'Sample: {img_path}')\\n    pil = Image.open(img_path).convert('RGB')\\n    transform = get_inference_transform(IMG_SIZE)\\n    tensor = transform(pil).unsqueeze(0).to(DEVICE)\\n    result = run_all_explainers(model, tensor, CLASSES, device=DEVICE)\\n    print(f'Prediction: {result[\"predictions\"][\"predicted_class\"]}')\\n    print(f'Confidence: {result[\"predictions\"][\"confidence\"]:.2%}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''


CSS_TEMPLATE = '''/* PROJECT_NAME - Project-Specific Styles */

.upload-zone:hover,
.upload-zone.drag-over {
    border-color: COLOR;
    background: rgba(RGB_R, RGB_G, RGB_B, 0.05);
}
'''

JS_TEMPLATE = '''/**
 * PROJECT_NAME - Project-Specific Script
 */
document.addEventListener('DOMContentLoaded', () => {
    const h1 = document.querySelector('h1');
    if (h1) {
        h1.style.borderLeft = '4px solid COLOR';
        h1.style.paddingLeft = '1rem';
    }
});
'''


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


for pid, info in PROJECTS.items():
    proj_dir = os.path.join(BASE, pid)
    static_dir = os.path.join(proj_dir, "static")
    classes_json = json.dumps(info["classes"])
    rgb = hex_to_rgb(info["color"])

    # model_loader.py
    loader = TEMPLATE_LOADER.replace("PROJECT_NAME", info["name"])
    with open(os.path.join(proj_dir, "model_loader.py"), "w") as f:
        f.write(loader)

    # download_data.py
    folder_map = "{\n    " + ",\n    ".join(f'"{c.lower()}": "{c}"' for c in info["classes"]) + ",\n}"
    dl = TEMPLATE_DOWNLOAD
    dl = dl.replace("PROJECT_NAME", info["name"])
    dl = dl.replace("CLASSES_LIST", classes_json)
    dl = dl.replace("DS_ID", info["dataset_id"])
    dl = dl.replace("FOLDER_MAP_ENTRY", folder_map)
    with open(os.path.join(proj_dir, "download_data.py"), "w") as f:
        f.write(dl)

    # model.ipynb
    nb = NOTEBOOK_TEMPLATE
    nb = nb.replace("PROJECT_NAME", info["name"])
    nb = nb.replace("MODEL_NAME", info["model"])
    with open(os.path.join(proj_dir, "model.ipynb"), "w") as f:
        f.write(nb)

    # static/style.css
    css = CSS_TEMPLATE
    css = css.replace("PROJECT_NAME", info["name"])
    css = css.replace("COLOR", info["color"])
    css = css.replace("RGB_R", str(rgb[0])).replace("RGB_G", str(rgb[1])).replace("RGB_B", str(rgb[2]))
    with open(os.path.join(static_dir, "style.css"), "w") as f:
        f.write(css)

    # static/script.js
    js = JS_TEMPLATE
    js = js.replace("PROJECT_NAME", info["name"])
    js = js.replace("COLOR", info["color"])
    with open(os.path.join(static_dir, "script.js"), "w") as f:
        f.write(js)

    print(f"  OK: {pid}")

print(f"\nAll 4 projects created!")
