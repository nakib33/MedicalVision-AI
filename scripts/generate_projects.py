#!/usr/bin/env python3
"""
Generate all 12 projects from the Brain Tumor template.
Run from project root: python scripts/generate_projects.py
"""

import json
from pathlib import Path

PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projects"
SCRIPTS_DIR = Path(__file__).resolve().parent


PROJECTS = [
    {
        "id": "02_pneumonia",
        "name": "Pneumonia Detection",
        "short_name": "Pneumonia",
        "desc": "Binary classification of chest X-rays - Normal vs Pneumonia",
        "classes": ["Normal", "Pneumonia"],
        "model": "densenet121",
        "dataset_id": "paultimothymooney/chest-xray-pneumonia",
        "color": "#e53e3e",
    },
    {
        "id": "03_covid19",
        "name": "COVID-19 Chest X-ray Classification",
        "short_name": "COVID-19",
        "desc": "Multi-class classification - COVID, Viral Pneumonia, Normal",
        "classes": ["COVID", "Viral Pneumonia", "Normal"],
        "model": "efficientnet_b0",
        "dataset_id": "pranavraikokte/covid19-image-dataset",
        "color": "#3182ce",
    },
    {
        "id": "04_tuberculosis",
        "name": "Tuberculosis Detection",
        "short_name": "TB Detection",
        "desc": "Binary classification - Tuberculosis vs Normal from chest X-rays",
        "classes": ["Normal", "Tuberculosis"],
        "model": "resnet50",
        "dataset_id": "tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
        "color": "#dd6b20",
    },
    {
        "id": "05_alzheimers",
        "name": "Alzheimer MRI Classification",
        "short_name": "Alzheimer",
        "desc": "Multi-class - Non Demented, Very Mild, Mild, Moderate",
        "classes": ["Non Demented", "Very Mild", "Mild", "Moderate"],
        "model": "densenet121",
        "dataset_id": "tourist55/alzheimers-dataset-4-class-of-images",
        "color": "#805ad5",
    },
    {
        "id": "06_skin_cancer",
        "name": "Skin Cancer Classification",
        "short_name": "Skin Cancer",
        "desc": "Binary - Benign vs Malignant skin lesions (HAM10000)",
        "classes": ["Benign", "Malignant"],
        "model": "efficientnet_b0",
        "dataset_id": "kmader/skin-cancer-mnist-ham10000",
        "color": "#d53f8c",
    },
    {
        "id": "07_diabetic_retinopathy",
        "name": "Diabetic Retinopathy Detection",
        "short_name": "Retinopathy",
        "desc": "Multi-class - 5 stages (APTOS 2019)",
        "classes": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        "model": "efficientnet_b0",
        "dataset_id": "mariaherrerot/aptos2019",
        "color": "#319795",
    },
    {
        "id": "08_malaria",
        "name": "Malaria Cell Classification",
        "short_name": "Malaria",
        "desc": "Binary - Parasitized vs Uninfected blood cells",
        "classes": ["Parasitized", "Uninfected"],
        "model": "resnet18",
        "dataset_id": "iarunava/cell-images-for-detecting-malaria",
        "color": "#38a169",
    },
    {
        "id": "09_breast_cancer",
        "name": "Breast Cancer Histopathology",
        "short_name": "Breast Cancer",
        "desc": "Binary - IDC Positive vs IDC Negative histopathology",
        "classes": ["IDC Negative", "IDC Positive"],
        "model": "densenet121",
        "dataset_id": "paultimothymooney/breast-histopathology-images",
        "color": "#e53e3e",
    },
    {
        "id": "10_bone_fracture",
        "name": "Bone Fracture Detection",
        "short_name": "Bone Fracture",
        "desc": "Binary - Fractured vs Normal bone X-rays",
        "classes": ["Normal", "Fractured"],
        "model": "efficientnet_b0",
        "dataset_id": "pkdarabi/bone-fracture-detection-computer-vision-project",
        "color": "#2b6cb0",
    },
    {
        "id": "11_lung_cancer",
        "name": "Lung Cancer CT Scan Classification",
        "short_name": "Lung Cancer",
        "desc": "Multi-class - 4 CT scan types",
        "classes": ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"],
        "model": "efficientnet_b0",
        "dataset_id": "andrewmvd/lung-cancer-subtype-classification",
        "color": "#6b46c1",
    },
]


def make_cells(proj):
    """Create notebook cells for a project."""
    ds_id = proj["dataset_id"]
    model = proj["model"]
    classes = proj["classes"]
    classes_json = json.dumps(classes)

    # Cell source code templates (using list of strings to avoid quote issues)
    c1_md = [
        f"# {proj['name']}",
        "",
        f"**Task:** {proj['desc']}",
        "",
        f"**Model:** {model}",
        "",
        f"**Dataset:** [{ds_id}](https://www.kaggle.com/datasets/{ds_id})",
    ]

    c2_setup = [
        "import sys",
        "sys.path.insert(0, '../..')",
        "",
        "import torch",
        "from shared.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE",
        "from shared.models import create_model",
        "from shared.pipelines.dataset import create_dataloaders",
        "from shared.pipelines.train import train_model",
        "",
        "print(f'Device: {DEVICE}')",
        "print(f'PyTorch: {torch.__version__}')",
    ]

    c3_config = [
        "from config import PROJECT_ID, MODEL_NAME, CLASSES, IMG_SIZE",
        "from shared.config import get_trained_model_path",
        "",
        "checkpoint_path = str(get_trained_model_path(PROJECT_ID))",
        f"print(f'Project: {{PROJECT_ID}}')",
        f"print(f'Model: {{MODEL_NAME}}')",
        f"print(f'Classes: {{CLASSES}}')",
    ]

    c4_data = [
        "train_loader, val_loader, test_loader, ds_classes = create_dataloaders(",
        "    data_root='data',",
        "    class_names=CLASSES,",
        "    img_size=IMG_SIZE,",
        "    batch_size=BATCH_SIZE,",
        ")",
        "print(f'Train: {len(train_loader.dataset)} images')",
        "print(f'Val: {len(val_loader.dataset)} images')",
        "print(f'Test: {len(test_loader.dataset)} images')",
    ]

    c5_model = [
        "model = create_model(MODEL_NAME, num_classes=len(CLASSES))",
        "total = sum(p.numel() for p in model.parameters())",
        f"print(f'Parameters: {{total:,}}')",
    ]

    c6_train = [
        "history = train_model(",
        "    model=model,",
        "    train_loader=train_loader,",
        "    val_loader=val_loader,",
        "    device=DEVICE,",
        "    num_epochs=EPOCHS,",
        "    lr=LEARNING_RATE,",
        "    patience=EARLY_STOPPING_PATIENCE,",
        "    checkpoint_path=checkpoint_path,",
        ")",
        "print('Training complete!')",
    ]

    c7_eval = [
        "from shared.utils.metrics import compute_metrics",
        "",
        "model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))",
        "model.eval()",
        "",
        "all_preds, all_labels, all_probs = [], [], []",
        "with torch.no_grad():",
        "    for images, labels in test_loader:",
        "        images, labels = images.to(DEVICE), labels.to(DEVICE)",
        "        outputs = model(images)",
        "        probs = torch.softmax(outputs, dim=1)",
        "        preds = probs.argmax(dim=1)",
        "        all_preds.extend(preds.cpu().numpy())",
        "        all_labels.extend(labels.cpu().numpy())",
        "        all_probs.extend(probs.cpu().numpy())",
        "",
        "m = compute_metrics(all_labels, all_preds, all_probs)",
        "print(f'Accuracy:  {m[\"accuracy\"]:.4f}')",
        "print(f'Precision: {m[\"precision\"]:.4f}')",
        "print(f'Recall:    {m[\"recall\"]:.4f}')",
        "print(f'F1-Score:  {m[\"f1_score\"]:.4f}')",
        "if m.get('roc_auc'):",
        "    print(f'ROC-AUC:   {m[\"roc_auc\"]:.4f}')",
    ]

    c8_xai = [
        "from shared.explainers.xai_factory import run_all_explainers",
        "from shared.pipelines.transforms import get_inference_transform",
        "from PIL import Image",
        "import os, random",
        "",
        "test_dir = 'data/test'",
        "if os.path.isdir(test_dir):",
        "    class_dir = random.choice(os.listdir(test_dir))",
        "    img_name = random.choice(os.listdir(os.path.join(test_dir, class_dir)))",
        "    img_path = os.path.join(test_dir, class_dir, img_name)",
        "    print(f'Sample: {img_path}')",
        "",
        "    pil = Image.open(img_path).convert('RGB')",
        "    transform = get_inference_transform(IMG_SIZE)",
        "    tensor = transform(pil).unsqueeze(0).to(DEVICE)",
        "",
        "    result = run_all_explainers(model, tensor, CLASSES, device=DEVICE)",
        "    print(f'Prediction: {result[\"predictions\"][\"predicted_class\"]}')",
        "    print(f'Confidence: {result[\"predictions\"][\"confidence\"]:.2%}')",
        "",
        "    from IPython.display import display, HTML",
        "    html = '<div style=\"display:grid;grid-template-columns:repeat(2,1fr);gap:10px;\">'",
        "    for key, exp in result['explanations'].items():",
        "        if 'overlay_base64' in exp:",
        "            html += f'<div style=\"padding:10px;border:1px solid #ddd;\">'",
        "            html += f'<h4>{exp[\"label\"]}</h4>'",
        "            html += f'<img src=\"data:image/png;base64,{exp[\"overlay_base64\"]}\" style=\"width:100%;\"/>'",
        "            html += '</div>'",
        "    display(HTML(html + '</div>'))",
    ]

    return [
        {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in c1_md]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c2_setup]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c3_config]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c4_data]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c5_model]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c6_train]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c7_eval]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in c8_xai]},
    ]


def generate_project(proj):
    """Generate all files for one project."""
    proj_dir = PROJECTS_DIR / proj["id"]
    static_dir = proj_dir / "static"
    proj_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    classes_repr = json.dumps(proj["classes"])
    color = proj["color"]

    # ── config.py ────────────────────────────────────────────
    with open(proj_dir / "config.py", "w") as f:
        f.write(f'"""' + proj["name"] + ' - Project Configuration."""\n\n')
        f.write(f'PROJECT_ID = "{proj["id"]}"\n')
        f.write(f'PROJECT_NAME = "{proj["name"]}"\n')
        f.write(f'SHORT_NAME = "{proj["short_name"]}"\n')
        f.write(f'DESCRIPTION = "{proj["desc"]}"\n\n')
        f.write(f"CLASSES = {classes_repr}\n")
        f.write(f"IMG_SIZE = 224\n")
        f.write(f'MODEL_NAME = "{proj["model"]}"\n\n')
        f.write(f'DATASET_URL = "https://www.kaggle.com/datasets/{proj["dataset_id"]}"\n')
        f.write(f'KAGGLE_DATASET = "{proj["dataset_id"]}"\n\n')
        f.write("CLASS_COLORS = {\n")
        for c in proj["classes"]:
            f.write(f'    "{c}": "{color}",\n')
        f.write("}\n")

    # ── download_data.py ────────────────────────────────────
    with open(proj_dir / "download_data.py", "w") as f:
        f.write(f'"""Download {proj["name"]} dataset from Kaggle."""\n\n')
        f.write("import shutil\nfrom pathlib import Path\nimport kagglehub\n\n")
        f.write("PROJECT_DIR = Path(__file__).resolve().parent\n")
        f.write('DATA_DIR = PROJECT_DIR / "data"\n')
        f.write(f"CLASSES = {classes_repr}\n")
        f.write(f'DATASET_ID = "{proj["dataset_id"]}"\n\n')
        f.write("""
def download():
    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)
    print("\\n[1/3] Downloading from Kaggle...")
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    print(f"  Downloaded to: {dataset_path}")
    print("\\n[2/3] Organising into train/val/test...")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(dataset_path)
    image_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    file_count = 0
    for file_path in downloaded_path.rglob("*"):
        if file_path.suffix.lower() not in image_ext:
            continue
        parent = file_path.parent.name.lower()
        path_str = str(file_path).lower()
        if "/train" in path_str: split = "train"
        elif "/val" in path_str or "/valid" in path_str: split = "val"
        elif "/test" in path_str: split = "test"
        else: split = "train"
        target = None
        for cls in CLASSES:
            if cls.lower() in parent: target = cls; break
        if target is None: continue
        shutil.copy2(file_path, DATA_DIR / split / target / file_path.name)
        file_count += 1
    print(f"  Organised {file_count} images")
    print("\\n[3/3] Summary:")
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    n = len(list(cls_dir.glob("*")))
                    print(f"    {split}/{cls_dir.name}: {n} images")
    print(f"\\nDataset ready at: {DATA_DIR}")

if __name__ == "__main__":
    download()
""")

    # ── model_loader.py ─────────────────────────────────────
    with open(proj_dir / "model_loader.py", "w") as f:
        f.write(f'"""' + proj["name"] + ' - Model loader for inference."""\n\n')
        f.write("""
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
        print(f"  Model loaded from {model_path.name}")
    else:
        print(f"  No trained model at {model_path}")
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
""")

    # ── model.ipynb ─────────────────────────────────────────
    notebook = {
        "cells": make_cells(proj),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(proj_dir / "model.ipynb", "w") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    # ── static/style.css ────────────────────────────────────
    rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    with open(static_dir / "style.css", "w") as f:
        f.write(f'/* {proj["name"]} - Project-Specific Styles */\n\n')
        f.write(f'.upload-zone:hover,\n.upload-zone.drag-over {{\n')
        f.write(f'    border-color: {color};\n')
        f.write(f'    background: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.05);\n')
        f.write('}\n')

    # ── static/script.js ────────────────────────────────────
    with open(static_dir / "script.js", "w") as f:
        f.write(f'/**\n * {proj["name"]} - Project-Specific Script\n */\n')
        f.write("document.addEventListener('DOMContentLoaded', () => {\n")
        f.write("    const h1 = document.querySelector('h1');\n")
        f.write("    if (h1) {\n")
        f.write(f"        h1.style.borderLeft = '4px solid {color}';\n")
        f.write("        h1.style.paddingLeft = '1rem';\n")
        f.write("    }\n")
        f.write('});\n')

    print(f"  OK  {proj['id']}: {proj['name']}")


def main():
    print("=" * 60)
    print("MedicalVision AI Suite - Project Generator")
    print("=" * 60)
    print(f"\nGenerating {len(PROJECTS)} projects...\n")
    for proj in PROJECTS:
        generate_project(proj)
    print(f"\nDone! {len(PROJECTS)} projects generated.")
    print("Now configure projects/__init__.py and main.py")


if __name__ == "__main__":
    main()
