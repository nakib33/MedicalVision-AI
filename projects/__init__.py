"""MedicalVision AI Suite — Project Discovery & Registry.

Auto-discovers all 16 projects and provides a registry for the main app.
Each project exposes: config, model_loader, and static routes.
"""

import importlib
import json
from pathlib import Path
from typing import List, Dict, Any

PROJECTS_DIR = Path(__file__).resolve().parent

# Ordered list of project IDs (matching folder names)
PROJECT_ORDER = [
    "01_brain_tumor",
    "02_pneumonia",
    "03_covid19",
    "04_tuberculosis",
    "05_alzheimers",
    "06_skin_cancer",
    "07_diabetic_retinopathy",
    "08_malaria",
    "09_breast_cancer",
    "10_bone_fracture",
    "11_lung_cancer",
    "12_medscan_ai",
    "13_retinavision",
    "14_gi_tract",
    "15_oralscan",
    "16_hemavision",
]


_discovered = None


def discover_projects() -> List[Dict[str, Any]]:
    """Discover all projects and return their metadata.

    Returns a list of dicts:
        {
            "id": "01_brain_tumor",
            "name": "Brain Tumor MRI Classification",
            "short_name": "Brain Tumor",
            "description": "...",
            "classes": ["Glioma", ...],
            "model": "efficientnet_b0",
            "img_size": 224,
            "config_module": <module>,
            "loader_module": <module>,
            "is_xai": True/False (has explainable AI features),
        }
    """
    global _discovered
    if _discovered is not None:
        return _discovered

    projects = []
    for pid in PROJECT_ORDER:
        proj_dir = PROJECTS_DIR / pid
        if not proj_dir.is_dir():
            continue

        try:
            # Load config module
            config = importlib.import_module(f"projects.{pid}.config")
            loader = importlib.import_module(f"projects.{pid}.model_loader")

            entry = {
                "id": pid,
                "name": getattr(config, "PROJECT_NAME", pid),
                "short_name": getattr(config, "SHORT_NAME", pid),
                "description": getattr(config, "DESCRIPTION", ""),
                "classes": getattr(config, "CLASSES", []),
                "model": getattr(config, "MODEL_NAME", "custom_cnn"),
                "img_size": getattr(config, "IMG_SIZE", 224),
                "config_module": config,
                "loader_module": loader,
                "is_xai": True,  # All projects have XAI
                "has_modalities": hasattr(config, "MODALITIES"),
                "modalities": getattr(config, "MODALITIES", None),
            }
            projects.append(entry)
        except Exception as e:
            print(f"  ! Could not load project {pid}: {e}")
            projects.append({
                "id": pid,
                "name": pid.replace("_", " ").title(),
                "short_name": pid.split("_", 1)[1].replace("_", " ").title(),
                "description": "Medical imaging classification project.",
                "classes": [],
                "model": "unknown",
                "img_size": 224,
                "config_module": None,
                "loader_module": None,
                "is_xai": True,
                "has_modalities": False,
                "modalities": None,
            })

    _discovered = projects
    return projects


def get_project(project_id: str) -> Dict[str, Any]:
    """Get a single project by its ID."""
    for p in discover_projects():
        if p["id"] == project_id:
            return p
    raise ValueError(f"Project '{project_id}' not found")


def get_projects_summary() -> List[Dict[str, Any]]:
    """Return a JSON-safe summary of all projects (no module refs)."""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "short_name": p["short_name"],
            "description": p["description"],
            "classes": p["classes"],
            "model": p["model"],
            "img_size": p["img_size"],
            "is_xai": p["is_xai"],
        }
        for p in discover_projects()
    ]
