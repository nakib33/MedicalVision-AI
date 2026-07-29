"""MedicalVision AI Suite — Central Configuration

Paths, device detection, and shared constants used across all 12 projects.
"""
import os
import torch
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent  # d:\DP Project\Medical CNN
PROJECTS_DIR = ROOT_DIR / "projects"
TRAINED_MODELS_DIR = ROOT_DIR / "trained_models"
SHARED_DIR = ROOT_DIR / "shared"

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_AVAILABLE = torch.cuda.is_available()

# ── Image Settings ────────────────────────────────────────────
DEFAULT_IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ── Training Defaults ─────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 7

# ── Server ────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── Helpers ───────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_trained_model_path(project_id: str) -> Path:
    """Return the expected path for a project's trained model weights."""
    return TRAINED_MODELS_DIR / f"{project_id}_best.pth"
