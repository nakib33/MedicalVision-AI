"""Shared Pipeline — Image transforms for train and inference."""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from shared.config import MEAN, STD


def get_train_transforms(img_size: int = 224):
    """Training transforms with augmentation."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms(img_size: int = 224):
    """Validation / test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def get_inference_transform(img_size: int = 224):
    """Inference transform (same as val, but returns PIL path info)."""
    return get_val_transforms(img_size)


def inverse_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(STD).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean
