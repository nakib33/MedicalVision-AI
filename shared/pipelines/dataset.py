"""Shared Pipeline — Dataset and DataLoader utilities."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from typing import Optional, Callable


class MedicalImageDataset(Dataset):
    """Generic medical image dataset from folder structure:

    root/
        class_0/
            img1.jpg
            img2.jpg
            ...
        class_1/
            ...
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 class_names: Optional[list] = None):
        """
        Args:
            root: Path to dataset folder with class subfolders.
            transform: Torchvision transform pipeline.
            class_names: If provided, filters to only these classes
                         (and uses this order).
        """
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, label_idx)
        self.classes = []

        # Discover classes
        if class_names:
            self.classes = class_names
        else:
            self.classes = sorted(
                [p.name for p in self.root.iterdir() if p.is_dir()]
            )

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Gather image paths
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in extensions:
                    self.samples.append((str(f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(data_root: str, class_names: list,
                       img_size: int = 224, batch_size: int = 32,
                       num_workers: int = 2):
    """Convenience: create train/val/test loaders from standard folder structure.

    Expects:
        data_root/
            train/
                class0/
                class1/
            val/
                class0/
                class1/
            test/
                class0/
                class1/
    """
    from .transforms import get_train_transforms, get_val_transforms

    train_ds = MedicalImageDataset(
        str(Path(data_root) / "train"),
        transform=get_train_transforms(img_size),
        class_names=class_names,
    )
    val_ds = MedicalImageDataset(
        str(Path(data_root) / "val"),
        transform=get_val_transforms(img_size),
        class_names=class_names,
    )
    test_ds = MedicalImageDataset(
        str(Path(data_root) / "test"),
        transform=get_val_transforms(img_size),
        class_names=class_names,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes
