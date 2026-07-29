"""Shared Pipeline — Training loop with early stopping, logging, and checkpointing."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from shared.utils.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return loss & metrics."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).detach()
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return loss & metrics."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics


def train_model(model, train_loader, val_loader, device,
                num_epochs=50, lr=1e-4, patience=7,
                checkpoint_path=None):
    """Full training loop with early stopping.

    Args:
        model: PyTorch model.
        train_loader, val_loader: DataLoaders.
        device: torch.device.
        num_epochs: Max epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        checkpoint_path: Where to save best model weights.

    Returns:
        dict of training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    model = model.to(device)
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        train_metrics.pop("confusion_matrix", None)
        val_metrics.pop("confusion_matrix", None)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"  Train Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")

        # Checkpoint best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_no_improve = 0
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  [OK] Saved best model (acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs "
                      f"(best epoch {best_epoch}, acc={best_val_acc:.4f})")
                break

    return history
