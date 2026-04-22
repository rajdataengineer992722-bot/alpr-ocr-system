"""Training script for the CNN-based ALPR character recognizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import get_settings
from src.dataset import prepare_dataloaders
from src.logger import logger, setup_logger
from src.recognize_cnn import CharacterCNN


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch accuracy from model logits."""

    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / max(1, labels.numel())


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        batches += 1

    return total_loss / max(1, batches), total_acc / max(1, batches)


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation without gradient updates."""

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, labels)
            batches += 1

    return total_loss / max(1, batches), total_acc / max(1, batches)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    classes: List[str],
    epoch: int,
    val_acc: float,
    history: Dict[str, List[float]],
) -> str:
    """Save a PyTorch model checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "epoch": epoch,
            "val_acc": val_acc,
            "history": history,
        },
        path,
    )
    logger.info("Saved checkpoint to {}", path)
    return str(path.resolve())


def save_history(history: Dict[str, List[float]], output_dir: str | Path) -> tuple[str, str]:
    """Save metrics history as JSON and loss/accuracy curves as PNG."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    plot_path = output_dir / "training_curves.png"
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    logger.info("Saved training history to {}", history_path)
    logger.info("Saved training curves to {}", plot_path)
    return str(history_path.resolve()), str(plot_path.resolve())


def train(
    dataset_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    image_size: int | None = None,
    output_model_path: str | Path | None = None,
    augment: bool = True,
    validation_split: float = 0.2,
    num_workers: int = 0,
) -> Dict[str, object]:
    """Train the character CNN and save best/final checkpoints."""

    settings = get_settings()
    output_model_path = Path(output_model_path or settings.cnn_model_path)
    output_dir = output_model_path.parent
    final_model_path = output_dir / "character_cnn_final.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    train_loader, val_loader, classes = prepare_dataloaders(
        root_dir=dataset_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size or settings.char_image_size,
        augment=augment,
        num_workers=num_workers,
    )

    model = CharacterCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }
    best_val_acc = -1.0
    best_model_path = str(output_model_path.resolve())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["learning_rate"].append(round(current_lr, 10))

        logger.info(
            "Epoch {}/{} | train_loss={:.4f} train_acc={:.4f} | val_loss={:.4f} val_acc={:.4f} | lr={:.6f}",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = save_checkpoint(output_model_path, model, classes, epoch, val_acc, history)

    final_model = save_checkpoint(final_model_path, model, classes, epochs, history["val_acc"][-1], history)
    history_path, curves_path = save_history(history, output_dir)

    return {
        "best_model_path": best_model_path,
        "final_model_path": final_model,
        "history_path": history_path,
        "curves_path": curves_path,
        "best_val_acc": round(best_val_acc, 6),
        "classes": classes,
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for CNN training."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train ALPR CNN character recognizer")
    parser.add_argument("--dataset_dir", default=str(settings.data_dir / "chars"), help="Character dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=settings.char_image_size)
    parser.add_argument("--output_model_path", default=str(settings.cnn_model_path))
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_augment", action="store_true", help="Disable training augmentation")
    return parser


def main() -> None:
    """CLI entry point."""

    setup_logger("train_cnn.log")
    args = build_argparser().parse_args()

    try:
        result = train(
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            image_size=args.image_size,
            output_model_path=args.output_model_path,
            augment=not args.no_augment,
            validation_split=args.validation_split,
            num_workers=args.num_workers,
        )
        logger.info("Training complete:\n{}", json.dumps(result, indent=2))
    except Exception as exc:
        logger.exception("CNN training failed: {}", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
