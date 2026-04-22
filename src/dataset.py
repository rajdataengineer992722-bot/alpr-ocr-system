"""Dataset utilities for CNN-based ALPR character recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.config import get_settings
from src.logger import logger


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_label_maps() -> tuple[Dict[str, int], Dict[int, str]]:
    """Build class mappings for digits 0-9 and uppercase letters A-Z."""

    classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    class_to_idx = {label: index for index, label in enumerate(classes)}
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_dataset(root_dir: str | Path) -> List[tuple[Path, int]]:
    """Scan a character dataset directory and return image paths with labels."""

    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Character dataset directory does not exist: {root}")

    class_to_idx, _ = build_label_maps()
    samples: List[tuple[Path, int]] = []

    for label, class_index in class_to_idx.items():
        class_dir = root / label
        if not class_dir.exists():
            logger.warning("Dataset class folder is missing: {}", class_dir)
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append((image_path, class_index))

    if not samples:
        raise ValueError(f"No character images found under {root}")

    logger.info("Loaded {} character image paths from {}", len(samples), root)
    return samples


def _build_transform(image_size: int, augment: bool = False):
    """Create torchvision transforms for grayscale character OCR."""

    steps = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]
    if augment:
        steps.extend(
            [
                transforms.RandomAffine(
                    degrees=6,
                    translate=(0.05, 0.05),
                    scale=(0.90, 1.10),
                    shear=3,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    return transforms.Compose(steps)


class CharacterDataset(Dataset):
    """PyTorch dataset for segmented alphanumeric character images."""

    def __init__(
        self,
        root_dir: str | Path,
        samples: Sequence[tuple[Path, int]] | None = None,
        transform=None,
        skip_bad_images: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_to_idx, self.idx_to_class = build_label_maps()
        self.classes = [self.idx_to_class[index] for index in range(len(self.idx_to_class))]
        self.samples = list(samples) if samples is not None else load_dataset(self.root_dir)
        self.transform = transform or _build_transform(get_settings().char_image_size, augment=False)
        self.skip_bad_images = skip_bad_images

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load an image as grayscale PIL image."""

        return Image.open(image_path).convert("L")

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        try:
            image = self._load_image(image_path)
        except (OSError, UnidentifiedImageError) as exc:
            if not self.skip_bad_images:
                raise
            logger.warning("Skipping unreadable image {}: {}", image_path, exc)
            fallback = torch.zeros((1, get_settings().char_image_size, get_settings().char_image_size))
            return fallback, label

        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    validation_split: float = 0.2,
    image_size: int | None = None,
    augment: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, List[str]]:
    """Create PyTorch train/validation dataloaders for CNN training."""

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    settings = get_settings()
    image_size = image_size or settings.char_image_size
    samples = load_dataset(root_dir)
    if len(samples) < 2:
        raise ValueError("At least 2 images are required for a train/validation split.")

    base_dataset = CharacterDataset(root_dir=root_dir, samples=samples, transform=None)
    val_size = max(1, int(len(base_dataset) * validation_split))
    val_size = min(val_size, len(base_dataset) - 1)
    train_size = len(base_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    train_dataset = CharacterDataset(
        root_dir=root_dir,
        samples=[base_dataset.samples[i] for i in train_subset.indices],
        transform=_build_transform(image_size, augment=augment),
    )
    val_dataset = CharacterDataset(
        root_dir=root_dir,
        samples=[base_dataset.samples[i] for i in val_subset.indices],
        transform=_build_transform(image_size, augment=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "Prepared dataloaders: train={} validation={} classes={}",
        len(train_dataset),
        len(val_dataset),
        len(train_dataset.classes),
    )
    return train_loader, val_loader, train_dataset.classes


def create_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    validation_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Backward-compatible wrapper used by train_cnn.py."""

    return prepare_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        augment=True,
    )
