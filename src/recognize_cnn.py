"""CNN-based OCR for segmented license plate characters."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import get_settings
from src.logger import logger
from src.segment_characters import CharacterSegmenter


class CharacterCNN(nn.Module):
    """Compact PyTorch CNN for alphanumeric character classification."""

    def __init__(self, num_classes: int = 36) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNRecognizer:
    """Recognize plate characters using a trained PyTorch CNN."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.default_classes = list(settings.allowed_characters)
        self.classes = self.default_classes.copy()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = Path(model_path or settings.cnn_model_path).expanduser().resolve()
        self.segmenter = CharacterSegmenter()
        self.model = CharacterCNN(num_classes=len(self.classes)).to(self.device)
        self.available = False
        self._load_model()

    def _load_model(self) -> None:
        """Load trained weights from the configured path."""

        if not self.model_path.exists():
            logger.warning(
                "CNN OCR weights not found at {}. Train the model with src/train_cnn.py before using cnn mode.",
                self.model_path,
            )
            self.available = False
            return

        try:
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                if state.get("classes"):
                    self.classes = list(state["classes"])
                    self.model = CharacterCNN(num_classes=len(self.classes)).to(self.device)
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.eval()
            self.available = True
            logger.info("Loaded CNN OCR model from {}", self.model_path)
        except Exception as exc:
            logger.exception("Failed to load CNN OCR model from {}: {}", self.model_path, exc)
            self.available = False

    def _prepare_character(self, char_image: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Normalize a segmented character for CNN inference."""

        if char_image is None or not isinstance(char_image, np.ndarray) or char_image.size == 0:
            raise ValueError("Character image must be a non-empty OpenCV numpy array.")

        normalized_image = self.segmenter.normalize_character(char_image)
        normalized = normalized_image.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, normalized_image

    def _predict_character(self, char_image: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Predict a single segmented character."""

        tensor, normalized_image = self._prepare_character(char_image)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)

        character = self.classes[int(pred_idx.item())]
        return character, float(confidence.item()), normalized_image

    def recognize_characters(self, char_images: Sequence[np.ndarray]) -> Tuple[str, List[dict], float]:
        """Recognize an ordered list of segmented character images."""

        if not self.available:
            logger.warning("CNN OCR requested, but no model is available.")
            return "", [], 0.0
        if not char_images:
            logger.debug("CNN OCR received an empty list of character images.")
            return "", [], 0.0

        text: List[str] = []
        char_results: List[dict] = []
        confidences: List[float] = []

        for index, char_image in enumerate(char_images):
            try:
                character, confidence, normalized_image = self._predict_character(char_image)
            except Exception as exc:
                logger.warning("Skipping invalid character at index {}: {}", index, exc)
                continue

            text.append(character)
            confidences.append(confidence)
            char_results.append(
                {
                    "index": index,
                    "character": character,
                    "confidence": round(confidence, 4),
                    "image": normalized_image,
                }
            )

        plate_text = "".join(text)
        overall_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        logger.debug("CNN OCR result='{}' confidence={:.4f}", plate_text, overall_confidence)
        return plate_text, char_results, overall_confidence

    def recognize(self, plate_or_characters: np.ndarray | Sequence[np.ndarray]) -> Tuple[str, float, List[dict]]:
        """Recognize characters from either a plate image or a list of character images.

        This keeps the module compatible with the current `main.py`, which passes
        a preprocessed plate image, and with direct character-list use cases.
        """

        if isinstance(plate_or_characters, np.ndarray):
            segments = self.segmenter.segment(plate_or_characters)
            if not segments:
                logger.debug("CNN OCR could not segment any characters from the plate image.")
                return "", 0.0, []
            char_images = [char_image for char_image, _ in segments]
            boxes = [bbox for _, bbox in segments]
        else:
            char_images = list(plate_or_characters)
            boxes = [None] * len(char_images)

        plate_text, char_results, overall_confidence = self.recognize_characters(char_images)
        for idx, bbox in enumerate(boxes):
            if idx < len(char_results):
                char_results[idx]["bbox"] = bbox
        return plate_text, overall_confidence, char_results


def recognize_with_cnn(char_images: Sequence[np.ndarray]) -> Tuple[str, List[dict], float]:
    """Functional helper for CNN OCR on segmented character images."""

    recognizer = CNNRecognizer()
    return recognizer.recognize_characters(char_images)
