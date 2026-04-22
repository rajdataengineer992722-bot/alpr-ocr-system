"""Tesseract OCR module for license plate recognition."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pytesseract
from pytesseract import Output

from src.config import get_settings
from src.logger import logger


class TesseractRecognizer:
    """Reusable wrapper around pytesseract for ALPR plate OCR."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self.settings = get_settings()
        self.tesseract_cmd = tesseract_cmd or self.settings.tesseract_cmd
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        """Configure the Tesseract executable path, with Windows-friendly support."""

        system_tesseract = shutil.which("tesseract")
        if system_tesseract:
            pytesseract.pytesseract.tesseract_cmd = system_tesseract
            logger.debug("Using Tesseract from PATH: {}", system_tesseract)
            return

        if self.tesseract_cmd and Path(self.tesseract_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            logger.debug("Using configured Tesseract path: {}", self.tesseract_cmd)
            return

        logger.warning(
            "Tesseract executable was not found on PATH and configured path '{}' does not exist.",
            self.tesseract_cmd,
        )

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate incoming OCR image."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Tesseract OCR expects a non-empty OpenCV numpy array.")
        return image

    def _clean_text(self, text: str) -> str:
        """Normalize raw OCR text into plate-friendly format."""

        text = text.upper().replace(" ", "").replace("\n", "").replace("\t", "")
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def _build_config(self) -> str:
        """Build the Tesseract CLI config string."""

        return (
            f"--oem {self.settings.tesseract_oem} "
            f"--psm {self.settings.tesseract_psm} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """Run Tesseract OCR and return cleaned text with average confidence."""

        image = self._validate_image(image)
        config = self._build_config()

        try:
            data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError:
            logger.error(
                "Tesseract is not installed or not accessible. Configure TESSERACT_CMD in config or environment."
            )
            return "", 0.0
        except Exception as exc:
            logger.exception("Tesseract OCR inference failed: {}", exc)
            return "", 0.0

        tokens = []
        confidences = []
        for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            cleaned = self._clean_text(raw_text or "")
            if not cleaned:
                continue
            tokens.append(cleaned)
            try:
                conf_value = float(raw_conf)
            except (TypeError, ValueError):
                conf_value = -1.0
            if conf_value >= 0:
                confidences.append(conf_value)

        merged_text = self._clean_text("".join(tokens))
        mean_confidence = round((sum(confidences) / len(confidences)) / 100.0, 4) if confidences else 0.0

        logger.debug(
            "Tesseract OCR result='{}' confidence={:.4f}",
            merged_text,
            mean_confidence,
        )
        return merged_text, mean_confidence


def recognize_with_tesseract(image: np.ndarray) -> Tuple[str, float]:
    """Functional helper for Tesseract plate OCR."""

    recognizer = TesseractRecognizer()
    return recognizer.recognize(image)
