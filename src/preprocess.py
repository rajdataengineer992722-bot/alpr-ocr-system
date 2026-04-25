"""Plate preprocessing pipelines for ALPR OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.logger import logger
from src.utils import save_image


class PlatePreprocessor:
    """Reusable preprocessing pipelines for plate OCR and segmentation."""

    def __init__(self) -> None:
        self.debug_steps: Dict[str, np.ndarray] = {}

    def _reset_debug_steps(self) -> None:
        self.debug_steps = {}

    def _record(self, name: str, image: np.ndarray) -> None:
        """Store a copy of an intermediate step for optional debugging."""

        self.debug_steps[name] = image.copy()

    def _validate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """Validate input plate crop and return a safe copy."""

        if plate_image is None or not isinstance(plate_image, np.ndarray) or plate_image.size == 0:
            raise ValueError("Plate image must be a non-empty OpenCV numpy array.")
        return plate_image.copy()

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR plate crop to grayscale if needed."""

        if image.ndim == 2:
            return image.copy()
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("Unsupported image dimensions for preprocessing.")

    def _resize_keep_aspect(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """Resize plate crop while preserving aspect ratio."""

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Plate image has invalid dimensions.")
        scale = target_width / float(width)
        target_height = max(1, int(height * scale))
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE to improve local contrast."""

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _trim_border(self, image: np.ndarray, margin_ratio: float = 0.04) -> np.ndarray:
        """Remove a small outer border that often contains plate frames or bolts."""

        height, width = image.shape[:2]
        if height < 12 or width < 24:
            return image

        x_margin = max(1, int(width * margin_ratio))
        y_margin = max(1, int(height * margin_ratio))
        trimmed = image[y_margin:height - y_margin, x_margin:width - x_margin]
        if trimmed.size == 0:
            return image
        return trimmed

    def preprocess_for_tesseract(self, plate_image: np.ndarray) -> np.ndarray:
        """Preprocess a plate crop for direct Tesseract OCR."""

        plate_image = self._validate_image(plate_image)
        self._reset_debug_steps()

        gray = self._to_grayscale(plate_image)
        self._record("01_gray", gray)

        trimmed = self._trim_border(gray)
        self._record("02_trimmed", trimmed)

        resized = self._resize_keep_aspect(trimmed, target_width=360)
        self._record("03_resized", resized)

        contrast = self._enhance_contrast(resized)
        self._record("04_contrast", contrast)

        denoised = cv2.bilateralFilter(contrast, d=9, sigmaColor=25, sigmaSpace=25)
        self._record("05_denoised", denoised)

        thresholded = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            8,
        )
        self._record("06_adaptive_threshold", thresholded)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=1)
        self._record("07_morph_close", morphed)

        return morphed

    def preprocess_for_cnn(self, plate_image: np.ndarray) -> np.ndarray:
        """Preprocess a plate crop for contour-based character segmentation and CNN OCR."""

        plate_image = self._validate_image(plate_image)
        self._reset_debug_steps()

        gray = self._to_grayscale(plate_image)
        self._record("01_gray", gray)

        trimmed = self._trim_border(gray)
        self._record("02_trimmed", trimmed)

        resized = self._resize_keep_aspect(trimmed, target_width=320)
        self._record("03_resized", resized)

        contrast = self._enhance_contrast(resized)
        self._record("04_contrast", contrast)

        blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
        self._record("05_gaussian_blur", blurred)

        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._record("06_otsu_threshold", thresholded)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
        self._record("07_morph_open", opened)

        cleaned = cv2.medianBlur(opened, 3)
        self._record("08_median_blur", cleaned)

        return cleaned

    def save_debug_steps(self, output_dir: str | Path, prefix: str) -> Dict[str, str]:
        """Persist intermediate preprocessing images to disk."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: Dict[str, str] = {}
        for step_name, image in self.debug_steps.items():
            path = output_dir / f"{prefix}_{step_name}.png"
            saved_paths[step_name] = save_image(image, path)

        logger.debug("Saved {} preprocessing debug steps for {}", len(saved_paths), prefix)
        return saved_paths

    # Backward-compatible names used elsewhere in the project.
    def for_tesseract(self, plate_image: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for the Tesseract preprocessing pipeline."""

        return self.preprocess_for_tesseract(plate_image)

    def for_segmentation(self, plate_image: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for the CNN preprocessing pipeline."""

        return self.preprocess_for_cnn(plate_image)


def preprocess_for_tesseract(
    plate_image: np.ndarray,
    debug_dir: Optional[str | Path] = None,
    prefix: str = "plate",
) -> np.ndarray:
    """Functional helper for Tesseract preprocessing."""

    processor = PlatePreprocessor()
    processed = processor.preprocess_for_tesseract(plate_image)
    if debug_dir is not None:
        processor.save_debug_steps(debug_dir, prefix)
    return processed


def preprocess_for_cnn(
    plate_image: np.ndarray,
    debug_dir: Optional[str | Path] = None,
    prefix: str = "plate",
) -> np.ndarray:
    """Functional helper for CNN/segmentation preprocessing."""

    processor = PlatePreprocessor()
    processed = processor.preprocess_for_cnn(plate_image)
    if debug_dir is not None:
        processor.save_debug_steps(debug_dir, prefix)
    return processed
