"""Character segmentation for license plate OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import get_settings
from src.logger import logger
from src.utils import save_image


BBox = tuple[int, int, int, int]
Segment = tuple[np.ndarray, BBox]


class CharacterSegmenter:
    """Segment character candidates from a preprocessed license plate crop."""

    def __init__(
        self,
        min_area: int = 80,
        max_area_ratio: float = 0.30,
        min_height_ratio: float = 0.30,
        max_height_ratio: float = 0.95,
        min_width_ratio: float = 0.02,
        max_width_ratio: float = 0.35,
        min_aspect_ratio: float = 0.15,
        max_aspect_ratio: float = 1.10,
        max_candidates: int = 16,
    ) -> None:
        self.settings = get_settings()
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.min_width_ratio = min_width_ratio
        self.max_width_ratio = max_width_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_candidates = max_candidates

    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """Validate and normalize the input plate image."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Segmenter expects a non-empty OpenCV numpy array.")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _prepare_binary(self, image: np.ndarray) -> np.ndarray:
        """Ensure the incoming plate image is a clean binary image."""

        gray = self._validate_input(image)

        # If the image is not truly binary yet, threshold it.
        unique_values = np.unique(gray)
        if len(unique_values) > 2:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = gray

        # If the background is white and text is dark, invert for contour extraction.
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return binary

    def _is_valid_candidate(self, bbox: BBox, plate_shape: tuple[int, int]) -> bool:
        """Apply geometric filters to reject obvious non-character regions."""

        x1, y1, x2, y2 = bbox
        plate_h, plate_w = plate_shape
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width == 0 or height == 0:
            return False

        area = width * height
        aspect_ratio = width / float(height)
        rel_height = height / float(plate_h)
        rel_width = width / float(plate_w)
        rel_area = area / float(plate_h * plate_w)

        if area < self.min_area:
            return False
        if rel_area > self.max_area_ratio:
            return False
        if not (self.min_height_ratio <= rel_height <= self.max_height_ratio):
            return False
        if not (self.min_width_ratio <= rel_width <= self.max_width_ratio):
            return False
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False
        return True

    def _find_candidates(self, binary_plate: np.ndarray) -> List[Segment]:
        """Find contour-based character candidates."""

        contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plate_h, plate_w = binary_plate.shape[:2]
        candidates: List[Segment] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            if not self._is_valid_candidate(bbox, (plate_h, plate_w)):
                continue

            char_crop = binary_plate[y:y + h, x:x + w]
            if char_crop.size == 0:
                continue
            candidates.append((char_crop, bbox))

        candidates.sort(key=lambda item: item[1][0])
        if len(candidates) > self.max_candidates:
            logger.debug(
                "Character segmentation produced {} candidates; truncating to {} based on left-to-right order.",
                len(candidates),
                self.max_candidates,
            )
            candidates = candidates[: self.max_candidates]
        return candidates

    def normalize_character(self, char_image: np.ndarray, size: Optional[int] = None) -> np.ndarray:
        """Normalize a character crop to a fixed square size."""

        size = size or self.settings.char_image_size
        char_image = self._validate_input(char_image)

        # Ensure the character is white on black for CNN input consistency.
        # Binary plate preprocessing usually creates white glyphs on a dark
        # background. If the crop is mostly bright, invert it to restore that
        # convention before resizing.
        if np.mean(char_image) > 127:
            char_image = cv2.bitwise_not(char_image)

        canvas = np.zeros((size, size), dtype=np.uint8)
        h, w = char_image.shape[:2]
        if h == 0 or w == 0:
            return canvas

        scale = min((size - 6) / float(w), (size - 6) / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(char_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def draw_bounding_boxes(self, plate_image: np.ndarray, segments: List[Segment]) -> np.ndarray:
        """Draw segmented character boxes on the plate image for debugging."""

        if plate_image.ndim == 2:
            canvas = cv2.cvtColor(plate_image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            canvas = plate_image.copy()

        for idx, (_, bbox) in enumerate(segments):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                canvas,
                str(idx),
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def save_segments(self, segments: List[Segment], output_dir: str | Path, prefix: str) -> List[str]:
        """Save normalized segmented character images."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: List[str] = []
        for idx, (char_image, _) in enumerate(segments):
            normalized = self.normalize_character(char_image)
            path = output_dir / f"{prefix}_char_{idx:02d}.png"
            paths.append(save_image(normalized, path))
        logger.debug("Saved {} segmented character crops for {}", len(paths), prefix)
        return paths

    def segment(
        self,
        plate_image: np.ndarray,
        debug: bool = False,
        debug_dir: str | Path | None = None,
        prefix: str = "plate",
    ) -> List[Segment]:
        """Segment characters and optionally save debug artifacts."""

        binary_plate = self._prepare_binary(plate_image)
        segments = self._find_candidates(binary_plate)

        if not segments:
            logger.debug("No valid character segments were detected.")
            return []

        if debug and debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            boxed = self.draw_bounding_boxes(binary_plate, segments)
            save_image(boxed, debug_dir / f"{prefix}_segments_boxed.png")
            self.save_segments(segments, debug_dir, prefix)

        return segments


def segment_characters(
    image: np.ndarray,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    prefix: str = "plate",
) -> Dict[str, object]:
    """Functional character segmentation entry point.

    Returns a dictionary containing normalized character images, bounding boxes,
    raw segments, and optional debug overlay path.
    """

    segmenter = CharacterSegmenter()
    segments = segmenter.segment(image, debug=debug, debug_dir=debug_dir, prefix=prefix)

    normalized_images = [segmenter.normalize_character(char_image) for char_image, _ in segments]
    boxes = [bbox for _, bbox in segments]

    debug_overlay_path = None
    if debug and debug_dir is not None and segments:
        debug_dir = Path(debug_dir)
        debug_overlay = segmenter.draw_bounding_boxes(segmenter._prepare_binary(image), segments)
        debug_overlay_path = save_image(debug_overlay, debug_dir / f"{prefix}_segments_overlay.png")

    return {
        "characters": normalized_images,
        "boxes": boxes,
        "segments": segments,
        "debug_overlay_path": debug_overlay_path,
    }
