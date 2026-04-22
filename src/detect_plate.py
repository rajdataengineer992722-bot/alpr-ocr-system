"""YOLO-based license plate detection module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.config import get_settings
from src.logger import logger
from src.utils import crop_region


@dataclass
class PlateDetection:
    """Structured detection result for a single license plate."""

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: Optional[int]
    class_name: str
    crop: np.ndarray


class PlateDetector:
    """Reusable Ultralytics YOLO wrapper for plate detection."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        settings = get_settings()
        self.model_path = Path(model_path or settings.yolo_model_path).expanduser().resolve()
        self.model: Optional[YOLO] = None
        self.load_model()

    def load_model(self) -> None:
        """Load the YOLO model from the configured path."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                "YOLO license plate model not found at "
                f"'{self.model_path}'. Set ALPR_YOLO_MODEL or place the weights in the configured location."
            )

        try:
            self.model = YOLO(str(self.model_path))
            logger.info("Loaded YOLO plate detector from {}", self.model_path)
        except Exception as exc:
            logger.exception("Failed to load YOLO model from {}: {}", self.model_path, exc)
            raise RuntimeError(f"Unable to load YOLO model from {self.model_path}") from exc

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> List[PlateDetection]:
        """Run detection on an OpenCV image/frame and return structured results."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty OpenCV numpy array.")
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded.")

        settings = get_settings()
        conf_threshold = settings.default_confidence if conf_threshold is None else conf_threshold
        iou_threshold = settings.default_iou if iou_threshold is None else iou_threshold

        try:
            predictions = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("YOLO inference failed: {}", exc)
            raise RuntimeError("YOLO inference failed.") from exc

        detections: List[PlateDetection] = []
        for prediction in predictions:
            names = prediction.names if hasattr(prediction, "names") else {}
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0].item())
                    class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else None
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) and class_id is not None else "plate"
                    bbox = (x1, y1, x2, y2)
                    crop = crop_region(image, bbox)
                    if crop.size == 0:
                        logger.debug("Skipping empty crop for bbox {}", bbox)
                        continue
                    detections.append(
                        PlateDetection(
                            bbox=bbox,
                            confidence=round(confidence, 4),
                            class_id=class_id,
                            class_name=class_name,
                            crop=crop,
                        )
                    )
                except Exception as exc:
                    logger.warning("Skipping malformed detection result: {}", exc)
                    continue

        logger.debug("Plate detector returned {} detections", len(detections))
        return detections

    def extract_crops(
        self,
        image: np.ndarray,
        detections: List[PlateDetection],
    ) -> List[np.ndarray]:
        """Extract plate crops from an image using detection bounding boxes."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty OpenCV numpy array.")
        return [crop_region(image, detection.bbox) for detection in detections]


# Backward-compatible alias used by the rest of the project.
YOLOPlateDetector = PlateDetector
