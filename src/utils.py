"""Reusable utility helpers for the ALPR project."""

from __future__ import annotations

import base64
import csv
import json
import re
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from src.config import get_settings
from src.logger import logger


def ensure_project_dirs(base_dir: str | Path | None = None) -> dict[str, Path]:
    """Create the standard ALPR project output directories."""

    settings = get_settings()
    root = Path(base_dir).resolve() if base_dir else settings.output_dir.resolve()
    dirs = {
        "root": root,
        "crops": root / "crops",
        "processed": root / "processed",
        "segmented": root / "segmented",
        "annotated": root / "annotated",
        "logs": root / "logs",
        "api_results": root / "api_results",
        "debug": root / "debug",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def timestamp_slug() -> str:
    """Return a compact filesystem-safe timestamp."""

    return time.strftime("%Y%m%d_%H%M%S")


def safe_filename(value: str, default: str = "alpr") -> str:
    """Convert arbitrary text into a safe filename stem."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def unique_name(prefix: str, suffix: str) -> str:
    """Generate a unique filename with timestamp and short random id."""

    return f"{safe_filename(prefix)}_{timestamp_slug()}_{uuid.uuid4().hex[:8]}{suffix}"


def read_image(image_path: str | Path) -> np.ndarray:
    """Safely read an image from disk in OpenCV BGR format."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def open_video(video_source: str | Path | int) -> cv2.VideoCapture:
    """Open a video file or webcam source with validation."""

    if not isinstance(video_source, int):
        path = Path(video_source)
        if not path.exists():
            raise FileNotFoundError(f"Video path does not exist: {path}")
        source = str(path)
    else:
        source = video_source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {video_source}")
    return cap


def image_from_base64(payload: str) -> np.ndarray:
    """Decode a base64 image payload into an OpenCV BGR image."""

    try:
        binary = base64.b64decode(payload)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    array = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode base64 payload as an image.")
    return image


def save_image(image: np.ndarray, path: str | Path) -> str:
    """Save an OpenCV image to disk and return the absolute path."""

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot save an empty image.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Unable to save image to {path}")
    logger.debug("Saved image to {}", path)
    return str(path.resolve())


def save_json(payload: object, path: str | Path) -> str:
    """Save a JSON payload to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    logger.debug("Saved JSON to {}", path)
    return str(path.resolve())


def save_csv(rows: Sequence[Mapping], path: str | Path) -> str:
    """Save rows to CSV using pandas for convenience."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    logger.debug("Saved CSV log to {}", path)
    return str(path.resolve())


def append_csv_log(row: Mapping, path: str | Path) -> str:
    """Append a single row to a CSV log, creating the file if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))
    return str(path.resolve())


def draw_detection(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str = "",
    confidence: float | None = None,
    track_id: int | None = None,
    color: tuple[int, int, int] = (36, 255, 12),
) -> np.ndarray:
    """Draw one bounding box and label on an image."""

    canvas = image
    x1, y1, x2, y2 = bbox
    h, w = canvas.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    label_parts = []
    if track_id is not None:
        label_parts.append(f"ID {track_id}")
    label_parts.append(text or "UNKNOWN")
    if confidence is not None:
        label_parts.append(f"{confidence:.2f}")
    label = " | ".join(label_parts)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    label_width = max(180, len(label) * 10)
    label_x2 = min(w - 1, x1 + label_width)
    label_y1 = max(0, y1 - 30)
    cv2.rectangle(canvas, (x1, label_y1), (label_x2, y1), color, -1)
    cv2.putText(
        canvas,
        label,
        (x1 + 5, max(16, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_plate_annotations(image: np.ndarray, results: Iterable[Mapping]) -> np.ndarray:
    """Draw all ALPR detections on a frame/image."""

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot draw annotations on an empty image.")

    canvas = image.copy()
    for item in results:
        bbox_raw = item.get("bbox")
        if bbox_raw is None:
            continue
        bbox = tuple(map(int, bbox_raw))
        text = str(item.get("stable_text") or item.get("detected_text") or "")
        confidence = item.get("stable_confidence")
        if confidence is None:
            confidence = item.get("combined_confidence")
        confidence = float(confidence) if confidence is not None else None
        track_id = item.get("track_id")
        track_id = int(track_id) if track_id is not None else None
        draw_detection(canvas, bbox, text=text, confidence=confidence, track_id=track_id)
    return canvas


def crop_region(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Safely crop an image region using a bounding box."""

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot crop from an empty image.")

    x1, y1, x2, y2 = map(int, bbox)
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=image.dtype)
    return image[y1:y2, x1:x2].copy()


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union for two bounding boxes."""

    ax1, ay1, ax2, ay2 = map(int, box_a)
    bx1, by1, bx2, by2 = map(int, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union else 0.0


def is_image_file(path: str | Path) -> bool:
    """Return whether a path has a supported image extension."""

    return Path(path).suffix.lower() in get_settings().image_extensions


def is_video_file(path: str | Path) -> bool:
    """Return whether a path has a supported video extension."""

    return Path(path).suffix.lower() in get_settings().video_extensions


def _json_default(value: object) -> object:
    """JSON serializer for dataclasses and paths."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
