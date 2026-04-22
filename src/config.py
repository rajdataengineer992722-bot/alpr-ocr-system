"""Centralized configuration for the ALPR project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment or return a project-relative default."""

    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Project settings with safe defaults and environment overrides."""

    project_name: str = "ALPR Pro"
    base_dir: Path = BASE_DIR

    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    model_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    output_dir: Path = field(default_factory=lambda: _env_path("ALPR_OUTPUT_DIR", BASE_DIR / "outputs"))

    yolo_model_path: Path = field(
        default_factory=lambda: _env_path(
            "ALPR_YOLO_MODEL",
            BASE_DIR / "models" / "yolo" / "license_plate_detector.pt",
        )
    )
    cnn_model_path: Path = field(
        default_factory=lambda: _env_path(
            "ALPR_CNN_MODEL",
            BASE_DIR / "models" / "cnn" / "character_cnn.pt",
        )
    )

    # Windows-friendly default. Override with TESSERACT_CMD on Linux/Docker or custom installs.
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    tesseract_oem: int = int(os.getenv("ALPR_TESSERACT_OEM", "3"))
    tesseract_psm: int = int(os.getenv("ALPR_TESSERACT_PSM", "7"))

    default_ocr_mode: str = os.getenv("ALPR_OCR_MODE", "tesseract")
    default_confidence: float = float(os.getenv("ALPR_CONFIDENCE", "0.25"))
    default_iou: float = float(os.getenv("ALPR_IOU", "0.45"))
    default_frame_skip: int = int(os.getenv("ALPR_FRAME_SKIP", "1"))
    debug: bool = field(default_factory=lambda: _env_bool("ALPR_DEBUG", False))

    char_image_size: int = int(os.getenv("ALPR_CHAR_IMAGE_SIZE", "32"))
    # Keep this order aligned with src.dataset.build_label_maps() and training checkpoints.
    allowed_characters: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    plate_regex_patterns: List[str] = field(
        default_factory=lambda: [
            r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$",
            r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$",
            r"^[A-Z0-9]{5,10}$",
        ]
    )

    confusion_map_alpha_to_digit: Dict[str, str] = field(
        default_factory=lambda: {
            "O": "0",
            "Q": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "B": "8",
        }
    )
    confusion_map_digit_to_alpha: Dict[str, str] = field(
        default_factory=lambda: {
            "0": "O",
            "1": "I",
            "2": "Z",
            "5": "S",
            "6": "G",
            "8": "B",
        }
    )

    tracker_iou_threshold: float = float(os.getenv("ALPR_TRACKER_IOU", "0.30"))
    tracker_max_missing: int = int(os.getenv("ALPR_TRACKER_MAX_MISSING", "12"))
    tracker_vote_window: int = int(os.getenv("ALPR_TRACKER_VOTE_WINDOW", "12"))

    max_video_frames: int = int(os.getenv("ALPR_MAX_VIDEO_FRAMES", "0"))
    save_json_results: bool = field(default_factory=lambda: _env_bool("ALPR_SAVE_JSON_RESULTS", False))

    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

    @property
    def output_subdirs(self) -> Dict[str, Path]:
        """Standard output locations used across CLI, UI, and API."""

        return {
            "crops": self.output_dir / "crops",
            "processed": self.output_dir / "processed",
            "segmented": self.output_dir / "segmented",
            "annotated": self.output_dir / "annotated",
            "logs": self.output_dir / "logs",
            "api_results": self.output_dir / "api_results",
            "debug": self.output_dir / "debug",
        }


_settings = Settings()


def get_settings() -> Settings:
    """Return the shared project settings object."""

    return _settings


settings = _settings


__all__ = ["BASE_DIR", "Settings", "get_settings", "settings"]
