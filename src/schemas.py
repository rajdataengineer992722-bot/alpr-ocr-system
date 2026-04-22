"""Pydantic schemas for the ALPR FastAPI backend."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    project: str
    model_ready: bool
    detail: str


class BoundingBox(BaseModel):
    """Detected plate bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int


class OutputPaths(BaseModel):
    """Saved output artifact paths."""

    crop: Optional[str] = None
    processed: Optional[str] = None
    segmented: List[str] = Field(default_factory=list)
    annotated: Optional[str] = None
    debug_steps: Dict[str, str] = Field(default_factory=dict)


class PlatePrediction(BaseModel):
    """Single plate prediction payload."""

    track_id: Optional[int] = None
    detected_text: str
    raw_text: str
    ocr_mode: str
    detection_confidence: float
    ocr_confidence: float
    combined_confidence: float
    stable_confidence: Optional[float] = None
    bbox: BoundingBox
    is_valid: bool
    stable_text: Optional[str] = None
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    outputs: OutputPaths = Field(default_factory=OutputPaths)


class PredictionResponse(BaseModel):
    """Top-level image prediction response."""

    source: str
    source_type: str
    ocr_mode: str
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    plate_count: int
    results: List[PlatePrediction]
    csv_path: Optional[str] = None
    json_path: Optional[str] = None
    annotated_path: Optional[str] = None


class ErrorResponse(BaseModel):
    """Structured API error response."""

    detail: str
    error_type: Optional[str] = None


# Backward-compatible aliases for previous API code.
InferenceResponse = PredictionResponse
PlateResult = PlatePrediction
