"""FastAPI routes for ALPR inference."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.config import get_settings
from src.logger import logger
from src.main import ALPRSystem
from src.schemas import ErrorResponse, HealthResponse, PredictionResponse


router = APIRouter(tags=["ALPR"])
settings = get_settings()


@lru_cache(maxsize=1)
def get_system(debug: bool = False) -> ALPRSystem:
    """Lazily initialize the ALPR backend so health can report readiness."""

    return ALPRSystem(debug=debug)


def _validate_image_upload(file: UploadFile) -> None:
    """Validate uploaded image metadata before inference."""

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.image_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{suffix}'. Supported types: {', '.join(settings.image_extensions)}",
        )


def _validate_runtime_options(ocr_mode: str, confidence_threshold: float) -> None:
    if ocr_mode not in {"tesseract", "cnn"}:
        raise HTTPException(status_code=422, detail="ocr_mode must be 'tesseract' or 'cnn'.")
    if not 0.0 < confidence_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="confidence_threshold must be between 0 and 1.")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and backend readiness."""

    try:
        get_system()
        return HealthResponse(
            status="ok",
            project=settings.project_name,
            model_ready=True,
            detail="ALPR backend initialized successfully.",
        )
    except Exception as exc:
        logger.exception("ALPR health check failed: {}", exc)
        return HealthResponse(
            status="degraded",
            project=settings.project_name,
            model_ready=False,
            detail=str(exc),
        )


@router.post(
    "/predict/image",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_image(
    file: UploadFile = File(..., description="Image file containing a vehicle license plate"),
    ocr_mode: str = Form("tesseract"),
    confidence_threshold: float = Form(0.25),
    debug: bool = Form(False),
) -> PredictionResponse:
    """Run ALPR on an uploaded image and return structured JSON results."""

    _validate_image_upload(file)
    _validate_runtime_options(ocr_mode, confidence_threshold)

    temp_path: str | None = None
    try:
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            payload = await file.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            temp_file.write(payload)
            temp_path = temp_file.name

        result = get_system(debug=debug).process_image(
            image=temp_path,
            source_name=Path(file.filename or "upload").stem,
            ocr_mode=ocr_mode,
            conf_threshold=confidence_threshold,
            save_outputs=True,
        )
        result.pop("annotated_image", None)

        if result.get("plate_count", 0) == 0:
            raise HTTPException(status_code=404, detail="No license plate was detected in the uploaded image.")

        if all(not item.get("detected_text") for item in result.get("results", [])):
            raise HTTPException(status_code=422, detail="Plate detected, but OCR did not produce readable text.")

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image prediction failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
