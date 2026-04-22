"""Uvicorn-compatible FastAPI application for ALPR Pro."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from src.config import get_settings
from src.logger import logger, setup_logger
from src.schemas import ErrorResponse


setup_logger("api.log")
settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    description="Automatic License Plate Recognition API powered by YOLO, OpenCV, and OCR.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return structured JSON for unexpected server errors."""

    logger.exception("Unhandled API error at {}: {}", request.url.path, exc)
    payload = ErrorResponse(detail="Internal server error.", error_type=type(exc).__name__)
    return JSONResponse(status_code=500, content=payload.model_dump())


app.include_router(router)
