FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSERACT_CMD=/usr/bin/tesseract \
    ALPR_OUTPUT_DIR=/app/outputs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p \
    outputs/crops \
    outputs/processed \
    outputs/segmented \
    outputs/annotated \
    outputs/logs \
    outputs/api_results \
    outputs/debug \
    models/yolo \
    models/cnn \
    data/samples

EXPOSE 8000 8501

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
