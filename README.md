# ALPR Pro

ALPR Pro is a production-style Automatic License Plate Recognition system built with Python, YOLO, OpenCV, Tesseract OCR, CNN-based character recognition, Streamlit, FastAPI, and Docker.

The project detects license plates from images, videos, and webcam streams, crops and preprocesses each plate, recognizes text with either Tesseract or a custom CNN OCR model, stabilizes recognition across frames, and exports annotated outputs plus CSV logs.

## Key Features

- YOLO-based license plate detection with configurable confidence and IoU thresholds
- Dual OCR support:
  - Tesseract OCR baseline
  - CNN-based character recognition using segmented characters
- Image, video, and webcam inference
- Multi-plate detection support
- Real-time temporal smoothing using lightweight tracking and confidence-aware voting
- OpenCV preprocessing pipelines for OCR and segmentation
- OCR cleanup, confusion correction, and regex-based validation
- Streamlit web UI for portfolio demos
- FastAPI backend for integration and deployment
- Docker and Docker Compose support
- CSV logs, cropped plates, processed plates, segmented characters, and annotated media exports

## Architecture

```text
Input image/video/webcam
  -> YOLO license plate detection
  -> Plate crop extraction
  -> OpenCV preprocessing
     -> grayscale
     -> resize
     -> contrast enhancement
     -> denoise / blur
     -> thresholding
     -> morphology
  -> OCR engine
     -> Tesseract direct OCR
     -> Character segmentation + CNN classifier
  -> OCR post-processing
     -> clean text
     -> correct common OCR confusions
     -> regex validation
  -> Temporal stabilization for video/webcam
  -> Annotated output + CSV logs + saved artifacts
```

## Folder Structure

```text
alpr-pro/
  api/
    app.py
    routes.py
  data/
    raw/
    processed/
    chars/
    samples/
  models/
    yolo/
    cnn/
  outputs/
    crops/
    processed/
    segmented/
    annotated/
    logs/
    api_results/
    debug/
  src/
    config.py
    detect_plate.py
    preprocess.py
    segment_characters.py
    recognize_tesseract.py
    recognize_cnn.py
    postprocess.py
    tracker.py
    utils.py
    logger.py
    dataset.py
    train_cnn.py
    main.py
    schemas.py
  app.py
  requirements.txt
  Dockerfile
  docker-compose.yml
  README.md
```

## Setup

### Python Version

Use Python `3.10`, `3.11`, or `3.12`. Python `3.11` is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Tesseract Installation

Windows:

Install Tesseract OCR, then set the executable path:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Linux:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

### Model Files

Add model weights before running inference:

```text
models/yolo/license_plate_detector.pt
models/cnn/character_cnn.pt
```

The YOLO model is required for detection. The CNN model is required only when using `--ocr_mode cnn`.

## CLI Usage

Image inference:

```bash
python -m src.main --image data/samples/car.jpg --ocr_mode tesseract --conf 0.25 --save --show
```

Video inference:

```bash
python -m src.main --video data/samples/traffic.mp4 --ocr_mode tesseract --frame_skip 2 --conf 0.25 --save
```

Webcam inference:

```bash
python -m src.main --webcam 0 --ocr_mode tesseract --frame_skip 2 --conf 0.25 --save --show
```

CNN OCR mode:

```bash
python -m src.main --image data/samples/car.jpg --ocr_mode cnn --conf 0.25 --save
```

Debug artifacts:

```bash
python -m src.main --image data/samples/car.jpg --ocr_mode cnn --save --debug
```

Custom output directory:

```bash
python -m src.main --image data/samples/car.jpg --save --output_dir runs/demo_01
```

## Streamlit UI

Run the web demo:

```bash
streamlit run app.py
```

The UI supports:

- image upload
- video upload
- optional webcam inference
- OCR mode selection
- confidence threshold tuning
- frame skipping for videos
- debug artifact generation
- annotated output and CSV downloads

## FastAPI API

Start the API server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Image prediction:

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@data/samples/car.jpg" \
  -F "ocr_mode=tesseract" \
  -F "confidence_threshold=0.25"
```

Example response fields:

```json
{
  "source": "car",
  "source_type": "image",
  "ocr_mode": "tesseract",
  "processing_time": 0.74,
  "plate_count": 1,
  "results": [
    {
      "detected_text": "MH12AB1234",
      "raw_text": "MH12A81234",
      "detection_confidence": 0.94,
      "ocr_confidence": 0.86,
      "combined_confidence": 0.90,
      "bbox": {
        "x1": 250,
        "y1": 310,
        "x2": 530,
        "y2": 385
      },
      "outputs": {
        "crop": "outputs/crops/...",
        "processed": "outputs/processed/...",
        "annotated": "outputs/annotated/..."
      }
    }
  ]
}
```

## Docker Usage

Build and run both API and Streamlit:

```bash
docker compose up --build
```

Services:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
```

Run API only:

```bash
docker build -t alpr-pro .
docker run --rm -p 8000:8000 \
  -e TESSERACT_CMD=/usr/bin/tesseract \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  -v ./data:/app/data \
  alpr-pro
```

## CNN Training

Prepare the dataset:

```text
data/chars/
  0/
  1/
  ...
  9/
  A/
  B/
  ...
  Z/
```

Train the CNN recognizer:

```bash
python -m src.train_cnn \
  --dataset_dir data/chars \
  --epochs 20 \
  --batch_size 64 \
  --lr 0.001 \
  --image_size 32 \
  --output_model_path models/cnn/character_cnn.pt
```

Training outputs:

```text
models/cnn/character_cnn.pt
models/cnn/character_cnn_final.pt
models/cnn/training_history.json
models/cnn/training_curves.png
```

## Sample Outputs

When `--save` is enabled, the system writes:

- annotated images or videos to `outputs/annotated/`
- cropped license plates to `outputs/crops/`
- processed plate crops to `outputs/processed/`
- segmented character images to `outputs/segmented/`
- CSV inference logs to `outputs/logs/`
- debug preprocessing steps to `outputs/debug/` when debug mode is enabled

Expected result:

- bounding boxes around detected plates
- recognized plate text rendered on the output image or video
- confidence scores for detection and OCR
- stable text across repeated video/webcam frames

## Configuration

Common environment variables:

```text
ALPR_YOLO_MODEL
ALPR_CNN_MODEL
ALPR_OUTPUT_DIR
TESSERACT_CMD
ALPR_OCR_MODE
ALPR_CONFIDENCE
ALPR_IOU
ALPR_FRAME_SKIP
ALPR_DEBUG
ALPR_MAX_VIDEO_FRAMES
```

Defaults are defined in `src/config.py`.

## Limitations

- Detection quality depends on the YOLO model trained for license plates.
- CNN OCR quality depends on the size and quality of the character dataset.
- Regex validation patterns may need tuning for different countries or plate formats.
- Low light, glare, motion blur, occlusion, and extreme angles can reduce OCR accuracy.
- Webcam performance depends on CPU/GPU capability and camera quality.

## Future Improvements

- Add ONNX/TensorRT export for faster inference.
- Add GPU-specific Docker image.
- Add ByteTrack or DeepSORT for stronger video tracking.
- Add country-specific plate format plugins.
- Add batch processing mode for folders.
- Add database-backed inference history.
- Add automated evaluation metrics for detection and OCR accuracy.

## Resume-Ready Bullet Points

- Built an end-to-end Automatic License Plate Recognition system using YOLO, OpenCV, and OCR techniques for image, video, and webcam inference.
- Developed real-time plate recognition with frame stabilization, confidence scoring, OCR cleanup, and regex-based validation.
- Implemented a dual OCR pipeline using Tesseract and CNN-based character recognition to support both baseline and trainable OCR workflows.
- Designed modular preprocessing, character segmentation, post-processing, tracking, and training components for a production-style computer vision pipeline.
- Built a polished Streamlit UI for interactive demos with upload support, result visualization, and downloadable annotated outputs.
- Exposed ALPR inference through a FastAPI backend with structured JSON responses, file upload handling, and production-ready error handling.
- Containerized the application with Docker and Docker Compose, including Tesseract and OpenCV runtime dependencies for portable deployment.
- Implemented artifact exports including cropped plates, processed crops, segmented characters, annotated media, CSV logs, and debug preprocessing steps.

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- pytesseract / Tesseract OCR
- PyTorch
- FastAPI
- Streamlit
- pandas / NumPy / Pillow
- Docker
