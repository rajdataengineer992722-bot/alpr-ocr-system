"""Core ALPR execution flow and CLI entry point."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import get_settings
from src.detect_plate import PlateDetection, YOLOPlateDetector
from src.logger import logger, setup_logger
from src.postprocess import PlatePostProcessor
from src.preprocess import PlatePreprocessor
from src.recognize_cnn import CNNRecognizer
from src.recognize_tesseract import TesseractRecognizer
from src.segment_characters import CharacterSegmenter
from src.tracker import TemporalPlateTracker
from src.utils import draw_plate_annotations, read_image, save_csv, save_image, save_json, unique_name


@dataclass
class EngineArtifacts:
    """Saved artifact paths for a single detected plate."""

    crop_path: Optional[str] = None
    processed_path: Optional[str] = None
    segmented_paths: List[str] = field(default_factory=list)
    debug_steps: Dict[str, str] = field(default_factory=dict)


class ALPRSystem:
    """Production-style ALPR engine shared by CLI, API, and Streamlit."""

    def __init__(self, output_dir: str | Path | None = None, debug: bool = False) -> None:
        self.settings = get_settings()
        self.debug = debug
        self.output_root = Path(output_dir).resolve() if output_dir else self.settings.output_dir.resolve()
        self.output_dirs = self._build_output_dirs(self.output_root)
        self._ensure_output_dirs()

        # Load heavyweight components once.
        self.detector = YOLOPlateDetector()
        self.preprocessor = PlatePreprocessor()
        self.segmenter = CharacterSegmenter()
        self.tesseract = TesseractRecognizer()
        self.cnn = CNNRecognizer()
        self.postprocessor = PlatePostProcessor()
        self.tracker = TemporalPlateTracker()

    @staticmethod
    def _build_output_dirs(root: Path) -> Dict[str, Path]:
        """Create the output directory mapping used by the runtime."""

        return {
            "root": root,
            "crops": root / "crops",
            "processed": root / "processed",
            "segmented": root / "segmented",
            "annotated": root / "annotated",
            "logs": root / "logs",
            "api_results": root / "api_results",
            "debug": root / "debug",
        }

    def _ensure_output_dirs(self) -> None:
        """Ensure runtime output directories exist."""

        for path in self.output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def _save_plate_artifacts(
        self,
        plate_crop: np.ndarray,
        processed_plate: np.ndarray,
        segmented_chars: List[dict],
        source_id: str,
        frame_index: int,
        plate_index: int,
    ) -> EngineArtifacts:
        """Persist plate-level artifacts for debugging and review."""

        prefix = f"{source_id}_f{frame_index:05d}_p{plate_index:02d}"
        artifacts = EngineArtifacts(
            crop_path=save_image(plate_crop, self.output_dirs["crops"] / f"{prefix}_crop.png"),
            processed_path=save_image(processed_plate, self.output_dirs["processed"] / f"{prefix}_processed.png"),
        )

        for idx, segment in enumerate(segmented_chars):
            if "image" not in segment:
                continue
            path = self.output_dirs["segmented"] / f"{prefix}_char_{idx:02d}.png"
            artifacts.segmented_paths.append(save_image(segment["image"], path))

        if self.debug:
            debug_dir = self.output_dirs["debug"] / source_id
            artifacts.debug_steps = self.preprocessor.save_debug_steps(debug_dir, prefix)

        return artifacts

    def _score_ocr_candidate(self, text: str, confidence: float) -> tuple[float, float, int]:
        """Rank OCR candidates by confidence first, then by usable text length."""

        normalized_text = self.postprocessor.clean_text(text)
        return (float(confidence), min(len(normalized_text), 10), len(normalized_text))

    def _run_tesseract_ocr(self, plate_crop: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Try a few OCR-friendly variants and keep the strongest Tesseract result."""

        grayscale = plate_crop if plate_crop.ndim == 2 else cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            grayscale,
            (360, max(1, int(grayscale.shape[0] * (360 / max(1, grayscale.shape[1]))))),
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
        processed = self.preprocessor.for_tesseract(plate_crop)

        variants = [
            ("enhanced", enhanced),
            ("processed", processed),
            ("resized_gray", resized),
        ]

        best_text = ""
        best_confidence = 0.0
        best_image = processed
        best_score = (-1.0, 0.0, 0)

        for variant_name, variant_image in variants:
            text, confidence = self.tesseract.recognize(variant_image)
            score = self._score_ocr_candidate(text, confidence)
            logger.debug(
                "Tesseract variant={} text='{}' confidence={:.4f}",
                variant_name,
                text,
                confidence,
            )
            if score > best_score:
                best_text = text
                best_confidence = confidence
                best_image = variant_image
                best_score = score

        return best_text, best_confidence, best_image

    def process_plate(
        self,
        detection: PlateDetection,
        ocr_mode: str,
        source_id: str,
        frame_index: int,
        plate_index: int,
        save_outputs: bool,
    ) -> dict:
        """Process a single detected plate crop end to end."""

        plate_crop = detection.crop
        raw_text = ""
        final_text = ""
        is_valid = False
        ocr_confidence = 0.0
        segmented_chars: List[dict] = []
        processed_plate = plate_crop
        artifacts = EngineArtifacts()

        if ocr_mode == "cnn":
            processed_plate = self.preprocessor.for_segmentation(plate_crop)

            # Segment explicitly here so the entry point controls failure handling,
            # even though the CNN recognizer also performs segmentation internally.
            candidate_segments = self.segmenter.segment(processed_plate)
            if not candidate_segments:
                logger.warning(
                    "No character segments found for frame {} plate {} in CNN mode.",
                    frame_index,
                    plate_index,
                )
            raw_text, ocr_confidence, segmented_chars = self.cnn.recognize(processed_plate)

            if not raw_text:
                logger.warning(
                    "CNN OCR produced no text for frame {} plate {}. Falling back to Tesseract.",
                    frame_index,
                    plate_index,
                )
                raw_text, ocr_confidence, processed_plate = self._run_tesseract_ocr(plate_crop)
                segmented_chars = []
        else:
            raw_text, ocr_confidence, processed_plate = self._run_tesseract_ocr(plate_crop)

        final_text, is_valid, candidates = self.postprocessor.process(raw_text)
        combined_confidence = round((float(detection.confidence) + float(ocr_confidence)) / 2.0, 4)
        track = self.tracker.update(detection.bbox, final_text, combined_confidence, frame_index)

        if save_outputs:
            artifacts = self._save_plate_artifacts(
                plate_crop=plate_crop,
                processed_plate=processed_plate,
                segmented_chars=segmented_chars,
                source_id=source_id,
                frame_index=frame_index,
                plate_index=plate_index,
            )

        result = {
            "track_id": track.track_id,
            "detected_text": final_text,
            "raw_text": raw_text,
            "ocr_mode": ocr_mode,
            "detection_confidence": round(float(detection.confidence), 4),
            "ocr_confidence": round(float(ocr_confidence), 4),
            "combined_confidence": combined_confidence,
            "stable_confidence": round(track.stable_confidence(), 4),
            "bbox": detection.bbox,
            "class_name": detection.class_name,
            "is_valid": is_valid,
            "stable_text": track.stable_text(),
            "postprocess_candidates": candidates,
            "frame_index": frame_index,
            "outputs": {
                "crop": artifacts.crop_path,
                "processed": artifacts.processed_path,
                "segmented": artifacts.segmented_paths,
                "debug_steps": artifacts.debug_steps,
            },
        }

        if not result["detected_text"]:
            logger.info(
                "OCR produced no final text for frame {} plate {}. Detection confidence={:.3f}",
                frame_index,
                plate_index,
                detection.confidence,
            )

        return result

    def run_on_frame(
        self,
        frame: np.ndarray,
        source_id: str,
        frame_index: int = 0,
        ocr_mode: str = "tesseract",
        conf_threshold: float | None = None,
        save_outputs: bool = False,
    ) -> tuple[List[dict], np.ndarray]:
        """Process all plate detections in a single frame."""

        detections = self.detector.detect(frame, conf_threshold=conf_threshold)
        if not detections:
            logger.debug("No plates detected on frame {}", frame_index)
            return [], frame.copy()

        results: List[dict] = []
        for plate_index, detection in enumerate(detections):
            results.append(
                self.process_plate(
                    detection=detection,
                    ocr_mode=ocr_mode,
                    source_id=source_id,
                    frame_index=frame_index,
                    plate_index=plate_index,
                    save_outputs=save_outputs,
                )
            )

        annotated = draw_plate_annotations(frame, results)
        return results, annotated

    def _flatten_result(self, result: dict, source: str) -> dict:
        """Flatten a result for CSV export."""

        bbox = result["bbox"]
        return {
            "source": source,
            "track_id": result.get("track_id"),
            "class_name": result.get("class_name"),
            "detected_text": result.get("detected_text"),
            "stable_text": result.get("stable_text"),
            "raw_text": result.get("raw_text"),
            "ocr_mode": result.get("ocr_mode"),
            "detection_confidence": result.get("detection_confidence"),
            "ocr_confidence": result.get("ocr_confidence"),
            "combined_confidence": result.get("combined_confidence"),
            "stable_confidence": result.get("stable_confidence"),
            "is_valid": result.get("is_valid"),
            "frame_index": result.get("frame_index"),
            "timestamp": result.get("timestamp"),
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "crop_path": result.get("outputs", {}).get("crop"),
            "processed_path": result.get("outputs", {}).get("processed"),
            "segmented_count": len(result.get("outputs", {}).get("segmented", [])),
        }

    def _finalize_response(
        self,
        source: str,
        source_type: str,
        ocr_mode: str,
        results: List[dict],
        processing_time: float,
        annotated_path: Optional[str],
        save_outputs: bool,
        precomputed_rows: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Build the normalized response payload."""

        rows = precomputed_rows if precomputed_rows is not None else [self._flatten_result(item, source) for item in results]
        csv_path = None
        json_path = None

        if save_outputs:
            base_name = unique_name(f"{source}_{ocr_mode}", "")
            csv_path = save_csv(rows, self.output_dirs["logs"] / f"{base_name}.csv")
            if self.settings.save_json_results:
                json_path = save_json(
                    {
                        "source": source,
                        "source_type": source_type,
                        "ocr_mode": ocr_mode,
                        "processing_time": round(processing_time, 4),
                        "plate_count": len(results),
                        "results": rows,
                        "annotated_path": annotated_path,
                        "csv_path": csv_path,
                    },
                    self.output_dirs["api_results"] / f"{base_name}.json",
                )

        normalized_results: List[dict] = []
        for item in results:
            normalized_results.append(
                {
                    "track_id": item.get("track_id"),
                    "detected_text": item.get("detected_text", ""),
                    "raw_text": item.get("raw_text", ""),
                    "ocr_mode": item.get("ocr_mode", ocr_mode),
                    "detection_confidence": item.get("detection_confidence", 0.0),
                    "ocr_confidence": item.get("ocr_confidence", 0.0),
                    "combined_confidence": item.get("combined_confidence", 0.0),
                    "stable_confidence": item.get("stable_confidence", 0.0),
                    "bbox": {
                        "x1": item["bbox"][0],
                        "y1": item["bbox"][1],
                        "x2": item["bbox"][2],
                        "y2": item["bbox"][3],
                    },
                    "is_valid": item.get("is_valid", False),
                    "stable_text": item.get("stable_text"),
                    "frame_index": item.get("frame_index"),
                    "timestamp": item.get("timestamp"),
                    "outputs": {
                        "crop": item.get("outputs", {}).get("crop"),
                        "processed": item.get("outputs", {}).get("processed"),
                        "segmented": item.get("outputs", {}).get("segmented", []),
                        "annotated": annotated_path,
                        "debug_steps": item.get("outputs", {}).get("debug_steps", {}),
                    },
                }
            )

        return {
            "source": source,
            "source_type": source_type,
            "ocr_mode": ocr_mode,
            "processing_time": round(processing_time, 4),
            "plate_count": len(results),
            "results": normalized_results,
            "csv_path": csv_path,
            "json_path": json_path,
            "annotated_path": annotated_path,
        }

    def process_image(
        self,
        image: str | Path | np.ndarray,
        source_name: Optional[str] = None,
        ocr_mode: str = "tesseract",
        conf_threshold: float | None = None,
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        """Run ALPR on a single image."""

        start = time.perf_counter()
        if isinstance(image, np.ndarray):
            frame = image.copy()
            source = source_name or "memory_image"
        else:
            frame = read_image(image)
            source = source_name or Path(image).stem

        results, annotated = self.run_on_frame(
            frame=frame,
            source_id=source,
            frame_index=0,
            ocr_mode=ocr_mode,
            conf_threshold=conf_threshold,
            save_outputs=save_outputs,
        )

        annotated_path = None
        if save_outputs:
            annotated_path = save_image(annotated, self.output_dirs["annotated"] / unique_name(source, ".png"))

        payload = self._finalize_response(
            source=source,
            source_type="image",
            ocr_mode=ocr_mode,
            results=results,
            processing_time=time.perf_counter() - start,
            annotated_path=annotated_path,
            save_outputs=save_outputs,
        )
        payload["annotated_image"] = annotated
        return payload

    def process_video_stream(
        self,
        video_source: str | int,
        source_name: Optional[str] = None,
        ocr_mode: str = "tesseract",
        conf_threshold: float | None = None,
        save_outputs: bool = True,
        show: bool = False,
        frame_skip: int = 1,
    ) -> Dict[str, Any]:
        """Run ALPR on a video file or webcam stream."""

        start = time.perf_counter()
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            if isinstance(video_source, int):
                raise ValueError(f"Unable to open webcam index {video_source}.")
            raise ValueError(f"Unable to open video source: {video_source}")

        source = source_name or (f"webcam_{video_source}" if isinstance(video_source, int) else Path(str(video_source)).stem)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        max_frames = self.settings.max_video_frames

        writer = None
        annotated_path = None
        if save_outputs:
            annotated_path = str((self.output_dirs["annotated"] / unique_name(source, ".mp4")).resolve())
            writer = cv2.VideoWriter(
                annotated_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        all_rows: List[dict] = []
        latest_results: List[dict] = []
        frame_index = 0
        processed_frames = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_skip > 1 and frame_index % frame_skip != 0:
                    frame_index += 1
                    continue

                results, annotated = self.run_on_frame(
                    frame=frame,
                    source_id=source,
                    frame_index=frame_index,
                    ocr_mode=ocr_mode,
                    conf_threshold=conf_threshold,
                    save_outputs=save_outputs,
                )
                latest_results = results
                timestamp_sec = frame_index / fps if fps else None

                for item in results:
                    item["timestamp"] = round(timestamp_sec, 4) if timestamp_sec is not None else None
                    all_rows.append(self._flatten_result(item, source))

                if writer is not None:
                    writer.write(annotated)

                if show:
                    cv2.imshow("ALPR Pro", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Video preview stopped by user.")
                        break

                frame_index += 1
                processed_frames += 1
                if max_frames and processed_frames >= max_frames:
                    logger.info("Stopping stream after configured max_video_frames={}", max_frames)
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        payload = self._finalize_response(
            source=source,
            source_type="webcam" if isinstance(video_source, int) else "video",
            ocr_mode=ocr_mode,
            results=latest_results,
            processing_time=time.perf_counter() - start,
            annotated_path=annotated_path,
            save_outputs=save_outputs,
            precomputed_rows=all_rows,
        )
        payload["processed_frames"] = processed_frames
        return payload

    def process_video(
        self,
        video_source: str | int,
        source_name: Optional[str] = None,
        ocr_mode: str = "tesseract",
        conf_threshold: float | None = None,
        save_outputs: bool = True,
        show: bool = False,
        frame_skip: int = 1,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper used by the API and Streamlit app."""

        return self.process_video_stream(
            video_source=video_source,
            source_name=source_name,
            ocr_mode=ocr_mode,
            conf_threshold=conf_threshold,
            save_outputs=save_outputs,
            show=show,
            frame_skip=frame_skip,
        )


def build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description="ALPR Pro CLI")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to an input image")
    input_group.add_argument("--video", help="Path to an input video")
    input_group.add_argument("--webcam", type=int, help="Webcam device index")

    parser.add_argument("--ocr_mode", choices=["tesseract", "cnn"], default=settings.default_ocr_mode)
    parser.add_argument("--save", action="store_true", help="Save annotated output and artifacts")
    parser.add_argument("--show", action="store_true", help="Show annotated preview while processing")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame for video/webcam")
    parser.add_argument("--conf", type=float, default=settings.default_confidence, help="YOLO detection confidence threshold")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the default outputs directory")
    parser.add_argument("--debug", action="store_true", help="Save preprocessing debug images and verbose runtime traces")
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate input selection and CLI values."""

    if args.image and not Path(args.image).exists():
        raise SystemExit(f"Image path does not exist: {args.image}")
    if args.video and not Path(args.video).exists():
        raise SystemExit(f"Video path does not exist: {args.video}")
    if args.frame_skip < 1:
        raise SystemExit("--frame_skip must be >= 1")
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")


def _set_log_level(debug: bool) -> None:
    """Enable debug logging when requested."""

    if debug:
        logger.remove()
        from sys import stdout

        logger.add(stdout, level="DEBUG", colorize=True, enqueue=True)
        logger.add(
            get_settings().output_subdirs["logs"] / "cli.log",
            level="DEBUG",
            rotation="10 MB",
            retention="10 days",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )


def _log_summary(payload: Dict[str, Any], mode_name: str) -> None:
    """Log a readable summary after inference."""

    logger.info("{} inference complete.", mode_name)
    logger.info("Plates detected: {}", payload.get("plate_count", 0))
    if payload.get("annotated_path"):
        logger.info("Annotated output: {}", payload["annotated_path"])
    if payload.get("csv_path"):
        logger.info("CSV log: {}", payload["csv_path"])
    debug_payload = {key: value for key, value in payload.items() if key != "annotated_image"}
    logger.debug("Full payload:\n{}", json.dumps(debug_payload, indent=2, default=str))


def main() -> None:
    """CLI entry point for ALPR Pro."""

    setup_logger("cli.log")
    parser = build_argparser()
    args = parser.parse_args()
    _validate_cli_args(args)
    _set_log_level(args.debug)

    try:
        system = ALPRSystem(output_dir=args.output_dir, debug=args.debug)

        if args.image:
            payload = system.process_image(
                image=args.image,
                ocr_mode=args.ocr_mode,
                conf_threshold=args.conf,
                save_outputs=args.save,
            )
            _log_summary(payload, "Image")
            if args.show and "annotated_image" in payload:
                cv2.imshow("ALPR Pro", payload["annotated_image"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return

        if args.video:
            payload = system.process_video_stream(
                video_source=args.video,
                ocr_mode=args.ocr_mode,
                conf_threshold=args.conf,
                save_outputs=args.save,
                show=args.show,
                frame_skip=args.frame_skip,
            )
            _log_summary(payload, "Video")
            return

        payload = system.process_video_stream(
            video_source=args.webcam,
            ocr_mode=args.ocr_mode,
            conf_threshold=args.conf,
            save_outputs=args.save,
            show=args.show,
            frame_skip=args.frame_skip,
        )
        _log_summary(payload, "Webcam")

    except FileNotFoundError as exc:
        logger.error("Missing required file: {}", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("Invalid runtime input: {}", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        logger.exception("ALPR pipeline failed: {}", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
