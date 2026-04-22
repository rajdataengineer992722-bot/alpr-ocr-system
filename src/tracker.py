"""Lightweight plate tracking and temporal OCR stabilization."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional

from src.config import get_settings
from src.logger import logger
from src.utils import iou


BBox = tuple[int, int, int, int]


def _center(box: BBox) -> tuple[float, float]:
    """Return the center point of a bounding box."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance(box_a: BBox, box_b: BBox) -> float:
    """Compute Euclidean distance between box centers."""

    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _box_diagonal(box: BBox) -> float:
    """Compute diagonal length for scale-aware distance matching."""

    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return (width**2 + height**2) ** 0.5


@dataclass
class PlateTrack:
    """State for one tracked license plate."""

    track_id: int
    bbox: BBox
    last_seen: int
    text_history: Deque[str] = field(default_factory=deque)
    confidence_history: Deque[float] = field(default_factory=deque)
    missed_frames: int = 0
    hits: int = 1

    def update(
        self,
        bbox: BBox,
        text: str,
        confidence: float,
        frame_index: int,
        history_size: int,
    ) -> None:
        """Update track state with a matched detection."""

        self.bbox = bbox
        self.last_seen = frame_index
        self.missed_frames = 0
        self.hits += 1

        cleaned_text = (text or "").strip().upper()
        if cleaned_text:
            self.text_history.append(cleaned_text)
            self.confidence_history.append(float(confidence or 0.0))

        while len(self.text_history) > history_size:
            self.text_history.popleft()
        while len(self.confidence_history) > history_size:
            self.confidence_history.popleft()

    def mark_missed(self) -> None:
        """Increment missing-frame count when no detection matched this track."""

        self.missed_frames += 1

    def stable_text(self) -> str:
        """Return confidence-aware majority-vote OCR text."""

        if not self.text_history:
            return ""

        weighted_votes: Dict[str, float] = defaultdict(float)
        plain_counts = Counter(self.text_history)
        for text, confidence in zip(self.text_history, self.confidence_history):
            weighted_votes[text] += max(0.01, float(confidence))

        # Sort by weighted confidence first, then frequency, then recency.
        recency_rank = {text: idx for idx, text in enumerate(self.text_history)}
        return max(
            weighted_votes,
            key=lambda item: (weighted_votes[item], plain_counts[item], recency_rank[item]),
        )

    def stable_confidence(self) -> float:
        """Return the average confidence of observations matching the stable text."""

        stable = self.stable_text()
        if not stable:
            return 0.0

        matching = [
            confidence
            for text, confidence in zip(self.text_history, self.confidence_history)
            if text == stable
        ]
        return float(sum(matching) / len(matching)) if matching else 0.0

    def as_dict(self) -> dict:
        """Serialize track state for overlays, logs, or API responses."""

        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "stable_text": self.stable_text(),
            "stable_confidence": round(self.stable_confidence(), 4),
            "last_seen": self.last_seen,
            "missed_frames": self.missed_frames,
            "hits": self.hits,
        }


class PlateTracker:
    """Simple real-time tracker for stabilizing OCR across frames."""

    def __init__(
        self,
        iou_threshold: float | None = None,
        max_center_distance_ratio: float = 0.75,
        max_missed_frames: int | None = None,
        history_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.tracker_iou_threshold
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_missed_frames = max_missed_frames if max_missed_frames is not None else settings.tracker_max_missing
        self.history_size = history_size if history_size is not None else settings.tracker_vote_window
        self.tracks: Dict[int, PlateTrack] = {}
        self.next_track_id = 1

    def _match_score(self, track: PlateTrack, bbox: BBox) -> float:
        """Score how well a detection matches an existing track."""

        overlap = iou(track.bbox, bbox)
        if overlap >= self.iou_threshold:
            return 1.0 + overlap

        distance = _center_distance(track.bbox, bbox)
        scale = max(_box_diagonal(track.bbox), _box_diagonal(bbox))
        normalized_distance = distance / max(1.0, scale)
        if normalized_distance <= self.max_center_distance_ratio:
            return 1.0 - normalized_distance

        return -1.0

    def _find_best_track(self, bbox: BBox, used_track_ids: set[int]) -> Optional[PlateTrack]:
        """Find the best unmatched track for a detection."""

        best_track: Optional[PlateTrack] = None
        best_score = -1.0

        for track in self.tracks.values():
            if track.track_id in used_track_ids:
                continue
            score = self._match_score(track, bbox)
            if score > best_score:
                best_score = score
                best_track = track

        return best_track if best_score >= 0 else None

    def _create_track(self, bbox: BBox, text: str, confidence: float, frame_index: int) -> PlateTrack:
        """Create and initialize a new track."""

        track = PlateTrack(track_id=self.next_track_id, bbox=bbox, last_seen=frame_index)
        self.next_track_id += 1
        track.update(bbox, text, confidence, frame_index, self.history_size)
        self.tracks[track.track_id] = track
        logger.debug("Created plate track {}", track.track_id)
        return track

    def _expire_old_tracks(self) -> None:
        """Remove tracks that have been missing for too long."""

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in expired:
            logger.debug("Expiring plate track {}", track_id)
            self.tracks.pop(track_id, None)

    def update(self, detections: Iterable[dict], frame_index: int) -> List[dict]:
        """Update tracker with current-frame detections.

        Each detection dictionary may contain:
        - bbox: tuple/list (x1, y1, x2, y2)
        - detected_text or text
        - combined_confidence, confidence, or ocr_confidence

        Returns copies of the detections with track_id, stable_text, and
        stable_confidence attached.
        """

        detections = list(detections)
        used_track_ids: set[int] = set()
        stabilized: List[dict] = []

        for detection in detections:
            bbox_raw = detection.get("bbox")
            if bbox_raw is None:
                logger.debug("Skipping detection without bbox during tracking.")
                continue

            bbox: BBox = tuple(map(int, bbox_raw))  # type: ignore[assignment]
            text = detection.get("detected_text") or detection.get("text") or ""
            confidence = (
                detection.get("combined_confidence")
                if detection.get("combined_confidence") is not None
                else detection.get("confidence", detection.get("ocr_confidence", 0.0))
            )

            track = self._find_best_track(bbox, used_track_ids)
            if track is None:
                track = self._create_track(bbox, text, float(confidence or 0.0), frame_index)
            else:
                track.update(bbox, text, float(confidence or 0.0), frame_index, self.history_size)

            used_track_ids.add(track.track_id)
            enriched = dict(detection)
            enriched.update(
                {
                    "track_id": track.track_id,
                    "stable_text": track.stable_text(),
                    "stable_confidence": round(track.stable_confidence(), 4),
                }
            )
            stabilized.append(enriched)

        for track in self.tracks.values():
            if track.track_id not in used_track_ids:
                track.mark_missed()

        self._expire_old_tracks()
        return stabilized

    def update_one(self, bbox: BBox, text: str, confidence: float, frame_index: int) -> PlateTrack:
        """Compatibility helper for code paths that update one detection at a time."""

        track = self._find_best_track(bbox, used_track_ids=set())
        if track is None:
            track = self._create_track(bbox, text, float(confidence or 0.0), frame_index)
        else:
            track.update(bbox, text, float(confidence or 0.0), frame_index, self.history_size)

        # The single-update API is called once per detection by main.py. Avoid
        # marking other tracks as missed on each per-plate call; expire only by
        # frame age so multi-plate frames remain stable.
        expired = [
            track_id
            for track_id, candidate in self.tracks.items()
            if frame_index - candidate.last_seen > self.max_missed_frames
        ]
        for track_id in expired:
            logger.debug("Expiring plate track {}", track_id)
            self.tracks.pop(track_id, None)
        return track

    def active_tracks(self) -> List[dict]:
        """Return active track summaries."""

        return [track.as_dict() for track in self.tracks.values()]

    def reset(self) -> None:
        """Clear all active tracks."""

        self.tracks.clear()
        self.next_track_id = 1


class TemporalPlateTracker(PlateTracker):
    """Backward-compatible tracker name used by the existing ALPR pipeline."""

    def update(  # type: ignore[override]
        self,
        bbox_or_detections: BBox | Iterable[dict],
        text: str | None = None,
        confidence: float | None = None,
        frame_index: int = 0,
    ) -> PlateTrack | List[dict]:
        """Support both old single-detection and new batch update APIs."""

        if isinstance(bbox_or_detections, tuple):
            return self.update_one(
                bbox=bbox_or_detections,
                text=text or "",
                confidence=float(confidence or 0.0),
                frame_index=frame_index,
            )

        return super().update(bbox_or_detections, frame_index=frame_index)
