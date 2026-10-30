"""Decide qué caja es el docente en cada frame a partir de los tracks de ByteTrack."""

import logging

from espejo.vision.geometry import BoundingBox
from espejo.vision.schemas import Detection

logger = logging.getLogger(__name__)


class TeacherTracker:
    """El docente es el track que aparece en más frames.

    El detector puede confundir por momentos a un alumno parado con el docente, pero esos
    tracks son cortos. El track dominante es casi siempre el docente real.
    """

    def __init__(self, teacher_fallback_conf: float) -> None:
        self.teacher_fallback_conf = teacher_fallback_conf

    def select_teacher_boxes(
        self, detections_per_frame: list[list[Detection]]
    ) -> list[BoundingBox | None]:
        """Devuelve una caja por frame, o None si el docente no es visible."""
        dominant_track_id = find_dominant_track_id(detections_per_frame)
        logger.info("Track dominante del docente: %s", dominant_track_id)

        teacher_boxes = []
        for detections in detections_per_frame:
            box = self.choose_teacher_box(detections, dominant_track_id)
            teacher_boxes.append(box)
        return teacher_boxes

    def choose_teacher_box(
        self, detections: list[Detection], dominant_track_id: int | None
    ) -> BoundingBox | None:
        for detection in detections:
            if dominant_track_id is not None and detection.track_id == dominant_track_id:
                return detection.box

        # Sin el track dominante (p. ej. tras una oclusión), solo aceptamos una caja muy segura
        fallback = most_confident(detections)
        if fallback is not None and fallback.confidence > self.teacher_fallback_conf:
            return fallback.box
        return None


def find_dominant_track_id(detections_per_frame: list[list[Detection]]) -> int | None:
    """Cuenta en cuántos frames aparece cada track_id y devuelve el más frecuente."""
    frame_counts: dict[int, int] = {}
    for detections in detections_per_frame:
        track_ids_in_frame = set()
        for detection in detections:
            if detection.track_id is not None:
                track_ids_in_frame.add(detection.track_id)
        for track_id in track_ids_in_frame:
            frame_counts[track_id] = frame_counts.get(track_id, 0) + 1

    if not frame_counts:
        return None

    dominant_track_id = None
    for track_id, count in frame_counts.items():
        if dominant_track_id is None or count > frame_counts[dominant_track_id]:
            dominant_track_id = track_id
    return dominant_track_id


def most_confident(detections: list[Detection]) -> Detection | None:
    best_detection = None
    for detection in detections:
        if best_detection is None or detection.confidence > best_detection.confidence:
            best_detection = detection
    return best_detection
