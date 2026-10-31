"""Decide qué caja es el docente en cada frame."""

from dataclasses import dataclass

from espejo.config import TeacherConfig
from espejo.vision.geometry import BoundingBox, Point
from espejo.vision.schemas import Detection, FrameDetections


@dataclass(frozen=True)
class TeacherSelection:
    """Caja del docente en un frame. interpolated = se rellenó un hueco corto con la última."""

    box: BoundingBox | None
    interpolated: bool = False


class TeacherTracker:
    """En cada frame, el docente es la caja "teacher" más confiable que supera teacher_min_conf.

    Antes se elegía el track que más aparecía; en una clase real eso eligió a un alumno quieto,
    porque el docente camina y su track se corta muchas veces. Si dos cajas tienen confianza
    parecida, gana la más cercana a la última posición conocida del docente.
    """

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config

    def select(self, frames: list[FrameDetections]) -> list[TeacherSelection]:
        selections = []
        last_box = None
        for frame in frames:
            box = self.choose_box(frame.teacher_detections, last_box)
            selections.append(TeacherSelection(box))
            if box is not None:
                last_box = box
        return fill_short_gaps(selections, frames, self.config.gap_fill_s)

    def choose_box(
        self, detections: list[Detection], last_box: BoundingBox | None
    ) -> BoundingBox | None:
        candidates = confident_detections(detections, self.config.teacher_min_conf)
        if not candidates:
            return None

        best = most_confident(candidates)
        if last_box is None:
            return best.box

        contenders = []
        for candidate in candidates:
            if best.confidence - candidate.confidence <= self.config.similar_conf_margin:
                contenders.append(candidate)
        return nearest_to(contenders, last_box).box


def confident_detections(detections: list[Detection], min_confidence: float) -> list[Detection]:
    selected = []
    for detection in detections:
        if detection.confidence >= min_confidence:
            selected.append(detection)
    return selected


def most_confident(detections: list[Detection]) -> Detection:
    best = detections[0]
    for detection in detections[1:]:
        if detection.confidence > best.confidence:
            best = detection
    return best


def foot(box: BoundingBox) -> Point:
    foot_x, foot_y = box.foot_point()
    return Point(foot_x, foot_y)


def nearest_to(detections: list[Detection], reference: BoundingBox) -> Detection:
    reference_point = foot(reference)
    nearest = detections[0]
    for detection in detections[1:]:
        distance = foot(detection.box).distance_to(reference_point)
        if distance < foot(nearest.box).distance_to(reference_point):
            nearest = detection
    return nearest


def fill_short_gaps(
    selections: list[TeacherSelection], frames: list[FrameDetections], gap_fill_s: float
) -> list[TeacherSelection]:
    """Rellena con la última caja los huecos más cortos que gap_fill_s entre dos apariciones.

    Un hueco al final del video (el docente no vuelve) no se rellena: no sabemos dónde está.
    """
    filled = list(selections)
    last_seen_index = None
    for index, selection in enumerate(selections):
        if selection.box is None:
            continue
        if last_seen_index is not None and index - last_seen_index > 1:
            gap_s = frames[index].timestamp_s - frames[last_seen_index].timestamp_s
            if gap_s < gap_fill_s:
                last_box = selections[last_seen_index].box
                for missing_index in range(last_seen_index + 1, index):
                    filled[missing_index] = TeacherSelection(last_box, interpolated=True)
        last_seen_index = index
    return filled
