"""Estado del docente en un frame y eventos de alumnos, a partir de las detecciones."""

from espejo.config import BehaviorConfig
from espejo.vision.geometry import BoundingBox, iou
from espejo.vision.schemas import Behavior, Detection


def overlaps_teacher(
    detection: Detection, teacher_box: BoundingBox, config: BehaviorConfig
) -> bool:
    return iou(detection.box, teacher_box) >= config.behavior_iou_threshold


def has_overlapping_class(
    detections: list[Detection],
    class_name: str,
    teacher_box: BoundingBox,
    config: BehaviorConfig,
) -> bool:
    for detection in detections:
        if detection.class_name == class_name and overlaps_teacher(detection, teacher_box, config):
            return True
    return False


def assign_teacher_state(
    detections: list[Detection],
    teacher_box: BoundingBox | None,
    config: BehaviorConfig,
) -> Behavior:
    """Exactamente un estado por frame, con prioridad board_writing > guide > other.

    Solo cuentan las conductas del docente que se superponen con su caja. answer y
    stage_interaction son de alumnos y nunca se asignan al docente, aunque se superpongan.
    """
    if teacher_box is None:
        return Behavior.NOT_VISIBLE

    if has_overlapping_class(detections, config.board_writing_class, teacher_box, config):
        return Behavior.BOARD_WRITING
    if has_overlapping_class(detections, config.guide_class, teacher_box, config):
        return Behavior.GUIDE
    return Behavior.OTHER


def has_stage_interaction(detections: list[Detection], config: BehaviorConfig) -> bool:
    """Un alumno al frente es un evento del aula: no depende de dónde esté el docente."""
    for detection in detections:
        if detection.class_name == config.stage_interaction_class:
            return True
    return False
