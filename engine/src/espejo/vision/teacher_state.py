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
    min_confidence: float,
) -> bool:
    for detection in detections:
        if detection.class_name != class_name or detection.confidence < min_confidence:
            continue
        if overlaps_teacher(detection, teacher_box, config):
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
    Cada una exige su confianza mínima (board_writing_min_conf y guide_min_conf): con poca
    confianza suele ser el docente parado, sin escribir ni revisar el trabajo de nadie.
    """
    if teacher_box is None:
        return Behavior.NOT_VISIBLE

    board_class = config.board_writing_class
    board_min_conf = config.board_writing_min_conf
    if has_overlapping_class(detections, board_class, teacher_box, config, board_min_conf):
        return Behavior.BOARD_WRITING
    if has_overlapping_class(
        detections, config.guide_class, teacher_box, config, config.guide_min_conf
    ):
        return Behavior.GUIDE
    return Behavior.OTHER


def reportable_behaviors(detections: list[Detection], config: BehaviorConfig) -> list[Detection]:
    """Sin las clases ignoradas (p. ej. answer) ni las conductas del docente bajo su umbral."""
    min_conf_by_class = {
        config.board_writing_class: config.board_writing_min_conf,
        config.guide_class: config.guide_min_conf,
    }
    kept = []
    for detection in detections:
        if detection.class_name in config.ignored_classes:
            continue
        if detection.confidence < min_conf_by_class.get(detection.class_name, 0.0):
            continue
        kept.append(detection)
    return kept


def has_stage_interaction(detections: list[Detection], config: BehaviorConfig) -> bool:
    """Un alumno al frente es un evento del aula: no depende de dónde esté el docente."""
    for detection in detections:
        if detection.class_name == config.stage_interaction_class:
            return True
    return False
