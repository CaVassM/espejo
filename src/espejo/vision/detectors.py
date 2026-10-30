"""Envoltorios de los modelos YOLO: docente, conductas y pose."""

import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

from espejo.config import BehaviorConfig, ModelsConfig, PoseConfig, TeacherConfig
from espejo.vision.geometry import BoundingBox, iou
from espejo.vision.schemas import (
    Behavior,
    BehaviorResult,
    Detection,
    Frame,
    Keypoint,
    PersonPose,
    PoseKeypoints,
)

logger = logging.getLogger(__name__)

# Índices de keypoints en el formato COCO que usa yolov8-pose
NOSE_INDEX = 0
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6


def load_model(weights: Path | str, missing_hint: str) -> YOLO:
    """Carga pesos YOLO con un error claro si faltan.

    Un nombre sin carpeta (p. ej. yolov8n-pose.pt) es un modelo oficial: ultralytics lo descarga.
    """
    weights_path = Path(weights)
    is_official_name = weights_path.parent == Path(".")
    if not weights_path.exists() and not is_official_name:
        raise FileNotFoundError(f"No se encontró {weights_path}. {missing_hint}")

    logger.info("Cargando modelo %s", weights_path)
    return YOLO(str(weights_path))


def detections_from_result(result: Results) -> list[Detection]:
    boxes = result.boxes
    corners_list = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()

    track_ids: list[int | None] = [None] * len(corners_list)
    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()

    detections = []
    for corners, confidence, class_id, track_id in zip(
        corners_list, confidences, class_ids, track_ids, strict=True
    ):
        class_name = result.names[int(class_id)]
        detections.append(Detection(BoundingBox(*corners), class_name, confidence, track_id))
    return detections


def filter_by_class(detections: list[Detection], class_name: str) -> list[Detection]:
    selected = []
    for detection in detections:
        if detection.class_name == class_name:
            selected.append(detection)
    return selected


class TeacherDetector:
    """Detecta y sigue al docente con ByteTrack, frame a frame."""

    def __init__(self, models_config: ModelsConfig, teacher_config: TeacherConfig) -> None:
        hint = "Entrena primero con scripts/train.py --dataset teacher"
        self.model = load_model(models_config.teacher_weights, hint)
        self.models_config = models_config
        self.teacher_config = teacher_config

    def track(self, frame: Frame) -> list[Detection]:
        # persist=True mantiene el estado del tracker entre llamadas consecutivas
        results = self.model.track(
            frame.image,
            persist=True,
            tracker=self.models_config.tracker_config,
            conf=self.teacher_config.detection_conf,
            imgsz=self.models_config.image_size,
            verbose=False,
        )
        detections = detections_from_result(results[0])
        return filter_by_class(detections, self.teacher_config.class_name)


class BehaviorDetector:
    """Detecta conductas en el frame completo, porque el modelo se entrenó con aulas completas."""

    def __init__(self, models_config: ModelsConfig, behavior_config: BehaviorConfig) -> None:
        hint = "Entrena primero con scripts/train.py --dataset behavior"
        self.model = load_model(models_config.behavior_weights, hint)
        self.models_config = models_config
        self.behavior_config = behavior_config

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            frame.image,
            conf=self.behavior_config.detection_conf,
            imgsz=self.models_config.image_size,
            verbose=False,
        )
        return detections_from_result(results[0])

    def assign(
        self, detections: list[Detection], teacher_box: BoundingBox | None
    ) -> BehaviorResult:
        return assign_behavior(detections, teacher_box, self.behavior_config)


def assign_behavior(
    detections: list[Detection],
    teacher_box: BoundingBox | None,
    config: BehaviorConfig,
) -> BehaviorResult:
    """Elige la conducta del docente y cuenta las señales de participación de alumnos."""
    participation_events = count_participation(detections, config.participation_classes)
    if teacher_box is None:
        return BehaviorResult(Behavior.NOT_VISIBLE, None, participation_events)

    best_detection = best_teacher_behavior(detections, teacher_box, config)
    if best_detection is None:
        return BehaviorResult(Behavior.OTHER, None, participation_events)

    behavior = behavior_for_class(best_detection.class_name, config)
    return BehaviorResult(behavior, best_detection.box, participation_events)


def best_teacher_behavior(
    detections: list[Detection],
    teacher_box: BoundingBox,
    config: BehaviorConfig,
) -> Detection | None:
    """Entre las conductas de docente que se solapan con su caja, gana la más confiable."""
    teacher_classes = {config.guide_class, config.board_writing_class}
    best_detection = None

    for detection in detections:
        if detection.class_name not in teacher_classes:
            continue
        if iou(detection.box, teacher_box) < config.behavior_iou_threshold:
            continue
        if best_detection is None or detection.confidence > best_detection.confidence:
            best_detection = detection
    return best_detection


def behavior_for_class(class_name: str, config: BehaviorConfig) -> Behavior:
    if class_name == config.guide_class:
        return Behavior.GUIDE
    if class_name == config.board_writing_class:
        return Behavior.BOARD_WRITING
    return Behavior.OTHER


def count_participation(detections: list[Detection], participation_classes: list[str]) -> int:
    total = 0
    for detection in detections:
        if detection.class_name in participation_classes:
            total += 1
    return total


class PoseEstimator:
    """Estima keypoints de todas las personas; luego se elige la del docente por IoU."""

    def __init__(self, models_config: ModelsConfig, pose_config: PoseConfig) -> None:
        self.model = load_model(models_config.pose_weights, "Revisa models.pose_weights.")
        self.models_config = models_config
        self.pose_config = pose_config

    def estimate(self, frame: Frame) -> list[PersonPose]:
        results = self.model.predict(
            frame.image,
            conf=self.pose_config.detection_conf,
            imgsz=self.models_config.image_size,
            verbose=False,
        )
        return poses_from_result(results[0])


def poses_from_result(result: Results) -> list[PersonPose]:
    if result.keypoints is None:
        return []

    corners_list = result.boxes.xyxy.cpu().tolist()
    keypoint_sets = result.keypoints.data.cpu().tolist()

    poses = []
    for corners, coco_points in zip(corners_list, keypoint_sets, strict=True):
        keypoints = pose_keypoints_from_coco(coco_points)
        poses.append(PersonPose(BoundingBox(*corners), keypoints))
    return poses


def pose_keypoints_from_coco(coco_points: list[list[float]]) -> PoseKeypoints:
    return PoseKeypoints(
        nose=to_keypoint(coco_points[NOSE_INDEX]),
        left_eye=to_keypoint(coco_points[LEFT_EYE_INDEX]),
        right_eye=to_keypoint(coco_points[RIGHT_EYE_INDEX]),
        left_shoulder=to_keypoint(coco_points[LEFT_SHOULDER_INDEX]),
        right_shoulder=to_keypoint(coco_points[RIGHT_SHOULDER_INDEX]),
    )


def to_keypoint(values: list[float]) -> Keypoint:
    x, y, confidence = values
    return Keypoint(x, y, confidence)
