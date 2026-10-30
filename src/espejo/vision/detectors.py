"""Envoltorios de los modelos YOLO: docente, conductas y pose."""

import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

from espejo.config import BehaviorConfig, ModelsConfig, PoseConfig, TeacherConfig
from espejo.vision.geometry import BoundingBox
from espejo.vision.schemas import (
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
        self.tracking_model = load_model(models_config.teacher_weights, hint)
        # Instancia aparte: llamar predict() sobre el modelo que trackea reinicia el tracker
        self.detection_model = load_model(models_config.teacher_weights, hint)
        self.models_config = models_config
        self.teacher_config = teacher_config

    def track(self, frame: Frame) -> list[Detection]:
        """Detecciones con track_id; si el tracker no entrega nada, detecciones sin track_id.

        ByteTrack no devuelve un objeto en el primer frame en que aparece como track nuevo
        (aún no está confirmado). Tras un corte u oclusión eso deja al docente "invisible"
        en un frame donde sí se ve; la detección simple lo recupera y luego TeacherTracker
        decide si la acepta con teacher_fallback_conf.
        """
        # persist=True mantiene el estado del tracker entre llamadas consecutivas
        results = self.tracking_model.track(
            frame.image,
            persist=True,
            tracker=self.models_config.tracker_config,
            conf=self.teacher_config.detection_conf,
            imgsz=self.models_config.image_size,
            verbose=False,
        )
        tracked = filter_by_class(
            detections_from_result(results[0]), self.teacher_config.class_name
        )
        if tracked:
            return tracked
        return self.detect_without_tracking(frame)

    def detect_without_tracking(self, frame: Frame) -> list[Detection]:
        results = self.detection_model.predict(
            frame.image,
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
