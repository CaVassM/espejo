"""Datos que viajan entre las etapas del pipeline de visión."""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from espejo.vision.geometry import BoundingBox


class Behavior(StrEnum):
    """Conducta del docente en un frame."""

    GUIDE = "guide"
    BOARD_WRITING = "board_writing"
    OTHER = "other"
    NOT_VISIBLE = "not_visible"


class Orientation(StrEnum):
    """Hacia dónde mira el docente respecto de la clase."""

    FACING = "facing"
    BACK = "back"
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"


@dataclass(frozen=True, eq=False)
class Frame:
    """Frame muestreado del video. La imagen está en BGR, como la entrega OpenCV."""

    index: int
    timestamp_s: float
    image: np.ndarray

    def width(self) -> int:
        return int(self.image.shape[1])

    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True)
class Detection:
    """Una caja detectada por un modelo YOLO."""

    box: BoundingBox
    class_name: str
    confidence: float
    track_id: int | None = None


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseKeypoints:
    """Solo los keypoints COCO que usamos para estimar la orientación."""

    nose: Keypoint
    left_eye: Keypoint
    right_eye: Keypoint
    left_shoulder: Keypoint
    right_shoulder: Keypoint


@dataclass(frozen=True)
class PersonPose:
    box: BoundingBox
    keypoints: PoseKeypoints


@dataclass(frozen=True)
class FrameDetections:
    """Salida cruda de los tres modelos en un frame; no guarda la imagen."""

    frame_index: int
    timestamp_s: float
    teacher_detections: list[Detection]
    behavior_detections: list[Detection]
    poses: list[PersonPose]


@dataclass(frozen=True)
class BehaviorResult:
    """Conducta asignada al docente y señales de participación del resto del aula."""

    behavior: Behavior
    behavior_box: BoundingBox | None
    participation_events: int


@dataclass(frozen=True)
class TeacherObservation:
    """Todo lo que sabemos del docente en un frame muestreado."""

    frame_index: int
    timestamp_s: float
    teacher_box: BoundingBox | None
    behavior: Behavior
    orientation: Orientation
    participation_events: int

    def is_teacher_visible(self) -> bool:
        return self.teacher_box is not None


@dataclass(frozen=True)
class Segment:
    """Tramo continuo de la clase con una misma etiqueta."""

    label: str
    start_s: float
    end_s: float

    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class VisionMetrics:
    analyzed_duration_s: float
    teacher_visible_ratio: float
    monitoring_ratio: float
    meets_monitoring_threshold: bool
    board_writing_ratio: float
    back_to_class_ratio: float
    path_length_normalized: float
    room_coverage_ratio: float
    student_participation_events: int


@dataclass(frozen=True)
class VideoInfo:
    """Metadatos del video de entrada."""

    path: str
    duration_s: float
    native_fps: float
    width: int
    height: int


@dataclass(frozen=True)
class VisionReport:
    video: VideoInfo
    sample_fps: float
    models: dict[str, str]
    behavior_timeline: list[Segment] = field(default_factory=list)
    orientation_timeline: list[Segment] = field(default_factory=list)
    metrics: VisionMetrics | None = None
