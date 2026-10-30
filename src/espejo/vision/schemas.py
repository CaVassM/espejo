"""Datos que viajan entre las etapas del módulo de visión."""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from espejo.vision.geometry import BoundingBox


class Behavior(StrEnum):
    """Estado del docente en un frame. Prioridad: board_writing > guide > other.

    "other" no es una clase aprendida: es el relleno cuando no se detecta una acción del docente.
    "not_visible" no es una conducta: indica que en ese frame no se encontró al docente.
    """

    BOARD_WRITING = "board_writing"
    GUIDE = "guide"
    OTHER = "other"
    NOT_VISIBLE = "not_visible"


class Orientation(StrEnum):
    """Solo la usa el estimador de pose, que hoy está desactivado (pose.enabled = false)."""

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
    """Salida cruda de los modelos en un frame; no guarda la imagen."""

    frame_index: int
    timestamp_s: float
    teacher_detections: list[Detection]
    behavior_detections: list[Detection]
    poses: list[PersonPose] = field(default_factory=list)


@dataclass(frozen=True)
class VideoInfo:
    """Metadatos del video de entrada."""

    path: str
    duration_s: float
    native_fps: float
    width: int
    height: int


@dataclass(frozen=True)
class DetectionRun:
    """Todo lo que VER produce sobre el video; INTERPRETAR trabaja solo con esto."""

    video: VideoInfo
    sample_fps: float
    models: dict[str, str]
    frames: list[FrameDetections]


@dataclass(frozen=True)
class TeacherObservation:
    """Lo que sabemos del docente y del aula en un frame muestreado."""

    frame_index: int
    timestamp_s: float
    teacher_box: BoundingBox | None
    state: Behavior
    stage_interaction_present: bool
    interpolated: bool = False

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
class PathSummary:
    """Recorrido del docente después de filtrar el temblor de la caja y los saltos imposibles."""

    length_normalized: float
    discarded_jumps: int


@dataclass(frozen=True)
class VisionMetrics:
    analyzed_duration_s: float
    teacher_visible_ratio: float
    interpolated_ratio: float
    board_writing_ratio: float
    guide_ratio: float
    stage_interaction_count: int
    path: PathSummary


@dataclass(frozen=True)
class VisionInterpretation:
    observations: list[TeacherObservation]
    timeline: list[Segment]
    stage_interaction_events: list[Segment]
    metrics: VisionMetrics
