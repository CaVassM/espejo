"""Datos del reporte final de la clase (class_report.json)."""

from dataclasses import dataclass

from espejo.audio.schemas import Question
from espejo.vision.schemas import Segment

MetricValue = float | int | bool | None


@dataclass(frozen=True)
class RubricMetric:
    """Una métrica con la rúbrica del Minedu a la que aporta evidencia."""

    name: str
    value: MetricValue
    rubrics: list[str]
    description: str
    note: str | None = None


@dataclass(frozen=True)
class DataQuality:
    """Qué tan confiables son los datos; None si la fuente no estuvo disponible."""

    teacher_visible_ratio: float | None
    interpolated_ratio: float | None
    discarded_path_jumps: int | None
    low_confidence_audio_ratio: float | None
    discarded_hallucinated_segments: int | None
    speech_ratio: float | None
    diarization_used: bool
    diarization_note: str
    notes: list[str]


@dataclass(frozen=True)
class ReportMetadata:
    video: str
    duration_s: float
    sample_fps: float | None
    models: dict[str, str]
    video_available: bool
    audio_available: bool
    diarization_used: bool


@dataclass(frozen=True)
class ClassReport:
    metadata: ReportMetadata
    video_timeline: list[Segment]
    stage_interaction_events: list[Segment]
    questions: list[Question]
    metrics: list[RubricMetric]
    data_quality: DataQuality
