"""Datos del módulo LLM: evidencias con cita, uso de tokens y análisis de la clase."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from espejo.audio.schemas import QuestionCategory


class FeedbackType(StrEnum):
    """Tipos de retroalimentación del manual (R3)."""

    INCORRECT = "feedback_incorrect"
    ELEMENTAL = "feedback_elemental"
    DESCRIPTIVE = "feedback_descriptive"
    DISCOVERY = "feedback_discovery"


class RegulationType(StrEnum):
    """Mecanismos de regulación del comportamiento del manual (R5)."""

    FORMATIVE = "regulation_formative"
    EXTERNAL_CONTROL = "regulation_external_control"
    MISTREATMENT = "regulation_mistreatment"


class Confidence(StrEnum):
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


class EvidenceKind(StrEnum):
    QUESTION = "question"
    FEEDBACK = "feedback"
    REGULATION = "regulation"


QUESTION_LABELS = [category.value for category in QuestionCategory]
FEEDBACK_LABELS = [feedback.value for feedback in FeedbackType]
REGULATION_LABELS = [regulation.value for regulation in RegulationType]
CONFIDENCE_LABELS = [confidence.value for confidence in Confidence]
LABELS_BY_KIND = {
    EvidenceKind.QUESTION: QUESTION_LABELS,
    EvidenceKind.FEEDBACK: FEEDBACK_LABELS,
    EvidenceKind.REGULATION: REGULATION_LABELS,
}


@dataclass(frozen=True)
class TranscriptLine:
    """Una línea "[mm:ss] DOCENTE: texto", lo único que ve el LLM de la clase."""

    start_s: float
    speaker_label: str
    text: str


@dataclass(frozen=True)
class CandidateQuestion:
    """Pregunta que marcó el audio y que el LLM debe clasificar por su contexto."""

    question_id: int
    start_s: float
    text: str
    rule_is_tag: bool


@dataclass(frozen=True)
class Evidence:
    """Un elemento marcado por el LLM. question_id solo existe para preguntas."""

    kind: EvidenceKind
    label: str
    timestamp_s: float
    quote: str
    reason: str
    confidence: str
    question_id: int | None = None


@dataclass(frozen=True)
class RejectedEvidence:
    evidence: Evidence
    reason: str


@dataclass(frozen=True)
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    calls: int = 0

    def plus(self, other: "LlmUsage") -> "LlmUsage":
        return LlmUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            seconds=self.seconds + other.seconds,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )


@dataclass(frozen=True)
class RawReply:
    """Texto que devolvió el proveedor, antes de validar el JSON."""

    text: str
    usage: LlmUsage


@dataclass(frozen=True)
class JsonReply:
    content: dict[str, Any]
    usage: LlmUsage


@dataclass(frozen=True)
class Suggestion:
    text: str
    timestamp_s: float


@dataclass(frozen=True)
class LlmAnalysis:
    """Resultado del LLM ya validado. used = False si no hubo LLM (se usan las reglas)."""

    used: bool
    note: str
    provider: str | None = None
    model: str | None = None
    question_labels: dict[int, Evidence] = field(default_factory=dict)
    feedback: list[Evidence] = field(default_factory=list)
    regulation: list[Evidence] = field(default_factory=list)
    rejected: list[RejectedEvidence] = field(default_factory=list)
    suggestion: Suggestion | None = None
    usage: LlmUsage = field(default_factory=LlmUsage)
