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


class LanguageSignal(StrEnum):
    """Señales del lenguaje para R1 (invitar a participar) y R4 (faltas de respeto)."""

    PARTICIPATION_INVITATION = "participation_invitation"
    RESPECT_ISSUE = "respect_issue"


class Confidence(StrEnum):
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


class EvidenceKind(StrEnum):
    QUESTION = "question"
    FEEDBACK = "feedback"
    REGULATION = "regulation"
    PARTICIPATION = "participation"
    RESPECT = "respect"


QUESTION_LABELS = [category.value for category in QuestionCategory]
FEEDBACK_LABELS = [feedback.value for feedback in FeedbackType]
REGULATION_LABELS = [regulation.value for regulation in RegulationType]
PARTICIPATION_LABELS = [LanguageSignal.PARTICIPATION_INVITATION.value]
RESPECT_LABELS = [LanguageSignal.RESPECT_ISSUE.value]
CONFIDENCE_LABELS = [confidence.value for confidence in Confidence]
LABELS_BY_KIND = {
    EvidenceKind.QUESTION: QUESTION_LABELS,
    EvidenceKind.FEEDBACK: FEEDBACK_LABELS,
    EvidenceKind.REGULATION: REGULATION_LABELS,
    EvidenceKind.PARTICIPATION: PARTICIPATION_LABELS,
    EvidenceKind.RESPECT: RESPECT_LABELS,
}


@dataclass(frozen=True)
class TranscriptLine:
    """Una línea "[mm:ss] DOCENTE: texto", lo único que ve el LLM de la clase."""

    start_s: float
    end_s: float
    speaker_label: str
    text: str


@dataclass(frozen=True)
class CandidateQuestion:
    """Pregunta que marcó el audio y que el LLM debe clasificar por su contexto."""

    question_id: int
    start_s: float
    text: str
    rule_is_tag: bool
    student_answered: bool = False


@dataclass(frozen=True)
class Evidence:
    """Un elemento marcado por el LLM.

    question_id solo existe para preguntas; addressee, solo para invitaciones a participar
    (el nombre o la palabra con que el docente se dirige a un estudiante, copiado de la cita).
    """

    kind: EvidenceKind
    label: str
    timestamp_s: float
    quote: str
    reason: str
    confidence: str
    question_id: int | None = None
    addressee: str | None = None


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


class Weakness(StrEnum):
    """Debilidades que puede trabajar el consejo, en orden de prioridad."""

    SHORT_WAIT = "short_wait"
    ELEMENTAL_FEEDBACK = "elemental_feedback"
    LOW_MONITORING = "low_monitoring"
    CLOSED_QUESTIONS = "closed_questions"


@dataclass(frozen=True)
class SuggestionFocus:
    """La debilidad elegida por las reglas y lo que se vio, con el minuto y la cita reales.

    timestamp_s y quote son None cuando la debilidad no tiene un momento puntual (monitoreo).
    """

    weakness: Weakness
    what_we_saw: str
    timestamp_s: float | None = None
    quote: str | None = None


@dataclass(frozen=True)
class Suggestion:
    """El consejo en tres partes. "Lo que vimos" sale de los datos; las otras dos, del LLM."""

    focus: SuggestionFocus
    why_it_matters: str
    try_this: str
    example_phrase: str

    def as_text(self) -> str:
        return (
            f"Lo que vimos: {self.focus.what_we_saw}\n"
            f"Por qué importa: {self.why_it_matters}\n"
            f'Prueba esto: {self.try_this} Por ejemplo: "{self.example_phrase}"'
        )


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
    participation: list[Evidence] = field(default_factory=list)
    respect: list[Evidence] = field(default_factory=list)
    rejected: list[RejectedEvidence] = field(default_factory=list)
    suggestion: Suggestion | None = None
    usage: LlmUsage = field(default_factory=LlmUsage)
