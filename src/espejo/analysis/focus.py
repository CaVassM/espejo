"""Elige la debilidad que trabajará el consejo, con el minuto y la cita que la muestran.

Prioridad: espera < 3 s, retroalimentación solo elemental, monitoreo < 25 % y preguntas
mayormente cerradas. Lo decide el código y no el LLM: así el consejo siempre parte de un dato
real, y "Lo que vimos" no puede citar un momento inventado.
"""

from dataclasses import dataclass

from espejo.analysis.schemas import RubricMetric
from espejo.audio.schemas import Question, QuestionCategory
from espejo.common.time_format import format_mmss
from espejo.llm.schemas import Evidence, FeedbackType, SuggestionFocus, Weakness

PERCENT = 100


@dataclass(frozen=True)
class FocusThresholds:
    wait_ok_s: float
    monitoring: float
    min_open_question_ratio: float


def numeric_metric(metrics: list[RubricMetric], name: str) -> float | None:
    """Valor numérico de una métrica; None si no existe o no tiene valor."""
    for metric in metrics:
        is_number = isinstance(metric.value, (int, float)) and not isinstance(metric.value, bool)
        if metric.name == name and is_number:
            return float(metric.value)
    return None


def format_decimal(value: float) -> str:
    """1.27 -> "1,3" (coma decimal, como se escribe en el Perú)."""
    return f"{value:.1f}".replace(".", ",")


def format_percent(ratio: float) -> str:
    return f"{round(ratio * PERCENT)} %"


def counted_questions(questions: list[Question]) -> list[Question]:
    counted = []
    for question in questions:
        if not question.is_rhetorical():
            counted.append(question)
    return counted


def example_short_wait(questions: list[Question]) -> Question | None:
    """La primera pregunta abierta con espera corta; si no hay, la primera con espera corta."""
    short = []
    for question in counted_questions(questions):
        if question.wait_time_s is not None and question.wait_time_ok is False:
            short.append(question)
    for question in short:
        if question.llm_category == QuestionCategory.OPEN:
            return question
    if short:
        return short[0]
    return None


def short_wait_focus(
    metrics: list[RubricMetric], questions: list[Question], thresholds: FocusThresholds
) -> SuggestionFocus | None:
    mean_wait_s = numeric_metric(metrics, "mean_wait_time_s")
    if mean_wait_s is None or mean_wait_s >= thresholds.wait_ok_s:
        return None
    question = example_short_wait(questions)
    if question is None or question.wait_time_s is None:
        return None
    what_we_saw = (
        f'En el minuto {format_mmss(question.start_s)} preguntaste "{question.text}" y la pausa '
        f"antes de que alguien hablara fue de {format_decimal(question.wait_time_s)} s. "
        f"En promedio esperaste {format_decimal(mean_wait_s)} s después de cada pregunta."
    )
    return SuggestionFocus(Weakness.SHORT_WAIT, what_we_saw, question.start_s, question.text)


def evidence_time(item: Evidence) -> float:
    return item.timestamp_s


def elemental_feedback_focus(feedback: list[Evidence]) -> SuggestionFocus | None:
    if not feedback:
        return None
    for item in feedback:
        if item.label != FeedbackType.ELEMENTAL:
            return None
    first = min(feedback, key=evidence_time)
    summary = "La única retroalimentación que encontramos fue así"
    if len(feedback) > 1:
        summary = f"Las {len(feedback)} retroalimentaciones que encontramos fueron así"
    what_we_saw = (
        f"En el minuto {format_mmss(first.timestamp_s)}, después de una respuesta, dijiste "
        f'"{first.quote}". {summary}: confirman que está bien, pero no le piden al estudiante '
        "explicar su idea."
    )
    return SuggestionFocus(Weakness.ELEMENTAL_FEEDBACK, what_we_saw, first.timestamp_s, first.quote)


def low_monitoring_focus(
    metrics: list[RubricMetric], thresholds: FocusThresholds
) -> SuggestionFocus | None:
    """El monitoreo no tiene un momento puntual: "Lo que vimos" va sin minuto ni cita."""
    ratio = numeric_metric(metrics, "monitoring_ratio")
    if ratio is None or ratio >= thresholds.monitoring:
        return None
    what_we_saw = (
        f"Recogiste evidencia de lo que entienden tus estudiantes (preguntando o recorriendo "
        f"sus lugares) durante el {format_percent(ratio)} de la clase; el manual pide al menos "
        f"el {format_percent(thresholds.monitoring)}."
    )
    return SuggestionFocus(Weakness.LOW_MONITORING, what_we_saw)


def first_closed_question(questions: list[Question]) -> Question | None:
    for question in questions:
        if question.llm_category == QuestionCategory.CLOSED:
            return question
    return None


def closed_questions_focus(
    metrics: list[RubricMetric], questions: list[Question], thresholds: FocusThresholds
) -> SuggestionFocus | None:
    ratio = numeric_metric(metrics, "open_question_ratio")
    if ratio is None or ratio >= thresholds.min_open_question_ratio:
        return None
    share = (
        f"Solo el {format_percent(ratio)} de tus preguntas pedía explicar o predecir; las demás "
        "pedían un dato o un sí/no."
    )
    question = first_closed_question(questions)
    if question is None:
        return SuggestionFocus(Weakness.CLOSED_QUESTIONS, share)
    what_we_saw = f'En el minuto {format_mmss(question.start_s)} preguntaste "{question.text}". '
    return SuggestionFocus(
        Weakness.CLOSED_QUESTIONS, what_we_saw + share, question.start_s, question.text
    )


def choose_focus(
    metrics: list[RubricMetric],
    questions: list[Question],
    feedback: list[Evidence],
    thresholds: FocusThresholds,
) -> SuggestionFocus | None:
    """La primera debilidad que aparece, en orden de prioridad. None si no hay ninguna."""
    candidates = [
        short_wait_focus(metrics, questions, thresholds),
        elemental_feedback_focus(feedback),
        low_monitoring_focus(metrics, thresholds),
        closed_questions_focus(metrics, questions, thresholds),
    ]
    for focus in candidates:
        if focus is not None:
            return focus
    return None
