"""Ficha por rúbrica: métricas del video y del audio, y citas del LLM con su minuto."""

from espejo.analysis.schemas import Citation, MetricEvidence, RubricEvidence, RubricMetric
from espejo.audio.schemas import QuestionCategory
from espejo.llm.schemas import (
    Evidence,
    FeedbackType,
    LanguageSignal,
    LlmAnalysis,
    RegulationType,
)

RUBRIC_TITLES = {
    "R1": "Involucra activamente a los estudiantes en el proceso de aprendizaje",
    "R2": "Promueve el razonamiento, la creatividad y/o el pensamiento crítico",
    "R3": "Evalúa el progreso de los aprendizajes",
    "R4": "Propicia un ambiente de respeto y proximidad",
    "R5": "Regula positivamente el comportamiento de los estudiantes",
}
SUFFICIENT = "con evidencia"
INSUFFICIENT = "evidencia insuficiente"
# Qué clases del LLM aportan evidencia a cada rúbrica. Las retóricas no aportan.
RUBRICS_BY_LABEL = {
    QuestionCategory.OPEN: ["R2"],
    QuestionCategory.CLOSED: ["R2"],
    FeedbackType.ELEMENTAL: ["R3"],
    FeedbackType.DESCRIPTIVE: ["R3"],
    FeedbackType.DISCOVERY: ["R3"],
    RegulationType.FORMATIVE: ["R5"],
    RegulationType.EXTERNAL_CONTROL: ["R5"],
    RegulationType.MISTREATMENT: ["R4", "R5"],
    LanguageSignal.PARTICIPATION_INVITATION: ["R1"],
    LanguageSignal.RESPECT_ISSUE: ["R4"],
}


def llm_evidence(analysis: LlmAnalysis) -> list[Evidence]:
    items = list(analysis.question_labels.values())
    items += analysis.feedback
    items += analysis.regulation
    items += analysis.participation
    items += analysis.respect
    return items


def citations_for(rubric: str, analysis: LlmAnalysis) -> list[Citation]:
    citations = []
    for item in llm_evidence(analysis):
        if rubric in RUBRICS_BY_LABEL.get(item.label, []):
            citations.append(Citation(item.timestamp_s, item.label, item.quote, item.reason))
    citations.sort(key=citation_time)
    return citations


def citation_time(citation: Citation) -> float:
    return citation.timestamp_s


def metrics_for(rubric: str, metrics: list[RubricMetric]) -> list[MetricEvidence]:
    """Solo métricas con valor: una métrica nula no es evidencia."""
    selected = []
    for metric in metrics:
        if rubric in metric.rubrics and metric.value is not None:
            selected.append(MetricEvidence(metric.name, metric.value, metric.experimental))
    return selected


def has_main_evidence(metrics: list[MetricEvidence]) -> bool:
    for metric in metrics:
        if not metric.experimental:
            return True
    return False


def build_rubric_evidence(
    metrics: list[RubricMetric], analysis: LlmAnalysis
) -> list[RubricEvidence]:
    """Sin métricas principales ni citas, la ficha dice "evidencia insuficiente".

    Las métricas experimentales se muestran, pero solas no bastan.
    """
    rubric_evidence = []
    for rubric, title in RUBRIC_TITLES.items():
        rubric_metrics = metrics_for(rubric, metrics)
        citations = citations_for(rubric, analysis)
        status = SUFFICIENT
        if not has_main_evidence(rubric_metrics) and not citations:
            status = INSUFFICIENT
        rubric_evidence.append(RubricEvidence(rubric, title, status, rubric_metrics, citations))
    return rubric_evidence
