"""Ficha por rúbrica: métricas del video y del audio, y citas del LLM con su minuto."""

from espejo.analysis.schemas import Citation, MetricEvidence, RubricEvidence, RubricMetric
from espejo.audio.schemas import QuestionCategory
from espejo.common.time_format import format_mmss
from espejo.llm.schemas import Evidence, FeedbackType, LlmAnalysis, RegulationType

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
    FeedbackType.INCORRECT: ["R3"],
    FeedbackType.ELEMENTAL: ["R3"],
    FeedbackType.DESCRIPTIVE: ["R3"],
    FeedbackType.DISCOVERY: ["R3"],
    RegulationType.FORMATIVE: ["R5"],
    RegulationType.EXTERNAL_CONTROL: ["R5"],
    RegulationType.MISTREATMENT: ["R4", "R5"],
}


def llm_evidence(analysis: LlmAnalysis) -> list[Evidence]:
    items = list(analysis.question_labels.values())
    items += analysis.feedback
    items += analysis.regulation
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
            selected.append(MetricEvidence(metric.name, metric.value))
    return selected


def build_rubric_evidence(
    metrics: list[RubricMetric], analysis: LlmAnalysis
) -> list[RubricEvidence]:
    """Sin métricas ni citas para una rúbrica, la ficha dice "evidencia insuficiente"."""
    rubric_evidence = []
    for rubric, title in RUBRIC_TITLES.items():
        rubric_metrics = metrics_for(rubric, metrics)
        citations = citations_for(rubric, analysis)
        status = SUFFICIENT
        if not rubric_metrics and not citations:
            status = INSUFFICIENT
        rubric_evidence.append(RubricEvidence(rubric, title, status, rubric_metrics, citations))
    return rubric_evidence


def evidence_as_text(rubric_evidence: list[RubricEvidence]) -> str:
    """La ficha en texto plano, para el prompt del consejo final."""
    lines = []
    for item in rubric_evidence:
        lines.append(f"{item.rubric} ({item.title}): {item.status}")
        for metric in item.metrics:
            lines.append(f"  - métrica {metric.name} = {metric.value}")
        for citation in item.citations:
            lines.append(
                f'  - [{format_mmss(citation.timestamp_s)}] {citation.label}: "{citation.quote}"'
            )
    return "\n".join(lines)


def evidence_times(rubric_evidence: list[RubricEvidence]) -> list[float]:
    times = []
    for item in rubric_evidence:
        for citation in item.citations:
            times.append(citation.timestamp_s)
    return times
