"""Lo que puede consultar el asistente de IA del docente (servidor MCP). Solo lectura.

Todo sale del reporte ya anonimizado (class_report.json): la ficha, métricas principales, citas
del docente con su minuto y el consejo. Nunca video, audio, la transcripción, los datos crudos
del worker ni texto de alumnos. Un análisis de otra clave se comporta igual que uno que no existe.
"""

from collections import Counter
from typing import Any

from app.repositories.analyses import AnalysisRecord, AnalysisRepository
from app.schemas.analyses import AnalysisStatus

RUBRICS = ["R1", "R2", "R3", "R4", "R5"]
SECONDS_PER_MINUTE = 60
DECIMALS = 3
MAIN_METRICS = [
    "teacher_question_count",
    "questions_per_minute",
    "open_question_ratio",
    "mean_wait_time_s",
    "wait_time_ok_ratio",
    "response_rate",
    "teacher_talk_ratio",
    "monitoring_ratio",
    "meets_monitoring_threshold",
]
COMPARED_METRICS = ["mean_wait_time_s", "open_question_ratio", "monitoring_ratio"]
SUGGESTION_FIELDS = ["what_we_saw", "why_it_matters", "try_this", "example_phrase"]
NOT_FOUND = "No existe ese análisis entre tus clases."
NOT_READY = "El reporte de ese análisis aún no está listo."


class AssistantError(ValueError):
    """Mensaje claro para el asistente, sin revelar si el análisis existe para otra clave."""


def round_number(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DECIMALS)
    return value


def duration_minutes(record: AnalysisRecord) -> float | None:
    if record.report is None:
        return None
    return round(record.report["metadata"]["duration_s"] / SECONDS_PER_MINUTE, 1)


def metric_values(report: dict[str, Any], names: list[str]) -> dict[str, Any]:
    values = {}
    for name in names:
        metric = report["metrics"].get(name)
        if metric is not None:
            values[name] = round_number(metric["value"])
    return values


def feedback_counts(report: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for item in report["llm_analysis"]["feedback"]:
        counts[item["label"].removeprefix("feedback_")] += 1
    return dict(counts)


def rubric_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Estado y métricas de una rúbrica; las experimentales vienen marcadas."""
    metrics = []
    for metric in item["metrics"]:
        metrics.append(
            {
                "nombre": metric["name"],
                "valor": round_number(metric["value"]),
                "experimental": metric.get("experimental", False),
            }
        )
    return {
        "rubrica": item["rubric"],
        "titulo": item["title"],
        "estado": item["status"],
        "metricas": metrics,
        "citas": len(item["citations"]),
    }


def suggestion_parts(report: dict[str, Any]) -> dict[str, Any] | None:
    suggestion = report.get("one_suggestion")
    if suggestion is None:
        return None
    parts = {"minuto": suggestion.get("timestamp")}
    for name in SUGGESTION_FIELDS:
        parts[name] = suggestion.get(name)
    return parts


def difference(value_a: Any, value_b: Any) -> float | None:
    if isinstance(value_a, int | float) and isinstance(value_b, int | float):
        return round(value_b - value_a, DECIMALS)
    return None


class AssistantQueries:
    def __init__(self, repository: AnalysisRepository) -> None:
        self.repository = repository

    def owned_report(self, key_hash: str | None, analysis_id: str) -> dict[str, Any]:
        record = None
        if key_hash is not None:
            record = self.repository.get_owned(analysis_id, key_hash)
        if record is None:
            raise AssistantError(NOT_FOUND)
        if record.status != AnalysisStatus.DONE or record.report is None:
            raise AssistantError(NOT_READY)
        return record.report

    def list_classes(self, key_hash: str | None) -> list[dict[str, Any]]:
        if key_hash is None:
            return []
        classes = []
        for record in self.repository.list_by_key(key_hash):
            classes.append(
                {
                    "id": record.id,
                    "fecha": record.created_at.isoformat(timespec="minutes"),
                    "tema": record.topic,
                    "duracion_min": duration_minutes(record),
                    "estado": record.stage,
                }
            )
        return classes

    def report(self, key_hash: str | None, analysis_id: str) -> dict[str, Any]:
        report = self.owned_report(key_hash, analysis_id)
        rubrics = []
        for item in report["rubric_evidence"]:
            rubrics.append(rubric_summary(item))
        return {
            "id": analysis_id,
            "ficha": rubrics,
            "metricas_principales": metric_values(report, MAIN_METRICS),
            "retroalimentaciones": feedback_counts(report),
            "consejo": suggestion_parts(report),
        }

    def evidence(self, key_hash: str | None, analysis_id: str, rubric: str) -> dict[str, Any]:
        rubric = rubric.strip().upper()
        if rubric not in RUBRICS:
            raise AssistantError("La rúbrica debe ser R1, R2, R3, R4 o R5.")
        report = self.owned_report(key_hash, analysis_id)
        for item in report["rubric_evidence"]:
            if item["rubric"] == rubric:
                citations = []
                for citation in item["citations"]:
                    citations.append(
                        {
                            "minuto": citation["timestamp"],
                            "tipo": citation["label"],
                            "cita": citation["quote"],
                            "por_que": citation["reason"],
                        }
                    )
                return {
                    "id": analysis_id,
                    "rubrica": rubric,
                    "titulo": item["title"],
                    "estado": item["status"],
                    "citas": citations,
                }
        raise AssistantError(NOT_FOUND)

    def compare(self, key_hash: str | None, id_a: str, id_b: str) -> dict[str, Any]:
        report_a = self.owned_report(key_hash, id_a)
        report_b = self.owned_report(key_hash, id_b)
        values_a = metric_values(report_a, COMPARED_METRICS)
        values_b = metric_values(report_b, COMPARED_METRICS)
        differences = {}
        for name in COMPARED_METRICS:
            differences[name] = difference(values_a.get(name), values_b.get(name))
        return {
            "clase_a": {"id": id_a, **values_a, "retroalimentaciones": feedback_counts(report_a)},
            "clase_b": {"id": id_b, **values_b, "retroalimentaciones": feedback_counts(report_b)},
            "diferencia_b_menos_a": differences,
        }

    def plan_prompt(self, key_hash: str | None, analysis_id: str) -> str:
        report = self.owned_report(key_hash, analysis_id)
        suggestion = suggestion_parts(report)
        citations = []
        for item in report["rubric_evidence"]:
            for citation in item["citations"][:3]:
                minute = citation["timestamp"]
                citations.append(f"- {item['rubric']} [{minute}] {citation['quote']}")
        return (
            "Eres un colega que acompaña a un docente del Perú. Con el consejo y las evidencias "
            "de su última clase, propón un plan breve para su próxima clase: 3 a 5 pasos "
            "concretos, cada uno con una frase de ejemplo que el docente pueda decir. "
            "No califiques al docente ni pongas notas; usa un tono cálido y palabras simples.\n\n"
            f"Consejo: {suggestion}\n\n"
            f"Métricas principales: {metric_values(report, MAIN_METRICS)}\n\n"
            "Algunas evidencias (minuto y cita del docente):\n" + "\n".join(citations)
        )
