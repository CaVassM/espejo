"""Lo que puede consultar el asistente de IA del docente (servidor MCP). Solo lectura.

Respuestas cortas en texto legible, con el detalle solo bajo pedido (evidencias por página, un
fragmento de ±60 s). Todo sale de datos ya anonimizados: el reporte y lo que dijo el docente
("[ALUMNO responde]" y "[NOMBRE]" ya vienen tapados). Nunca video, audio, la transcripción
completa ni datos crudos. Un análisis de otra clave se comporta igual que uno que no existe.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.repositories.analyses import AnalysisRecord, AnalysisRepository
from app.schemas.analyses import AnalysisStatus
from espejo.audio.capture_io import diarization_from_dict, segment_from_dict
from espejo.audio.diarization import assign_speakers
from espejo.audio.speaker_roles import assign_roles
from espejo.common.config_io import read_yaml
from espejo.common.time_format import format_mmss

RUBRICS = ["R1", "R2", "R3", "R4", "R5"]
SECONDS_PER_MINUTE = 60
PERCENT = 100
CONFIDENCE_ORDER = {"alta": 0, "media": 1, "baja": 2}
UNKNOWN_CONFIDENCE_RANK = 3
EVIDENCE_LISTS = ["question_labels", "feedback", "regulation", "participation", "respect"]
LABEL_NAMES = {
    "question_open": "pregunta abierta",
    "question_closed": "pregunta cerrada",
    "feedback_discovery": "retroalimentación por descubrimiento",
    "feedback_descriptive": "retroalimentación descriptiva",
    "feedback_elemental": "retroalimentación elemental",
    "regulation_formative": "regulación formativa",
    "regulation_external_control": "regulación por control externo",
    "regulation_mistreatment": "maltrato",
    "participation_invitation": "invitación a participar",
    "respect_issue": "falta de respeto",
}
ROLE_NAMES = {"teacher": "DOCENTE", "student": "ALUMNO", "unknown": "HABLANTE"}
MINUTE_PATTERN = re.compile(r"^\s*(\d+)(?::(\d{1,2}))?\s*$")
NOT_FOUND = "No existe ese análisis entre tus clases."
NOT_READY = "El reporte de ese análisis aún no está listo."
CUT_NOTICE = "\n… (respuesta recortada: pasaba de {limit} caracteres; pide menos detalle)"


@dataclass(frozen=True)
class AssistantSettings:
    max_response_chars: int
    max_classes: int
    evidence_page_size: int
    fragment_window_s: float


def load_assistant_settings(config_path: Any) -> AssistantSettings:
    return AssistantSettings(**read_yaml(config_path)["assistant"])


class AssistantError(ValueError):
    """Mensaje claro para el asistente, sin revelar si el análisis existe para otra clave."""


def fit(text: str, limit: int) -> str:
    """Tope de caracteres por respuesta: corta en un salto de línea y lo avisa en el texto."""
    if len(text) <= limit:
        return text
    notice = CUT_NOTICE.format(limit=limit)
    cut = text[: limit - len(notice)]
    if "\n" in cut:
        cut = cut[: cut.rfind("\n")]
    return cut + notice


def decimal(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def percent(value: float) -> str:
    return f"{round(value * PERCENT)} %"


def metric(report: dict[str, Any], name: str) -> Any:
    item = report["metrics"].get(name)
    return None if item is None else item["value"]


def duration_text(record: AnalysisRecord) -> str:
    if record.report is None:
        return "sin reporte aún"
    return f"{round(record.report['metadata']['duration_s'] / SECONDS_PER_MINUTE)} min"


def class_title(record: AnalysisRecord) -> str:
    return f"{record.created_at:%Y-%m-%d} · {record.topic or 'sin tema'}"


def key_metric_lines(report: dict[str, Any]) -> list[str]:
    """Las 5 métricas clave, en palabras simples."""
    lines = []
    questions = metric(report, "teacher_question_count")
    per_minute = metric(report, "questions_per_minute")
    open_ratio = metric(report, "open_question_ratio")
    if questions is not None:
        line = f"- Preguntas: {questions}"
        if per_minute is not None:
            line += f" ({decimal(per_minute)} por minuto)"
        if open_ratio is not None:
            line += f", {percent(open_ratio)} abiertas"
        lines.append(line)
    wait = metric(report, "mean_wait_time_s")
    if wait is not None:
        lines.append(f"- Espera promedio después de preguntar: {decimal(wait)} s")
    response_rate = metric(report, "response_rate")
    if response_rate is not None:
        lines.append(f"- Preguntas con respuesta de un alumno: {percent(response_rate)}")
    monitoring = metric(report, "monitoring_ratio")
    if monitoring is not None:
        lines.append(f"- Monitoreo (preguntas con respuesta): {percent(monitoring)}")
    talk = metric(report, "teacher_talk_ratio")
    if talk is not None:
        lines.append(f"- El docente habla el {percent(talk)} del tiempo")
    return lines


def suggestion_lines(report: dict[str, Any]) -> list[str]:
    suggestion = report.get("one_suggestion")
    if suggestion is None:
        return ["- (sin consejo en esta clase)"]
    return [
        f"- Lo que vimos: {suggestion['what_we_saw']}",
        f"- Por qué importa: {suggestion['why_it_matters']}",
        f'- Prueba esto: {suggestion["try_this"]} Por ejemplo: "{suggestion["example_phrase"]}"',
    ]


def confidence_lookup(report: dict[str, Any]) -> dict[tuple[str, str, str], str]:
    """La confianza del LLM de cada cita, para ordenar las evidencias por relevancia."""
    lookup = {}
    for list_name in EVIDENCE_LISTS:
        for item in report["llm_analysis"].get(list_name, []):
            key = (item.get("timestamp"), item.get("label"), item.get("quote"))
            if item.get("confidence"):
                lookup[key] = item["confidence"]
    return lookup


def parse_minute(text: str) -> float:
    """ "3" -> 180 s; "03:12" -> 192 s."""
    match = MINUTE_PATTERN.match(text)
    if match is None:
        raise AssistantError('El minuto debe ser como "3" o "03:12".')
    seconds = int(match.group(2) or 0)
    return float(int(match.group(1)) * SECONDS_PER_MINUTE + seconds)


class AssistantQueries:
    def __init__(self, repository: AnalysisRepository, settings: AssistantSettings) -> None:
        self.repository = repository
        self.settings = settings

    def answer(self, text: str) -> str:
        return fit(text, self.settings.max_response_chars)

    def owned(self, key_hash: str | None, analysis_id: str) -> AnalysisRecord:
        record = None
        if key_hash is not None:
            record = self.repository.get_owned(analysis_id, key_hash)
        if record is None:
            raise AssistantError(NOT_FOUND)
        return record

    def owned_report(self, key_hash: str | None, analysis_id: str) -> dict[str, Any]:
        record = self.owned(key_hash, analysis_id)
        if record.status != AnalysisStatus.DONE or record.report is None:
            raise AssistantError(NOT_READY)
        return record.report

    def list_classes(self, key_hash: str | None) -> str:
        records = []
        if key_hash is not None:
            records = self.repository.list_by_key(key_hash)[: self.settings.max_classes]
        if not records:
            return "No hay clases para esta clave."
        lines = [f"Tus clases (las {self.settings.max_classes} más recientes):"]
        for record in records:
            lines.append(
                f"- {record.id} | {class_title(record)} | {duration_text(record)} | {record.stage}"
            )
        return self.answer("\n".join(lines))

    def report(self, key_hash: str | None, analysis_id: str) -> str:
        record = self.owned(key_hash, analysis_id)
        report = self.owned_report(key_hash, analysis_id)
        lines = [f"Clase {analysis_id} ({class_title(record)}, {duration_text(record)})", ""]
        lines += ["Métricas clave:", *key_metric_lines(report), ""]
        lines += ["Consejo:", *suggestion_lines(report), ""]
        lines.append("Ficha por rúbrica:")
        for item in report["rubric_evidence"]:
            count = len(item["citations"])
            lines.append(f"- {item['rubric']} {item['title']}: {item['status']}, {count} citas")
        lines += ["", "Para ver citas: ver_evidencias(id, rubrica)."]
        return self.answer("\n".join(lines))

    def evidence(self, key_hash: str | None, analysis_id: str, rubric: str, page: int) -> str:
        rubric = rubric.strip().upper()
        if rubric not in RUBRICS:
            raise AssistantError("La rúbrica debe ser R1, R2, R3, R4 o R5.")
        if page < 1:
            raise AssistantError("La página empieza en 1.")
        report = self.owned_report(key_hash, analysis_id)
        item = None
        for entry in report["rubric_evidence"]:
            if entry["rubric"] == rubric:
                item = entry
        if item is None:
            raise AssistantError(NOT_FOUND)
        confidences = confidence_lookup(report)

        ranked = []
        for citation in item["citations"]:
            key = (citation["timestamp"], citation["label"], citation["quote"])
            confidence = confidences.get(key, "")
            rank = CONFIDENCE_ORDER.get(confidence, UNKNOWN_CONFIDENCE_RANK)
            ranked.append((rank, citation["timestamp"], confidence, citation))
        ranked.sort(key=lambda entry: (entry[0], entry[1]))

        size = self.settings.evidence_page_size
        start = (page - 1) * size
        shown = ranked[start : start + size]
        lines = [f"{rubric} {item['title']} ({item['status']}), página {page}:"]
        if not shown:
            lines.append("- No hay citas en esta página.")
        for _, minute, confidence, citation in shown:
            label = LABEL_NAMES.get(citation["label"], citation["label"])
            detail = f"{label}, confianza {confidence}" if confidence else label
            lines.append(f'- [{minute}] ({detail}) "{citation["quote"]}" — {citation["reason"]}')
        remaining = len(ranked) - (start + len(shown))
        if remaining > 0:
            lines.append(f"Hay {remaining} más. Pide pagina={page + 1}.")
        return self.answer("\n".join(lines))

    def compare(self, key_hash: str | None, id_a: str, id_b: str) -> str:
        """Solo lo que cambió, ya calculado: "espera: 1,2 s → 2,8 s"."""
        report_a = self.owned_report(key_hash, id_a)
        report_b = self.owned_report(key_hash, id_b)
        changes = []
        for name, label, show in [
            ("mean_wait_time_s", "Espera promedio", lambda value: f"{decimal(value)} s"),
            ("open_question_ratio", "Preguntas abiertas", percent),
            ("monitoring_ratio", "Monitoreo", percent),
            ("teacher_question_count", "Preguntas", str),
        ]:
            value_a, value_b = metric(report_a, name), metric(report_b, name)
            if value_a is None or value_b is None or show(value_a) == show(value_b):
                continue
            changes.append(f"- {label}: {show(value_a)} → {show(value_b)}")

        feedback_a = Counter(item["label"] for item in report_a["llm_analysis"]["feedback"])
        feedback_b = Counter(item["label"] for item in report_b["llm_analysis"]["feedback"])
        for label in sorted(set(feedback_a) | set(feedback_b)):
            if feedback_a[label] != feedback_b[label]:
                name = LABEL_NAMES.get(label, label).capitalize()
                changes.append(f"- {name}: {feedback_a[label]} → {feedback_b[label]}")

        header = f"Cambios de la clase {id_a} a la clase {id_b}:"
        if not changes:
            return self.answer(
                header + "\n- Sin cambios en espera, preguntas, monitoreo ni retroalimentación."
            )
        return self.answer("\n".join([header, *changes]))

    def fragment(self, key_hash: str | None, analysis_id: str, minute: str) -> str:
        """Lo que se dijo alrededor de un minuto (±fragment_window_s), ya anonimizado."""
        record = self.owned(key_hash, analysis_id)
        if record.audio_raw is None:
            raise AssistantError("Esa clase no tiene audio analizado.")
        center_s = parse_minute(minute)
        window_s = self.settings.fragment_window_s
        segments = []
        for raw in record.audio_raw["segments"]:
            segments.append(segment_from_dict(raw))
        diarization = diarization_from_dict(record.audio_raw["diarization"])
        lines = [
            f"Clase {analysis_id}, de {format_mmss(max(0.0, center_s - window_s))} a "
            f"{format_mmss(center_s + window_s)}:"
        ]
        for segment in assign_roles(assign_speakers(segments, diarization)):
            if segment.end_s < center_s - window_s or segment.start_s > center_s + window_s:
                continue
            speaker = ROLE_NAMES.get(segment.role.value, "HABLANTE")
            lines.append(f"[{format_mmss(segment.start_s)}] {speaker}: {segment.text}")
        if len(lines) == 1:
            lines.append("(no se dijo nada en ese tramo)")
        return self.answer("\n".join(lines))

    def plan_prompt(self, key_hash: str | None, analysis_id: str) -> str:
        summary = self.report(key_hash, analysis_id)
        return self.answer(
            "Eres un colega que acompaña a un docente del Perú. Con este resumen de su última "
            "clase, propón un plan breve para su próxima clase: 3 a 5 pasos concretos, cada uno "
            "con una frase de ejemplo que el docente pueda decir. No califiques al docente ni "
            "pongas notas; usa un tono cálido y palabras simples. Si necesitas ejemplos, usa "
            "ver_evidencias.\n\n" + summary
        )
