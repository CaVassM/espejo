"""class_report.json y el resumen que se imprime en consola."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from espejo.analysis.schemas import ClassReport, MetricValue, RubricMetric
from espejo.audio.schemas import Question, StudentResponse
from espejo.common.json_io import write_json
from espejo.common.time_format import format_mmss, round_seconds
from espejo.vision.schemas import Segment

RATIO_DECIMALS = 3
SUMMARY_QUESTION_COUNT = 5


def round_value(value: MetricValue) -> MetricValue:
    """bool es subclase de int en Python: se revisa antes para no redondearlo."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    return round(value, RATIO_DECIMALS)


def segment_to_dict(segment: Segment) -> dict[str, Any]:
    return {
        "label": segment.label,
        "start_s": round_seconds(segment.start_s),
        "end_s": round_seconds(segment.end_s),
        "start": format_mmss(segment.start_s),
        "end": format_mmss(segment.end_s),
        "duration_s": round_seconds(segment.duration_s()),
    }


def response_to_dict(response: StudentResponse | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return {
        "start_s": round_seconds(response.start_s),
        "end_s": round_seconds(response.end_s),
        "start": format_mmss(response.start_s),
        "speaker": response.speaker,
        "certainty": response.certainty,
    }


def question_to_dict(question: Question) -> dict[str, Any]:
    wait_time_s = None
    if question.wait_time_s is not None:
        wait_time_s = round_seconds(question.wait_time_s)

    return {
        "text": question.text,
        "start_s": round_seconds(question.start_s),
        "question_end_s": round_seconds(question.question_end_s),
        "start": format_mmss(question.start_s),
        "speaker": question.speaker,
        "speaker_known": question.speaker_known,
        "reliable": question.reliable,
        "is_tag": question.is_tag,
        "llm_category": question.llm_category,
        "counts_as_question": not question.is_rhetorical(),
        "wait_time_s": wait_time_s,
        "wait_time_ok": question.wait_time_ok,
        "response": response_to_dict(question.response),
    }


def metric_to_dict(metric: RubricMetric) -> dict[str, Any]:
    return {
        "value": round_value(metric.value),
        "rubric": metric.rubrics,
        "description": metric.description,
        "note": metric.note,
    }


def list_of_dicts(
    items: list[Any], converter: Callable[[Any], dict[str, Any] | None]
) -> list[dict[str, Any] | None]:
    converted = []
    for item in items:
        converted.append(converter(item))
    return converted


def report_to_dict(report: ClassReport) -> dict[str, Any]:
    metadata = asdict(report.metadata)
    metadata["duration_s"] = round_seconds(report.metadata.duration_s)
    metadata["duration"] = format_mmss(report.metadata.duration_s)

    metrics = {}
    for metric in report.metrics:
        metrics[metric.name] = metric_to_dict(metric)

    data_quality = {}
    for name, value in asdict(report.data_quality).items():
        data_quality[name] = round_value(value)

    return {
        "metadata": metadata,
        "video_timeline": list_of_dicts(report.video_timeline, segment_to_dict),
        "stage_interaction_events": list_of_dicts(report.stage_interaction_events, segment_to_dict),
        "audio_events": list_of_dicts(report.questions, question_to_dict),
        "metrics": metrics,
        "data_quality": data_quality,
    }


def save_class_report(report: ClassReport, output_path: Path) -> None:
    write_json(report_to_dict(report), output_path)


def format_metric(metric: RubricMetric) -> str:
    value = metric.value
    if value is None:
        shown = "null"
    elif value is True:
        shown = "sí"
    elif value is False:
        shown = "no"
    elif isinstance(value, float):
        shown = f"{value:.3f}"
    else:
        shown = str(value)

    note = ""
    if metric.note:
        note = f"  ({metric.note})"
    rubrics = ",".join(metric.rubrics)
    return f"  [{rubrics:<5}] {metric.name:<28} {shown}{note}"


def format_question(question: Question) -> str:
    wait_text = "sin voz posterior"
    if question.wait_time_s is not None:
        wait_text = f"{question.wait_time_s:.1f} s"

    response_text = "sin respuesta"
    if question.response is not None:
        response_text = f"respuesta (certeza {question.response.certainty})"

    reliability = ""
    if not question.reliable:
        reliability = " [audio dudoso]"
    minute = format_mmss(question.start_s)
    return f"  [{minute}] espera {wait_text}, {response_text}{reliability}: {question.text}"


def format_summary(report: ClassReport) -> str:
    metadata = report.metadata
    lines = [
        f"Video: {metadata.video} ({format_mmss(metadata.duration_s)})",
        f"Diarización: {report.data_quality.diarization_note}",
        "",
        "Métricas:",
    ]
    for metric in report.metrics:
        lines.append(format_metric(metric))

    if report.data_quality.notes:
        lines.append("")
        lines.append("Avisos:")
        for note in report.data_quality.notes:
            lines.append(f"  - {note}")

    lines.append("")
    lines.append(f"Primeras {SUMMARY_QUESTION_COUNT} preguntas:")
    for question in report.questions[:SUMMARY_QUESTION_COUNT]:
        lines.append(format_question(question))
    return "\n".join(lines)
