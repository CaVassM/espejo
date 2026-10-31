"""class_report.json y el resumen que se imprime en consola."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from espejo.analysis.schemas import ClassReport, MetricValue, RubricEvidence, RubricMetric
from espejo.audio.schemas import Question, StudentResponse
from espejo.common.json_io import write_json
from espejo.common.time_format import format_mmss, round_seconds
from espejo.llm.schemas import AiTip, Evidence, LlmAnalysis, RejectedEvidence, Suggestion
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


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "timestamp_s": round_seconds(evidence.timestamp_s),
        "timestamp": format_mmss(evidence.timestamp_s),
        "label": evidence.label,
        "quote": evidence.quote,
        "reason": evidence.reason,
        "confidence": evidence.confidence,
        "question_id": evidence.question_id,
        "addressee": evidence.addressee,
    }


def rejected_to_dict(rejected: RejectedEvidence) -> dict[str, Any]:
    converted = evidence_to_dict(rejected.evidence)
    converted["rejection_reason"] = rejected.reason
    return converted


def llm_to_dict(analysis: LlmAnalysis) -> dict[str, Any]:
    """Proveedor, modelo, tokens, tiempo y costo (0 con Ollama), y lo que el LLM marcó."""
    usage = asdict(analysis.usage)
    usage["seconds"] = round_seconds(analysis.usage.seconds)
    question_labels = list(analysis.question_labels.values())
    return {
        "used": analysis.used,
        "note": analysis.note,
        "provider": analysis.provider,
        "model": analysis.model,
        "usage": usage,
        "question_labels": list_of_dicts(question_labels, evidence_to_dict),
        "feedback": list_of_dicts(analysis.feedback, evidence_to_dict),
        "regulation": list_of_dicts(analysis.regulation, evidence_to_dict),
        "participation": list_of_dicts(analysis.participation, evidence_to_dict),
        "respect": list_of_dicts(analysis.respect, evidence_to_dict),
        "rejected_evidence": list_of_dicts(analysis.rejected, rejected_to_dict),
        "one_suggestion": suggestion_to_dict(analysis.suggestion),
        "ai_tips": list_of_dicts(analysis.ai_tips, ai_tip_to_dict),
    }


def ai_tip_to_dict(tip: AiTip) -> dict[str, Any]:
    return {
        "title": tip.title,
        "what_for": tip.what_for,
        "how": tip.how,
        "example_prompt": tip.example_prompt,
    }


def suggestion_to_dict(suggestion: Suggestion | None) -> dict[str, Any] | None:
    if suggestion is None:
        return None
    focus = suggestion.focus
    timestamp = None
    if focus.timestamp_s is not None:
        timestamp = format_mmss(focus.timestamp_s)
    return {
        "weakness": focus.weakness.value,
        "timestamp": timestamp,
        "quote": focus.quote,
        "what_we_saw": focus.what_we_saw,
        "why_it_matters": suggestion.why_it_matters,
        "try_this": suggestion.try_this,
        "example_phrase": suggestion.example_phrase,
        "text": suggestion.as_text(),
    }


def rubric_evidence_to_dict(item: RubricEvidence) -> dict[str, Any]:
    metrics = []
    for metric in item.metrics:
        metrics.append({"name": metric.name, "value": round_value(metric.value)})
    citations = []
    for citation in item.citations:
        citations.append(
            {
                "timestamp": format_mmss(citation.timestamp_s),
                "label": citation.label,
                "quote": citation.quote,
                "reason": citation.reason,
            }
        )
    return {
        "rubric": item.rubric,
        "title": item.title,
        "status": item.status,
        "metrics": metrics,
        "citations": citations,
    }


def report_to_dict(report: ClassReport) -> dict[str, Any]:
    llm_analysis = llm_to_dict(report.llm)
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
        "rubric_evidence": list_of_dicts(report.rubric_evidence, rubric_evidence_to_dict),
        "llm_analysis": llm_analysis,
        "one_suggestion": llm_analysis["one_suggestion"],
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
        f"LLM: {report.llm.note}",
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

    suggestion = report.llm.suggestion
    if suggestion is not None:
        lines.append("")
        lines.append("Consejo:")
        lines.append(suggestion.as_text())
    return "\n".join(lines)
