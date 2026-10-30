"""Salidas del módulo de audio: audio_report.json, transcript.txt y resumen de consola."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from espejo.audio.schemas import AudioReport, Question, SpeakerRole, TranscriptSegment, Word
from espejo.common.json_io import write_json
from espejo.common.time_format import format_mmss, round_seconds

RATIO_DECIMALS = 3
LOW_CONFIDENCE_TAG = "[AUDIO DUDOSO] "
# Etiqueta de cada rol en transcript.txt; sin diarización todos son "HABLANTE"
ROLE_LABELS = {
    SpeakerRole.TEACHER: "DOCENTE",
    SpeakerRole.STUDENT: "ESTUDIANTE",
    SpeakerRole.UNKNOWN: "HABLANTE",
}
SUMMARY_QUESTION_COUNT = 5


def word_to_dict(word: Word) -> dict[str, Any]:
    return {
        "text": word.text.strip(),
        "start_s": round_seconds(word.start_s),
        "end_s": round_seconds(word.end_s),
        "probability": round(word.probability, RATIO_DECIMALS),
    }


def segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
    words = []
    for word in segment.words:
        words.append(word_to_dict(word))

    return {
        "start_s": round_seconds(segment.start_s),
        "end_s": round_seconds(segment.end_s),
        "start": format_mmss(segment.start_s),
        "end": format_mmss(segment.end_s),
        "speaker": segment.speaker,
        "role": segment.role,
        "text": segment.text,
        "avg_logprob": round(segment.avg_logprob, RATIO_DECIMALS),
        "no_speech_prob": round(segment.no_speech_prob, RATIO_DECIMALS),
        "low_confidence": segment.low_confidence,
        "words": words,
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
        "role": question.role,
        "reliable": question.reliable,
        "wait_time_s": wait_time_s,
        "wait_time_ok": question.wait_time_ok,
    }


def metrics_to_dict(report: AudioReport) -> dict[str, Any]:
    if report.metrics is None:
        return {}

    rounded = {}
    for name, value in asdict(report.metrics).items():
        if isinstance(value, float):
            rounded[name] = round(value, RATIO_DECIMALS)
        else:
            rounded[name] = value
    rounded["analyzed_duration"] = format_mmss(report.metrics.analyzed_duration_s)
    return rounded


def report_to_dict(report: AudioReport) -> dict[str, Any]:
    segments = []
    for segment in report.segments:
        segments.append(segment_to_dict(segment))
    questions = []
    for question in report.questions:
        questions.append(question_to_dict(question))

    metadata = {
        "video": Path(report.video_path).name,
        "duration_s": round_seconds(report.duration_s),
        "duration": format_mmss(report.duration_s),
        "language": report.language,
        "whisper_model": report.transcriber.model_name,
        "device": report.transcriber.device,
        "compute_type": report.transcriber.compute_type,
        "diarization_used": report.diarization_used,
        "diarization_note": report.diarization_note,
    }
    return {
        "metadata": metadata,
        "segments": segments,
        "questions": questions,
        "metrics": metrics_to_dict(report),
    }


def save_report_json(report: AudioReport, output_path: Path) -> None:
    write_json(report_to_dict(report), output_path)


def transcript_line(segment: TranscriptSegment) -> str:
    """Ejemplo: "[03:12] DOCENTE: ¿Qué creen que pasará con el agua?"."""
    tag = ""
    if segment.low_confidence:
        tag = LOW_CONFIDENCE_TAG
    label = ROLE_LABELS[segment.role]
    return f"[{format_mmss(segment.start_s)}] {label}: {tag}{segment.text}"


def save_transcript(report: AudioReport, output_path: Path) -> None:
    lines = []
    for segment in report.segments:
        lines.append(transcript_line(segment))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_wait_time(question: Question) -> str:
    if question.wait_time_s is None:
        return "sin voz posterior"
    return f"{question.wait_time_s:.1f} s"


def format_question_line(question: Question) -> str:
    reliability = ""
    if not question.reliable:
        reliability = " [audio dudoso]"
    wait_text = format_wait_time(question)
    return f"  [{format_mmss(question.start_s)}] espera {wait_text}{reliability}: {question.text}"


def format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "sin datos"
    return f"{value:.1f} s"


def format_optional_ratio(value: float | None, reason: str | None) -> str:
    if value is not None:
        return f"{value:.1%}"
    if reason:
        return f"null ({reason})"
    return "null"


def format_audio_summary(report: AudioReport) -> str:
    metrics = report.metrics
    if metrics is None:
        return "Sin métricas."

    lines = [
        f"Video: {Path(report.video_path).name} ({format_mmss(report.duration_s)})",
        f"Whisper: {report.transcriber.model_name} en {report.transcriber.device}",
        f"Diarización: {report.diarization_note}",
        f"Voz en el audio:           {metrics.speech_ratio:.1%}",
        "Habla del docente:         "
        + format_optional_ratio(metrics.teacher_talk_ratio, metrics.teacher_talk_ratio_reason),
        f"Preguntas:                 {metrics.question_count} "
        f"({metrics.questions_per_minute:.2f} por minuto, {metrics.reliable_question_count} "
        "confiables)",
        f"Wait time medio:           {format_optional_seconds(metrics.mean_wait_time_s)}",
        f"Wait time mediano:         {format_optional_seconds(metrics.median_wait_time_s)}",
        "Espera suficiente (ok):    "
        + format_optional_ratio(metrics.wait_time_ok_ratio, "sin preguntas medibles"),
        f"Audio dudoso:              {metrics.low_confidence_ratio:.1%}",
        "",
        f"Primeras {SUMMARY_QUESTION_COUNT} preguntas:",
    ]
    for question in report.questions[:SUMMARY_QUESTION_COUNT]:
        lines.append(format_question_line(question))
    return "\n".join(lines)
