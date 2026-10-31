"""Aplica la clasificación del LLM al audio y recalcula sus métricas."""

from dataclasses import replace

from espejo.analysis.report_builder import AudioSide
from espejo.audio.metrics import AudioMetricsCalculator
from espejo.llm.analyzer import apply_question_labels
from espejo.llm.schemas import LlmAnalysis


def with_llm_labels(audio: AudioSide, analysis: LlmAnalysis) -> AudioSide:
    """Con LLM, las preguntas retóricas que él marque dejan de contar en espera y monitoreo."""
    if not analysis.used:
        return audio

    interpretation = audio.interpretation
    questions = apply_question_labels(interpretation.questions, analysis.question_labels)
    metrics = AudioMetricsCalculator().calculate(
        interpretation.segments,
        questions,
        audio.capture.speech_intervals,
        audio.capture.duration_s,
        diarization_used=interpretation.diarization_used,
    )
    updated = replace(interpretation, questions=questions, metrics=metrics)
    return AudioSide(audio.capture, updated)
