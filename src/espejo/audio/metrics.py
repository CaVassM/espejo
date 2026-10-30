"""Métricas del habla en clase: preguntas, tiempo de espera y respuestas."""

import statistics

from espejo.audio.quality import low_confidence_duration_s
from espejo.audio.schemas import (
    AudioMetrics,
    Question,
    SpeakerRole,
    SpeechInterval,
    TranscriptSegment,
)
from espejo.common.ratios import safe_ratio

SECONDS_PER_MINUTE = 60
NO_DIARIZATION_REASON = (
    "sin diarización no se puede separar la voz del docente de la de los alumnos"
)


class AudioMetricsCalculator:
    def calculate(
        self,
        segments: list[TranscriptSegment],
        questions: list[Question],
        speech_intervals: list[SpeechInterval],
        duration_s: float,
        diarization_used: bool,
    ) -> AudioMetrics:
        wait_times_s = reliable_wait_times_s(questions)
        response_count = count_responses(questions)

        teacher_talk_ratio = None
        teacher_talk_ratio_reason = NO_DIARIZATION_REASON
        if diarization_used:
            teacher_talk_ratio = safe_ratio(teacher_talk_s(segments), duration_s)
            teacher_talk_ratio_reason = None

        return AudioMetrics(
            analyzed_duration_s=duration_s,
            speech_ratio=safe_ratio(total_speech_s(speech_intervals), duration_s),
            teacher_talk_ratio=teacher_talk_ratio,
            teacher_talk_ratio_reason=teacher_talk_ratio_reason,
            teacher_question_count=len(questions),
            reliable_question_count=count_reliable(questions),
            questions_per_minute=safe_ratio(len(questions), duration_s / SECONDS_PER_MINUTE),
            mean_wait_time_s=mean_or_none(wait_times_s),
            median_wait_time_s=median_or_none(wait_times_s),
            wait_time_ok_ratio=wait_time_ok_ratio(questions),
            student_response_count=response_count,
            response_rate=ratio_or_none(response_count, len(questions)),
            low_confidence_ratio=safe_ratio(low_confidence_duration_s(segments), duration_s),
        )


def total_speech_s(intervals: list[SpeechInterval]) -> float:
    total_s = 0.0
    for interval in intervals:
        total_s += interval.duration_s()
    return total_s


def teacher_talk_s(segments: list[TranscriptSegment]) -> float:
    total_s = 0.0
    for segment in segments:
        if segment.role == SpeakerRole.TEACHER:
            total_s += segment.duration_s()
    return total_s


def count_reliable(questions: list[Question]) -> int:
    total = 0
    for question in questions:
        if question.reliable:
            total += 1
    return total


def count_responses(questions: list[Question]) -> int:
    total = 0
    for question in questions:
        if question.response is not None:
            total += 1
    return total


def reliable_wait_times_s(questions: list[Question]) -> list[float]:
    """Solo preguntas en audio confiable y con voz posterior; las demás no son medibles."""
    wait_times_s = []
    for question in questions:
        if question.reliable and question.wait_time_s is not None:
            wait_times_s.append(question.wait_time_s)
    return wait_times_s


def wait_time_ok_ratio(questions: list[Question]) -> float | None:
    measured = 0
    ok_count = 0
    for question in questions:
        if not question.reliable or question.wait_time_ok is None:
            continue
        measured += 1
        if question.wait_time_ok:
            ok_count += 1
    return ratio_or_none(ok_count, measured)


def ratio_or_none(numerator: int, denominator: int) -> float | None:
    """None cuando no hay nada que medir: un 0 % sería engañoso."""
    if denominator == 0:
        return None
    return numerator / denominator


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)
