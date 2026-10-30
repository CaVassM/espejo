import pytest

from espejo.audio.metrics import AudioMetricsCalculator
from espejo.audio.schemas import (
    Question,
    ResponseCertainty,
    SpeakerRole,
    SpeechInterval,
    StudentResponse,
    TranscriptSegment,
)

DURATION_S = 120.0
RESPONSE = StudentResponse(5.0, 7.0, "unknown", ResponseCertainty.LOW)


def make_segment(
    start_s: float, end_s: float, role: SpeakerRole, low_confidence: bool = False
) -> TranscriptSegment:
    return TranscriptSegment(
        start_s=start_s,
        end_s=end_s,
        text="texto",
        words=[],
        avg_logprob=-0.3,
        no_speech_prob=0.1,
        role=role,
        low_confidence=low_confidence,
    )


def make_question(
    wait_time_s: float | None, reliable: bool = True, answered: bool = False
) -> Question:
    wait_time_ok = None
    if wait_time_s is not None:
        wait_time_ok = wait_time_s >= 3.0
    response = None
    if answered:
        response = RESPONSE
    return Question(
        text="¿Qué observan?",
        start_s=0.0,
        question_end_s=1.0,
        speaker="unknown",
        role=SpeakerRole.UNKNOWN,
        reliable=reliable,
        wait_time_s=wait_time_s,
        wait_time_ok=wait_time_ok,
        response=response,
    )


def test_metrics_without_diarization() -> None:
    # Clase inventada de 2 minutos: 60 s de voz, 4 preguntas (una en audio dudoso), 2 respondidas
    segments = [
        make_segment(0, 30, SpeakerRole.UNKNOWN),
        make_segment(30, 42, SpeakerRole.UNKNOWN, low_confidence=True),
    ]
    questions = [
        make_question(4.0, answered=True),
        make_question(1.0, answered=True),
        make_question(None),
        make_question(10.0, reliable=False),
    ]
    intervals = [SpeechInterval(0, 40), SpeechInterval(50, 70)]

    metrics = AudioMetricsCalculator().calculate(
        segments, questions, intervals, DURATION_S, diarization_used=False
    )

    assert metrics.speech_ratio == pytest.approx(0.5)
    assert metrics.teacher_talk_ratio is None
    assert metrics.teacher_talk_ratio_reason is not None
    assert metrics.teacher_question_count == 4
    assert metrics.questions_per_minute == pytest.approx(2.0)
    assert metrics.reliable_question_count == 3
    assert metrics.mean_wait_time_s == pytest.approx(2.5)
    assert metrics.median_wait_time_s == pytest.approx(2.5)
    assert metrics.wait_time_ok_ratio == pytest.approx(0.5)
    assert metrics.student_response_count == 2
    assert metrics.response_rate == pytest.approx(0.5)
    assert metrics.low_confidence_ratio == pytest.approx(0.1)


def test_teacher_talk_ratio_with_diarization() -> None:
    segments = [
        make_segment(0, 60, SpeakerRole.TEACHER),
        make_segment(60, 72, SpeakerRole.STUDENT),
    ]

    metrics = AudioMetricsCalculator().calculate(
        segments, [], [], DURATION_S, diarization_used=True
    )

    assert metrics.teacher_talk_ratio == pytest.approx(0.5)
    assert metrics.teacher_talk_ratio_reason is None


def test_no_questions_gives_null_rates() -> None:
    metrics = AudioMetricsCalculator().calculate([], [], [], DURATION_S, diarization_used=False)

    assert metrics.teacher_question_count == 0
    assert metrics.mean_wait_time_s is None
    assert metrics.wait_time_ok_ratio is None
    assert metrics.response_rate is None
