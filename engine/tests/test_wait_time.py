import pytest

from espejo.audio.schemas import Question, SpeakerRole, SpeechInterval
from espejo.audio.wait_time import WaitTimeCalculator
from espejo.config import WaitTimeConfig

CONFIG = WaitTimeConfig(min_gap_s=0.5, max_wait_s=15.0, ok_threshold_s=3.0)
QUESTION_END_S = 10.0


def make_question(question_end_s: float) -> Question:
    return Question(
        text="¿Qué observan?",
        start_s=question_end_s - 1.0,
        question_end_s=question_end_s,
        speaker="unknown",
        role=SpeakerRole.UNKNOWN,
        reliable=True,
    )


def test_answer_after_four_seconds() -> None:
    intervals = [SpeechInterval(8.0, 10.1), SpeechInterval(14.0, 16.0)]

    wait_time_s = WaitTimeCalculator(CONFIG).wait_time_s(QUESTION_END_S, intervals)

    assert wait_time_s == pytest.approx(4.0)


def test_short_internal_pause_is_ignored() -> None:
    # 0.3 s después de la pregunta hay un ruido corto; la respuesta real llega a los 4 s
    intervals = [
        SpeechInterval(8.0, 10.0),
        SpeechInterval(10.3, 10.5),
        SpeechInterval(14.0, 16.0),
    ]

    wait_time_s = WaitTimeCalculator(CONFIG).wait_time_s(QUESTION_END_S, intervals)

    assert wait_time_s == pytest.approx(4.0)


def test_no_speech_after_question_is_none() -> None:
    intervals = [SpeechInterval(8.0, 10.0)]
    assert WaitTimeCalculator(CONFIG).wait_time_s(QUESTION_END_S, intervals) is None


def test_wait_time_is_capped() -> None:
    intervals = [SpeechInterval(8.0, 10.0), SpeechInterval(40.0, 42.0)]

    wait_time_s = WaitTimeCalculator(CONFIG).wait_time_s(QUESTION_END_S, intervals)

    assert wait_time_s == pytest.approx(CONFIG.max_wait_s)


def test_teacher_who_keeps_talking_did_not_wait() -> None:
    # La voz que contiene el fin de la pregunta sigue 5 s más: no hubo pausa
    intervals = [SpeechInterval(8.0, 15.0), SpeechInterval(20.0, 22.0)]

    wait_time_s = WaitTimeCalculator(CONFIG).wait_time_s(QUESTION_END_S, intervals)

    assert wait_time_s == 0.0


def test_annotate_marks_wait_time_ok() -> None:
    intervals = [SpeechInterval(14.0, 15.0), SpeechInterval(21.0, 22.0)]
    questions = [make_question(10.0), make_question(20.0)]

    annotated = WaitTimeCalculator(CONFIG).annotate(questions, intervals)

    assert annotated[0].wait_time_ok is True
    assert annotated[1].wait_time_s == pytest.approx(1.0)
    assert annotated[1].wait_time_ok is False


def test_annotate_leaves_ok_as_none_without_speech() -> None:
    annotated = WaitTimeCalculator(CONFIG).annotate([make_question(10.0)], [])
    assert annotated[0].wait_time_s is None
    assert annotated[0].wait_time_ok is None
