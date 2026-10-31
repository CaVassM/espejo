import pytest

from espejo.audio.responses import StudentResponseFinder
from espejo.audio.schemas import ResponseCertainty, SpeakerTurn, SpeechInterval
from espejo.config import ResponsesConfig

FINDER = StudentResponseFinder(ResponsesConfig(response_window_s=10.0), min_gap_s=0.5)
QUESTION_END_S = 20.0
TEACHER = "SPEAKER_00"
STUDENT = "SPEAKER_01"


def test_with_diarization_first_other_speaker_answers() -> None:
    # El docente sigue hablando un poco y luego responde un alumno
    turns = [
        SpeakerTurn(TEACHER, 15.0, 22.0),
        SpeakerTurn(STUDENT, 23.0, 26.0),
    ]

    response = FINDER.response_with_diarization(QUESTION_END_S, turns, TEACHER)

    assert response is not None
    assert response.speaker == STUDENT
    assert response.start_s == pytest.approx(23.0)
    assert response.certainty == ResponseCertainty.HIGH


def test_with_diarization_teacher_voice_is_not_a_response() -> None:
    turns = [SpeakerTurn(TEACHER, 15.0, 20.0), SpeakerTurn(TEACHER, 23.0, 30.0)]
    assert FINDER.response_with_diarization(QUESTION_END_S, turns, TEACHER) is None


def test_with_diarization_late_answer_is_outside_the_window() -> None:
    turns = [SpeakerTurn(STUDENT, 35.0, 37.0)]
    assert FINDER.response_with_diarization(QUESTION_END_S, turns, TEACHER) is None


def test_without_diarization_any_voice_is_a_low_certainty_response() -> None:
    intervals = [SpeechInterval(15.0, 20.0), SpeechInterval(24.0, 27.0)]

    response = FINDER.response_without_diarization(QUESTION_END_S, intervals)

    assert response is not None
    assert response.start_s == pytest.approx(24.0)
    assert response.certainty == ResponseCertainty.LOW


def test_without_diarization_silence_means_no_response() -> None:
    intervals = [SpeechInterval(15.0, 20.0), SpeechInterval(40.0, 42.0)]
    assert FINDER.response_without_diarization(QUESTION_END_S, intervals) is None
