import pytest

from espejo.analysis.monitoring import (
    TimeInterval,
    compute_monitoring,
    episode_end_s,
    guide_intervals,
    question_episodes,
    union_duration_s,
)
from espejo.audio.schemas import Question, ResponseCertainty, SpeakerRole, StudentResponse
from espejo.vision.schemas import Behavior, Segment

DURATION_S = 100.0
THRESHOLD = 0.25
MAX_WAIT_S = 15.0


def make_question(
    start_s: float,
    end_s: float,
    response: StudentResponse | None = None,
    wait_time_s: float | None = None,
    reliable: bool = True,
) -> Question:
    return Question(
        text="¿Qué observan?",
        start_s=start_s,
        question_end_s=end_s,
        speaker="unknown",
        role=SpeakerRole.UNKNOWN,
        reliable=reliable,
        wait_time_s=wait_time_s,
        response=response,
    )


def test_overlapping_intervals_are_not_double_counted() -> None:
    intervals = [TimeInterval(0, 10), TimeInterval(5, 15)]
    assert union_duration_s(intervals) == pytest.approx(15)


def test_separate_intervals_add_up() -> None:
    intervals = [TimeInterval(0, 10), TimeInterval(20, 25)]
    assert union_duration_s(intervals) == pytest.approx(15)


def test_video_and_audio_union_without_double_counting() -> None:
    # Guía 0-20 s (video); pregunta 15-25 s (audio): se solapan 5 s, la unión es 25 s
    video = [TimeInterval(0, 20)]
    audio = [TimeInterval(15, 25)]

    result = compute_monitoring(video, audio, DURATION_S, THRESHOLD)

    assert result.video_ratio == pytest.approx(0.20)
    assert result.audio_ratio == pytest.approx(0.10)
    assert result.ratio == pytest.approx(0.25)
    assert result.meets_threshold is True


def test_without_audio_uses_only_video() -> None:
    result = compute_monitoring([TimeInterval(0, 10)], None, DURATION_S, THRESHOLD)

    assert result.audio_ratio is None
    assert result.ratio == pytest.approx(0.10)
    assert result.meets_threshold is False


def test_guide_intervals_come_from_guide_segments() -> None:
    timeline = [
        Segment(Behavior.GUIDE, 0, 10),
        Segment(Behavior.OTHER, 10, 20),
        Segment(Behavior.GUIDE, 20, 25),
    ]
    assert guide_intervals(timeline) == [TimeInterval(0, 10), TimeInterval(20, 25)]


def test_episode_ends_with_the_response() -> None:
    response = StudentResponse(14.0, 18.0, "SPEAKER_01", ResponseCertainty.HIGH)
    question = make_question(10.0, 12.0, response=response, wait_time_s=2.0)
    assert episode_end_s(question, MAX_WAIT_S) == pytest.approx(18.0)


def test_episode_without_response_ends_after_the_wait() -> None:
    question = make_question(10.0, 12.0, wait_time_s=4.0)
    assert episode_end_s(question, MAX_WAIT_S) == pytest.approx(16.0)


def test_unreliable_questions_are_left_out() -> None:
    questions = [make_question(10, 12, wait_time_s=1.0), make_question(30, 32, reliable=False)]

    episodes = question_episodes(questions, MAX_WAIT_S, DURATION_S)

    assert episodes == [TimeInterval(10, 13)]
