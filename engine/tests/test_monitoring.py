import pytest

from espejo.analysis.monitoring import (
    TimeInterval,
    compute_monitoring,
    guide_intervals,
    question_episodes,
    union_duration_s,
)
from espejo.audio.schemas import Question, ResponseCertainty, SpeakerRole, StudentResponse
from espejo.vision.schemas import Behavior, Segment

DURATION_S = 100.0
THRESHOLD = 0.25
MAX_EPISODE_S = 15.0


def make_question(
    start_s: float,
    end_s: float,
    response: StudentResponse | None = None,
    wait_time_s: float | None = None,
    reliable: bool = True,
    is_tag: bool = False,
) -> Question:
    return Question(
        text="¿Qué observan?",
        start_s=start_s,
        question_end_s=end_s,
        speaker="unknown",
        role=SpeakerRole.UNKNOWN,
        reliable=reliable,
        is_tag=is_tag,
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


def answer(start_s: float, end_s: float) -> StudentResponse:
    return StudentResponse(start_s, end_s, "SPEAKER_01", ResponseCertainty.HIGH)


def test_episode_goes_from_question_to_end_of_answer() -> None:
    questions = [make_question(10.0, 12.0, response=answer(14.0, 18.0))]
    assert question_episodes(questions, MAX_EPISODE_S, DURATION_S) == [TimeInterval(10, 18)]


def test_long_answer_is_capped() -> None:
    questions = [make_question(10.0, 12.0, response=answer(14.0, 60.0))]
    assert question_episodes(questions, MAX_EPISODE_S, DURATION_S) == [TimeInterval(10, 25)]


def test_unanswered_questions_do_not_count() -> None:
    questions = [make_question(10.0, 12.0, wait_time_s=4.0)]
    assert question_episodes(questions, MAX_EPISODE_S, DURATION_S) == []


def test_tag_questions_and_unreliable_audio_do_not_count() -> None:
    questions = [
        make_question(10.0, 11.0, response=answer(12.0, 13.0), is_tag=True),
        make_question(30.0, 32.0, response=answer(33.0, 35.0), reliable=False),
    ]
    assert question_episodes(questions, MAX_EPISODE_S, DURATION_S) == []
