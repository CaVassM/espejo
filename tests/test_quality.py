import pytest

from espejo.audio.quality import AudioQualityChecker, low_confidence_duration_s
from espejo.audio.schemas import TranscriptSegment
from espejo.config import QualityConfig

CONFIG = QualityConfig(min_avg_logprob=-1.0, max_no_speech_prob=0.6)


def make_segment(
    start_s: float, end_s: float, avg_logprob: float, no_speech_prob: float
) -> TranscriptSegment:
    return TranscriptSegment(
        start_s=start_s,
        end_s=end_s,
        text="texto",
        words=[],
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


def test_clear_audio_is_trusted() -> None:
    segment = make_segment(0, 5, avg_logprob=-0.3, no_speech_prob=0.1)
    assert not AudioQualityChecker(CONFIG).is_low_confidence(segment)


def test_low_logprob_is_low_confidence() -> None:
    segment = make_segment(0, 5, avg_logprob=-1.4, no_speech_prob=0.1)
    assert AudioQualityChecker(CONFIG).is_low_confidence(segment)


def test_probable_silence_is_low_confidence() -> None:
    segment = make_segment(0, 5, avg_logprob=-0.3, no_speech_prob=0.8)
    assert AudioQualityChecker(CONFIG).is_low_confidence(segment)


def test_low_confidence_duration_sums_marked_segments() -> None:
    segments = [
        make_segment(0, 5, avg_logprob=-0.3, no_speech_prob=0.1),
        make_segment(5, 8, avg_logprob=-1.5, no_speech_prob=0.1),
        make_segment(8, 12, avg_logprob=-0.2, no_speech_prob=0.9),
    ]

    marked = AudioQualityChecker(CONFIG).mark(segments)

    assert low_confidence_duration_s(marked) == pytest.approx(7.0)
