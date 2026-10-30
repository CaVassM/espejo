import pytest

from espejo.vision.report_writer import format_mmss
from espejo.vision.schemas import Segment
from espejo.vision.timeline import TimedLabel, TimelineBuilder

SAMPLE_INTERVAL_S = 0.5


def make_labels(labels: list[str]) -> list[TimedLabel]:
    """Etiquetas muestreadas a 2 fps, empezando en t=0."""
    timed_labels = []
    for position, label in enumerate(labels):
        timed_labels.append(TimedLabel(position * SAMPLE_INTERVAL_S, label))
    return timed_labels


def test_smoothing_removes_isolated_flicker() -> None:
    builder = TimelineBuilder(smoothing_window_s=3.0, min_segment_s=0.0)
    labels = ["guide"] * 5 + ["other"] + ["guide"] * 5

    smoothed = builder.smooth(make_labels(labels))

    smoothed_values = []
    for item in smoothed:
        smoothed_values.append(item.label)
    assert smoothed_values == ["guide"] * 11


def test_build_produces_contiguous_segments() -> None:
    builder = TimelineBuilder(smoothing_window_s=1.0, min_segment_s=0.0)
    labels = ["guide"] * 6 + ["board_writing"] * 6

    segments = builder.build(make_labels(labels), end_s=6.0)

    assert segments == [
        Segment("guide", 0.0, 3.0),
        Segment("board_writing", 3.0, 6.0),
    ]


def test_short_segment_merges_into_longer_neighbor() -> None:
    builder = TimelineBuilder(smoothing_window_s=3.0, min_segment_s=1.5)
    segments = [
        Segment("guide", 0.0, 10.0),
        Segment("other", 10.0, 11.0),
        Segment("board_writing", 11.0, 13.0),
    ]

    merged = builder.merge_short_segments(segments)

    assert merged == [
        Segment("guide", 0.0, 11.0),
        Segment("board_writing", 11.0, 13.0),
    ]


def test_merge_joins_neighbors_with_same_label() -> None:
    builder = TimelineBuilder(smoothing_window_s=3.0, min_segment_s=1.5)
    segments = [
        Segment("guide", 0.0, 5.0),
        Segment("other", 5.0, 5.5),
        Segment("guide", 5.5, 9.0),
    ]

    merged = builder.merge_short_segments(segments)

    assert merged == [Segment("guide", 0.0, 9.0)]


def test_timeline_covers_whole_analysis() -> None:
    builder = TimelineBuilder(smoothing_window_s=3.0, min_segment_s=1.5)
    labels = ["guide", "other", "back", "guide", "other"] * 8

    segments = builder.build(make_labels(labels), end_s=20.0)

    assert segments[0].start_s == 0.0
    assert segments[-1].end_s == pytest.approx(20.0)
    for previous, current in zip(segments, segments[1:], strict=False):
        assert previous.end_s == current.start_s


def test_empty_input_gives_empty_timeline() -> None:
    builder = TimelineBuilder(smoothing_window_s=3.0, min_segment_s=1.5)
    assert builder.build([], end_s=10.0) == []


def test_format_mmss() -> None:
    assert format_mmss(0) == "00:00"
    assert format_mmss(75.4) == "01:15"
    assert format_mmss(3725) == "62:05"
