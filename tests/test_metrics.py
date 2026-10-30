import pytest

from espejo.vision.geometry import BoundingBox
from espejo.vision.metrics import MetricsCalculator
from espejo.vision.schemas import (
    Behavior,
    Orientation,
    Segment,
    TeacherObservation,
    VisionMetrics,
)

FRAME_WIDTH = 800
FRAME_HEIGHT = 600


def make_observation(
    timestamp_s: float, box: BoundingBox | None, participation_events: int = 0
) -> TeacherObservation:
    return TeacherObservation(
        frame_index=int(timestamp_s * 2),
        timestamp_s=timestamp_s,
        teacher_box=box,
        behavior=Behavior.OTHER,
        orientation=Orientation.FACING,
        participation_events=participation_events,
    )


def calculate(observations: list[TeacherObservation]) -> VisionMetrics:
    # Clase inventada de 100 s: 30 guiando, 20 en la pizarra, 40 otra cosa, 10 sin docente
    behavior_timeline = [
        Segment(Behavior.GUIDE, 0, 30),
        Segment(Behavior.BOARD_WRITING, 30, 50),
        Segment(Behavior.OTHER, 50, 90),
        Segment(Behavior.NOT_VISIBLE, 90, 100),
    ]
    orientation_timeline = [
        Segment(Orientation.FACING, 0, 75),
        Segment(Orientation.BACK, 75, 90),
        Segment(Orientation.NOT_VISIBLE, 90, 100),
    ]
    calculator = MetricsCalculator(monitoring_threshold=0.25, grid_columns=8, grid_rows=6)
    return calculator.calculate(
        observations, behavior_timeline, orientation_timeline, FRAME_WIDTH, FRAME_HEIGHT
    )


def test_ratios_from_timelines() -> None:
    metrics = calculate([])

    assert metrics.analyzed_duration_s == pytest.approx(100)
    assert metrics.teacher_visible_ratio == pytest.approx(0.9)
    assert metrics.monitoring_ratio == pytest.approx(0.3)
    assert metrics.meets_monitoring_threshold is True
    assert metrics.board_writing_ratio == pytest.approx(0.2)
    assert metrics.back_to_class_ratio == pytest.approx(0.15)


def test_path_length_and_coverage() -> None:
    # El docente camina 400 px a la derecha y luego 300 px hacia abajo; un frame no se le ve
    observations = [
        make_observation(0.0, BoundingBox(0, 0, 100, 100)),
        make_observation(0.5, BoundingBox(400, 0, 500, 100)),
        make_observation(1.0, None),
        make_observation(1.5, BoundingBox(400, 300, 500, 400)),
    ]

    metrics = calculate(observations)

    assert metrics.path_length_normalized == pytest.approx(700 / FRAME_WIDTH)
    assert metrics.room_coverage_ratio == pytest.approx(3 / 48)


def test_participation_events_are_summed() -> None:
    observations = [
        make_observation(0.0, None, participation_events=2),
        make_observation(0.5, None, participation_events=1),
    ]

    assert calculate(observations).student_participation_events == 3


def test_monitoring_below_threshold() -> None:
    calculator = MetricsCalculator(monitoring_threshold=0.25, grid_columns=8, grid_rows=6)
    behavior_timeline = [Segment(Behavior.GUIDE, 0, 10), Segment(Behavior.OTHER, 10, 100)]

    metrics = calculator.calculate([], behavior_timeline, [], FRAME_WIDTH, FRAME_HEIGHT)

    assert metrics.monitoring_ratio == pytest.approx(0.1)
    assert metrics.meets_monitoring_threshold is False


def test_empty_timeline_gives_zero_ratios() -> None:
    calculator = MetricsCalculator(monitoring_threshold=0.25, grid_columns=8, grid_rows=6)
    metrics = calculator.calculate([], [], [], FRAME_WIDTH, FRAME_HEIGHT)
    assert metrics.monitoring_ratio == 0.0
    assert metrics.teacher_visible_ratio == 0.0
