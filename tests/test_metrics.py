import pytest

from espejo.config import PathConfig
from espejo.vision.geometry import BoundingBox
from espejo.vision.metrics import VisionMetricsCalculator, robust_path
from espejo.vision.schemas import Behavior, Segment, TeacherObservation, VisionMetrics

FRAME_WIDTH = 800
PATH_CONFIG = PathConfig(median_window_frames=1, max_step_ratio=0.25)
BOX_SIZE = 100


def box_with_foot_at(foot_x: float, foot_y: float) -> BoundingBox:
    """Caja cuyo punto de los pies cae exactamente en (foot_x, foot_y)."""
    half_width = BOX_SIZE / 2
    return BoundingBox(foot_x - half_width, foot_y - BOX_SIZE, foot_x + half_width, foot_y)


def make_observation(timestamp_s: float, box: BoundingBox | None) -> TeacherObservation:
    return TeacherObservation(
        frame_index=int(timestamp_s * 2),
        timestamp_s=timestamp_s,
        teacher_box=box,
        state=Behavior.OTHER,
        stage_interaction_present=False,
    )


def calculate(observations: list[TeacherObservation]) -> VisionMetrics:
    # Clase inventada de 100 s: 30 guiando, 20 en la pizarra, 40 otra cosa, 10 sin docente
    timeline = [
        Segment(Behavior.GUIDE, 0, 30),
        Segment(Behavior.BOARD_WRITING, 30, 50),
        Segment(Behavior.OTHER, 50, 90),
        Segment(Behavior.NOT_VISIBLE, 90, 100),
    ]
    stage_events = [Segment("stage_interaction", 10, 20), Segment("stage_interaction", 60, 65)]
    calculator = VisionMetricsCalculator(PATH_CONFIG)
    return calculator.calculate(observations, timeline, stage_events, FRAME_WIDTH)


def test_ratios_from_timeline() -> None:
    metrics = calculate([])

    assert metrics.analyzed_duration_s == pytest.approx(100)
    assert metrics.teacher_visible_ratio == pytest.approx(0.9)
    assert metrics.guide_ratio == pytest.approx(0.3)
    assert metrics.board_writing_ratio == pytest.approx(0.2)


def test_stage_interaction_counts_events_not_frames() -> None:
    assert calculate([]).stage_interaction_count == 2


def test_path_sums_valid_steps() -> None:
    # 100 px a la derecha y 150 px hacia abajo; un frame sin docente en medio no rompe el recorrido
    observations = [
        make_observation(0.0, box_with_foot_at(100, 300)),
        make_observation(0.5, box_with_foot_at(200, 300)),
        make_observation(1.0, None),
        make_observation(1.5, box_with_foot_at(200, 450)),
    ]

    path = robust_path(observations, FRAME_WIDTH, PATH_CONFIG)

    assert path.length_normalized == pytest.approx(250 / FRAME_WIDTH)
    assert path.discarded_jumps == 0


def test_large_jump_is_discarded_and_counted() -> None:
    # Un salto de 500 px en medio segundo no es una caminata (tope: 0.25 * 800 = 200 px)
    observations = [
        make_observation(0.0, box_with_foot_at(100, 300)),
        make_observation(0.5, box_with_foot_at(150, 300)),
        make_observation(1.0, box_with_foot_at(650, 300)),
        make_observation(1.5, box_with_foot_at(700, 300)),
    ]

    path = robust_path(observations, FRAME_WIDTH, PATH_CONFIG)

    assert path.length_normalized == pytest.approx(100 / FRAME_WIDTH)
    assert path.discarded_jumps == 1


def test_rolling_median_removes_box_jitter() -> None:
    # La caja tiembla 40 px un solo frame; con mediana de 3 el temblor no suma recorrido
    config = PathConfig(median_window_frames=3, max_step_ratio=0.25)
    observations = [
        make_observation(0.0, box_with_foot_at(400, 300)),
        make_observation(0.5, box_with_foot_at(400, 300)),
        make_observation(1.0, box_with_foot_at(440, 300)),
        make_observation(1.5, box_with_foot_at(400, 300)),
        make_observation(2.0, box_with_foot_at(400, 300)),
    ]

    path = robust_path(observations, FRAME_WIDTH, config)

    assert path.length_normalized == pytest.approx(0.0)


def test_empty_timeline_gives_zero_ratios() -> None:
    calculator = VisionMetricsCalculator(PATH_CONFIG)
    metrics = calculator.calculate([], [], [], FRAME_WIDTH)
    assert metrics.guide_ratio == 0.0
    assert metrics.teacher_visible_ratio == 0.0
