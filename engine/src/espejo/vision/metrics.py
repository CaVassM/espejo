"""Métricas de video a partir de la línea de tiempo y del recorrido del docente."""

import statistics

from espejo.common.ratios import safe_ratio
from espejo.config import PathConfig
from espejo.vision.geometry import Point
from espejo.vision.schemas import (
    Behavior,
    PathSummary,
    Segment,
    TeacherObservation,
    VisionMetrics,
)


class VisionMetricsCalculator:
    """Los ratios se calculan sobre la línea de tiempo suavizada, no sobre frames sueltos."""

    def __init__(self, path_config: PathConfig) -> None:
        self.path_config = path_config

    def calculate(
        self,
        observations: list[TeacherObservation],
        timeline: list[Segment],
        stage_interaction_events: list[Segment],
        frame_width: int,
    ) -> VisionMetrics:
        analyzed_duration_s = timeline_duration_s(timeline)
        visible_s = analyzed_duration_s - label_duration_s(timeline, Behavior.NOT_VISIBLE)

        return VisionMetrics(
            analyzed_duration_s=analyzed_duration_s,
            teacher_visible_ratio=safe_ratio(visible_s, analyzed_duration_s),
            interpolated_ratio=interpolated_ratio(observations),
            board_writing_ratio=label_ratio(timeline, Behavior.BOARD_WRITING),
            guide_ratio=label_ratio(timeline, Behavior.GUIDE),
            stage_interaction_count=len(stage_interaction_events),
            path=robust_path(observations, frame_width, self.path_config),
        )


def timeline_duration_s(timeline: list[Segment]) -> float:
    total_s = 0.0
    for segment in timeline:
        total_s += segment.duration_s()
    return total_s


def label_duration_s(timeline: list[Segment], label: str) -> float:
    total_s = 0.0
    for segment in timeline:
        if segment.label == label:
            total_s += segment.duration_s()
    return total_s


def label_ratio(timeline: list[Segment], label: str) -> float:
    """Fracción de la duración analizada con esa etiqueta."""
    return safe_ratio(label_duration_s(timeline, label), timeline_duration_s(timeline))


def interpolated_ratio(observations: list[TeacherObservation]) -> float:
    """Fracción de frames en que la posición del docente se rellenó con la última conocida."""
    interpolated_count = 0
    for observation in observations:
        if observation.interpolated:
            interpolated_count += 1
    return safe_ratio(interpolated_count, len(observations))


def visible_foot_points(observations: list[TeacherObservation]) -> list[Point]:
    points = []
    for observation in observations:
        if observation.teacher_box is None:
            continue
        foot_x, foot_y = observation.teacher_box.foot_point()
        points.append(Point(foot_x, foot_y))
    return points


def rolling_median(points: list[Point], window: int) -> list[Point]:
    """Mediana en una ventana centrada: quita el temblor de la caja sin borrar la caminata."""
    half_window = window // 2
    smoothed = []
    for index in range(len(points)):
        neighbors = points[max(0, index - half_window) : index + half_window + 1]
        x_values = []
        y_values = []
        for neighbor in neighbors:
            x_values.append(neighbor.x)
            y_values.append(neighbor.y)
        smoothed.append(Point(statistics.median(x_values), statistics.median(y_values)))
    return smoothed


def robust_path(
    observations: list[TeacherObservation], frame_width: int, config: PathConfig
) -> PathSummary:
    """Recorrido en anchos de frame; los saltos imposibles se descartan y se cuentan."""
    points = rolling_median(visible_foot_points(observations), config.median_window_frames)
    max_step_px = config.max_step_ratio * frame_width

    total_px = 0.0
    discarded_jumps = 0
    for previous, current in zip(points, points[1:], strict=False):
        step_px = previous.distance_to(current)
        if step_px > max_step_px:
            discarded_jumps += 1
            continue
        total_px += step_px

    return PathSummary(safe_ratio(total_px, frame_width), discarded_jumps)
