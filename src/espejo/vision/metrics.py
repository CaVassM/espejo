"""Métricas de la clase a partir de las líneas de tiempo y del recorrido del docente."""

import math

from espejo.vision.geometry import FloorGrid, GridCell
from espejo.vision.schemas import (
    Behavior,
    Orientation,
    Segment,
    TeacherObservation,
    VisionMetrics,
)


class MetricsCalculator:
    """Los ratios se calculan sobre las líneas de tiempo suavizadas, no sobre frames sueltos."""

    def __init__(self, monitoring_threshold: float, grid_columns: int, grid_rows: int) -> None:
        self.monitoring_threshold = monitoring_threshold
        self.grid_columns = grid_columns
        self.grid_rows = grid_rows

    def calculate(
        self,
        observations: list[TeacherObservation],
        behavior_timeline: list[Segment],
        orientation_timeline: list[Segment],
        frame_width: int,
        frame_height: int,
    ) -> VisionMetrics:
        analyzed_duration_s = timeline_duration_s(behavior_timeline)
        not_visible_s = label_duration_s(behavior_timeline, Behavior.NOT_VISIBLE)
        visible_s = analyzed_duration_s - not_visible_s
        monitoring_ratio = label_ratio(behavior_timeline, Behavior.GUIDE)
        grid = FloorGrid(self.grid_columns, self.grid_rows, frame_width, frame_height)

        return VisionMetrics(
            analyzed_duration_s=analyzed_duration_s,
            teacher_visible_ratio=safe_ratio(visible_s, analyzed_duration_s),
            monitoring_ratio=monitoring_ratio,
            meets_monitoring_threshold=monitoring_ratio >= self.monitoring_threshold,
            board_writing_ratio=label_ratio(behavior_timeline, Behavior.BOARD_WRITING),
            back_to_class_ratio=label_ratio(orientation_timeline, Orientation.BACK),
            path_length_normalized=path_length_normalized(observations, frame_width),
            room_coverage_ratio=room_coverage_ratio(observations, grid),
            student_participation_events=count_participation_events(observations),
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


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def path_length_normalized(observations: list[TeacherObservation], frame_width: int) -> float:
    """Distancia recorrida por el foot_point, medida en anchos de frame."""
    total_px = 0.0
    previous_point = None

    for observation in observations:
        if observation.teacher_box is None:
            continue
        current_point = observation.teacher_box.foot_point()
        if previous_point is not None:
            total_px += math.dist(previous_point, current_point)
        previous_point = current_point

    return safe_ratio(total_px, frame_width)


def visited_cells(observations: list[TeacherObservation], grid: FloorGrid) -> set[GridCell]:
    cells = set()
    for observation in observations:
        if observation.teacher_box is None:
            continue
        foot_x, foot_y = observation.teacher_box.foot_point()
        cells.add(grid.cell_at(foot_x, foot_y))
    return cells


def room_coverage_ratio(observations: list[TeacherObservation], grid: FloorGrid) -> float:
    return safe_ratio(len(visited_cells(observations, grid)), grid.cell_count())


def count_participation_events(observations: list[TeacherObservation]) -> int:
    total_events = 0
    for observation in observations:
        total_events += observation.participation_events
    return total_events
