"""Convierte etiquetas por frame en segmentos de tiempo suavizados."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from espejo.vision.schemas import Segment


@dataclass(frozen=True)
class TimedLabel:
    """Etiqueta de un frame muestreado en un instante del video."""

    timestamp_s: float
    label: str


class TimelineBuilder:
    """Suaviza las etiquetas con voto mayoritario y elimina segmentos demasiado cortos."""

    def __init__(self, smoothing_window_s: float, min_segment_s: float) -> None:
        self.smoothing_window_s = smoothing_window_s
        self.min_segment_s = min_segment_s

    def build(self, timed_labels: list[TimedLabel], end_s: float) -> list[Segment]:
        """Las etiquetas deben venir ordenadas por tiempo; end_s cierra el último segmento."""
        if not timed_labels:
            return []

        smoothed_labels = self.smooth(timed_labels)
        segments = group_into_segments(smoothed_labels, end_s)
        return self.merge_short_segments(segments)

    def smooth(self, timed_labels: list[TimedLabel]) -> list[TimedLabel]:
        """Cada frame toma la etiqueta más votada en una ventana centrada en él."""
        timestamps = []
        for item in timed_labels:
            timestamps.append(item.timestamp_s)

        half_window_s = self.smoothing_window_s / 2
        smoothed_labels = []
        for item in timed_labels:
            first_index = bisect_left(timestamps, item.timestamp_s - half_window_s)
            last_index = bisect_right(timestamps, item.timestamp_s + half_window_s)
            window = timed_labels[first_index:last_index]

            winner = majority_label(window, preferred_label=item.label)
            smoothed_labels.append(TimedLabel(item.timestamp_s, winner))
        return smoothed_labels

    def merge_short_segments(self, segments: list[Segment]) -> list[Segment]:
        """Absorbe el segmento más corto en su vecino hasta que todos superen el mínimo."""
        merged_segments = join_adjacent_equal_labels(segments)

        while len(merged_segments) > 1:
            shortest_index = index_of_shortest(merged_segments)
            if merged_segments[shortest_index].duration_s() >= self.min_segment_s:
                break
            merged_segments = absorb_into_neighbor(merged_segments, shortest_index)

        return merged_segments


def majority_label(window: list[TimedLabel], preferred_label: str) -> str:
    """En caso de empate conserva la etiqueta original para no inventar cambios."""
    counts: dict[str, int] = {}
    for item in window:
        counts[item.label] = counts.get(item.label, 0) + 1

    best_count = max(counts.values())
    if counts.get(preferred_label, 0) == best_count:
        return preferred_label

    for item in window:
        if counts[item.label] == best_count:
            return item.label
    return preferred_label


def group_into_segments(timed_labels: list[TimedLabel], end_s: float) -> list[Segment]:
    """Cada segmento dura hasta el primer frame con una etiqueta distinta."""
    segments = []
    current_label = timed_labels[0].label
    current_start_s = timed_labels[0].timestamp_s

    for item in timed_labels[1:]:
        if item.label == current_label:
            continue
        segments.append(Segment(current_label, current_start_s, item.timestamp_s))
        current_label = item.label
        current_start_s = item.timestamp_s

    segments.append(Segment(current_label, current_start_s, end_s))
    return segments


def relabel_short_segments(
    segments: list[Segment], label: str, min_duration_s: float, replacement: str
) -> list[Segment]:
    """Los segmentos de esa etiqueta más cortos que min_duration_s pasan a ser replacement."""
    relabeled = []
    for segment in segments:
        if segment.label == label and segment.duration_s() < min_duration_s:
            relabeled.append(Segment(replacement, segment.start_s, segment.end_s))
        else:
            relabeled.append(segment)
    return join_adjacent_equal_labels(relabeled)


def join_adjacent_equal_labels(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []

    joined = [segments[0]]
    for segment in segments[1:]:
        previous = joined[-1]
        if segment.label == previous.label:
            joined[-1] = Segment(previous.label, previous.start_s, segment.end_s)
        else:
            joined.append(segment)
    return joined


def index_of_shortest(segments: list[Segment]) -> int:
    shortest_index = 0
    for index, segment in enumerate(segments):
        if segment.duration_s() < segments[shortest_index].duration_s():
            shortest_index = index
    return shortest_index


def absorb_into_neighbor(segments: list[Segment], short_index: int) -> list[Segment]:
    """El segmento corto pasa a ser parte del vecino más largo."""
    neighbor_index = longest_neighbor_index(segments, short_index)
    short_segment = segments[short_index]
    neighbor = segments[neighbor_index]

    start_s = min(short_segment.start_s, neighbor.start_s)
    end_s = max(short_segment.end_s, neighbor.end_s)
    extended_neighbor = Segment(neighbor.label, start_s, end_s)

    remaining = []
    for index, segment in enumerate(segments):
        if index == short_index:
            continue
        if index == neighbor_index:
            remaining.append(extended_neighbor)
        else:
            remaining.append(segment)
    return join_adjacent_equal_labels(remaining)


def longest_neighbor_index(segments: list[Segment], index: int) -> int:
    has_left = index > 0
    has_right = index < len(segments) - 1
    if not has_left:
        return index + 1
    if not has_right:
        return index - 1

    left_duration = segments[index - 1].duration_s()
    right_duration = segments[index + 1].duration_s()
    if left_duration >= right_duration:
        return index - 1
    return index + 1
