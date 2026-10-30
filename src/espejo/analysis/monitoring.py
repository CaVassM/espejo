"""Monitoreo (R3): unión, sin doble conteo, de lo que ve el video y lo que se oye en el audio."""

from dataclasses import dataclass

from espejo.audio.schemas import Question
from espejo.common.ratios import safe_ratio
from espejo.vision.schemas import Behavior, Segment


@dataclass(frozen=True)
class TimeInterval:
    start_s: float
    end_s: float

    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class MonitoringResult:
    """video_ratio o audio_ratio son None cuando esa fuente no estuvo disponible."""

    ratio: float
    video_ratio: float | None
    audio_ratio: float | None
    meets_threshold: bool


def interval_start(interval: TimeInterval) -> float:
    return interval.start_s


def merge_intervals(intervals: list[TimeInterval]) -> list[TimeInterval]:
    """Une los intervalos que se solapan o se tocan, para no contar dos veces el mismo tiempo."""
    merged: list[TimeInterval] = []
    for interval in sorted(intervals, key=interval_start):
        if merged and interval.start_s <= merged[-1].end_s:
            previous = merged[-1]
            merged[-1] = TimeInterval(previous.start_s, max(previous.end_s, interval.end_s))
        else:
            merged.append(interval)
    return merged


def union_duration_s(intervals: list[TimeInterval]) -> float:
    total_s = 0.0
    for interval in merge_intervals(intervals):
        total_s += interval.duration_s()
    return total_s


def guide_intervals(video_timeline: list[Segment]) -> list[TimeInterval]:
    intervals = []
    for segment in video_timeline:
        if segment.label == Behavior.GUIDE:
            intervals.append(TimeInterval(segment.start_s, segment.end_s))
    return intervals


def counts_for_monitoring(question: Question) -> bool:
    """Solo preguntas reales (no muletillas), en audio confiable y que alguien respondió."""
    if question.is_rhetorical() or not question.reliable:
        return False
    return question.response is not None


def question_episodes(
    questions: list[Question], max_episode_s: float, duration_s: float
) -> list[TimeInterval]:
    """Cada episodio va desde la pregunta hasta el fin de la respuesta, con un máximo."""
    episodes = []
    for question in questions:
        if not counts_for_monitoring(question):
            continue
        end_s = min(question.response.end_s, question.start_s + max_episode_s, duration_s)
        episodes.append(TimeInterval(question.start_s, end_s))
    return episodes


def compute_monitoring(
    video_intervals: list[TimeInterval] | None,
    audio_intervals: list[TimeInterval] | None,
    duration_s: float,
    threshold: float,
) -> MonitoringResult:
    """Pasa None en una fuente si no estuvo disponible (p. ej. --no-audio)."""
    all_intervals: list[TimeInterval] = []
    video_ratio = None
    audio_ratio = None

    if video_intervals is not None:
        all_intervals += video_intervals
        video_ratio = safe_ratio(union_duration_s(video_intervals), duration_s)
    if audio_intervals is not None:
        all_intervals += audio_intervals
        audio_ratio = safe_ratio(union_duration_s(audio_intervals), duration_s)

    ratio = safe_ratio(union_duration_s(all_intervals), duration_s)
    return MonitoringResult(ratio, video_ratio, audio_ratio, ratio >= threshold)
