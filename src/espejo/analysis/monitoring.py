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


def episode_end_s(question: Question, max_wait_s: float) -> float:
    """El episodio termina con la respuesta; si nadie respondió, al terminar la espera."""
    if question.response is not None:
        return question.response.end_s
    if question.wait_time_s is not None:
        return question.question_end_s + question.wait_time_s
    return question.question_end_s + max_wait_s


def question_episodes(
    questions: list[Question], max_wait_s: float, duration_s: float
) -> list[TimeInterval]:
    """Episodios de pregunta a la clase. Solo preguntas en audio confiable."""
    episodes = []
    for question in questions:
        if not question.reliable:
            continue
        end_s = min(episode_end_s(question, max_wait_s), duration_s)
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
