"""Tiempo de espera ("wait time") después de cada pregunta."""

from dataclasses import replace

from espejo.audio.schemas import Question, SpeechInterval
from espejo.config import WaitTimeConfig


def find_next_speech(
    intervals: list[SpeechInterval], after_s: float, min_gap_s: float
) -> SpeechInterval | None:
    """Primer intervalo de voz que empieza después de after_s, ignorando huecos < min_gap_s.

    Los intervalos que ya empezaron antes de after_s (la cola de la propia pregunta) y los que
    empiezan tras un hueco corto (respiración, pausa interna) no cuentan como "voz nueva".
    Los intervalos deben venir ordenados por tiempo.
    """
    for interval in intervals:
        gap_s = interval.start_s - after_s
        if gap_s >= min_gap_s:
            return interval
    return None


def speech_continues(intervals: list[SpeechInterval], after_s: float, min_gap_s: float) -> bool:
    """True si la voz que contiene after_s sigue más de min_gap_s: no hubo pausa."""
    for interval in intervals:
        if interval.start_s <= after_s < interval.end_s:
            return interval.end_s - after_s > min_gap_s
    return False


class WaitTimeCalculator:
    def __init__(self, config: WaitTimeConfig) -> None:
        self.config = config

    def wait_time_s(self, question_end_s: float, intervals: list[SpeechInterval]) -> float | None:
        """Segundos desde el fin de la pregunta hasta que vuelve a haber voz (de quien sea).

        Si el docente sigue hablando sin pausa después de la pregunta, no hubo espera: 0.
        Se limita a max_wait_s. Si no vuelve a haber voz, devuelve None.
        """
        if speech_continues(intervals, question_end_s, self.config.min_gap_s):
            return 0.0

        next_speech = find_next_speech(intervals, question_end_s, self.config.min_gap_s)
        if next_speech is None:
            return None

        gap_s = next_speech.start_s - question_end_s
        return min(gap_s, self.config.max_wait_s)

    def annotate(
        self, questions: list[Question], intervals: list[SpeechInterval]
    ) -> list[Question]:
        annotated = []
        for question in questions:
            wait_time_s = self.wait_time_s(question.question_end_s, intervals)
            wait_time_ok = None
            if wait_time_s is not None:
                wait_time_ok = wait_time_s >= self.config.ok_threshold_s
            annotated.append(replace(question, wait_time_s=wait_time_s, wait_time_ok=wait_time_ok))
        return annotated
