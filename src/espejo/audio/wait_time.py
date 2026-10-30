"""Tiempo de espera ("wait time") después de cada pregunta."""

from dataclasses import replace

from espejo.audio.schemas import Question, SpeechInterval
from espejo.config import WaitTimeConfig


class WaitTimeCalculator:
    def __init__(self, config: WaitTimeConfig) -> None:
        self.config = config

    def wait_time_s(self, question_end_s: float, intervals: list[SpeechInterval]) -> float | None:
        """Segundos desde el fin de la pregunta hasta que vuelve a haber voz (de quien sea).

        - Se toma el siguiente intervalo del VAD que empieza después de la pregunta.
        - Los huecos menores a min_gap_s se ignoran: son respiración o la cola de la pregunta.
        - Si la voz que contiene el fin de la pregunta sigue más de min_gap_s, el docente
          siguió hablando sin detenerse: no hubo espera y el resultado es 0.
        - Se limita a max_wait_s. Si no vuelve a haber voz, devuelve None.
        Los intervalos deben venir ordenados por tiempo.
        """
        for interval in intervals:
            if interval.end_s <= question_end_s:
                continue

            if interval.start_s <= question_end_s:
                if interval.end_s - question_end_s > self.config.min_gap_s:
                    return 0.0
                continue

            gap_s = interval.start_s - question_end_s
            if gap_s < self.config.min_gap_s:
                continue
            return min(gap_s, self.config.max_wait_s)

        return None

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
