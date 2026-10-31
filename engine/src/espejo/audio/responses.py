"""Respuesta de un alumno después de cada pregunta del docente."""

from dataclasses import replace

from espejo.audio.schemas import (
    UNKNOWN_SPEAKER,
    Question,
    ResponseCertainty,
    SpeakerTurn,
    SpeechInterval,
    StudentResponse,
)
from espejo.audio.wait_time import find_next_speech
from espejo.config import ResponsesConfig


class StudentResponseFinder:
    """Busca la primera voz de alguien distinto al docente poco después de la pregunta."""

    def __init__(self, config: ResponsesConfig, min_gap_s: float) -> None:
        self.config = config
        self.min_gap_s = min_gap_s

    def response_with_diarization(
        self, question_end_s: float, turns: list[SpeakerTurn], teacher_speaker: str
    ) -> StudentResponse | None:
        """Primer turno de otro hablante que empieza dentro de la ventana de respuesta."""
        window_end_s = question_end_s + self.config.response_window_s
        for turn in turns:
            if turn.speaker == teacher_speaker or turn.end_s <= question_end_s:
                continue
            if turn.start_s > window_end_s:
                return None
            return StudentResponse(turn.start_s, turn.end_s, turn.speaker, ResponseCertainty.HIGH)
        return None

    def response_without_diarization(
        self, question_end_s: float, intervals: list[SpeechInterval]
    ) -> StudentResponse | None:
        """Sin saber quién habla, cualquier voz nueva en la ventana es una posible respuesta."""
        next_speech = find_next_speech(intervals, question_end_s, self.min_gap_s)
        if next_speech is None:
            return None
        if next_speech.start_s - question_end_s > self.config.response_window_s:
            return None
        return StudentResponse(
            next_speech.start_s, next_speech.end_s, UNKNOWN_SPEAKER, ResponseCertainty.LOW
        )

    def annotate(
        self,
        questions: list[Question],
        intervals: list[SpeechInterval],
        turns: list[SpeakerTurn] | None,
        teacher_speaker: str | None,
    ) -> list[Question]:
        annotated = []
        for question in questions:
            if turns is not None and teacher_speaker is not None:
                response = self.response_with_diarization(
                    question.question_end_s, turns, teacher_speaker
                )
            else:
                response = self.response_without_diarization(question.question_end_s, intervals)
            annotated.append(replace(question, response=response))
        return annotated
