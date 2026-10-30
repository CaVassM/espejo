"""Detección de preguntas en la transcripción."""

import re
from dataclasses import dataclass

from espejo.audio.schemas import Question, SpeakerRole, TranscriptSegment, Word
from espejo.config import QuestionsConfig

QUESTION_MARKS = ("?", "¿")
SENTENCE_ENDINGS = (".", "?", "!", "…")
# Signos que se quitan de los bordes de una palabra antes de compararla con un interrogativo
EDGE_PUNCTUATION = "¿?¡!.,;:\"'«»()…-—"
# Corta el texto después de un signo de cierre seguido de espacio
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.?!…])\s+")
UNKNOWN_WORD_PROBABILITY = 0.0


@dataclass(frozen=True)
class SpokenWord:
    """Palabra junto al segmento de Whisper del que viene."""

    word: Word
    segment: TranscriptSegment


@dataclass(frozen=True)
class Sentence:
    text: str
    start_s: float
    end_s: float
    speaker: str
    role: SpeakerRole
    reliable: bool


def is_question(text: str, interrogatives: list[str]) -> bool:
    """Pregunta si tiene signos de interrogación o si empieza con un interrogativo.

    Un "qué" a la mitad de la oración ("no sé qué hacer") no la convierte en pregunta.
    """
    for mark in QUESTION_MARKS:
        if mark in text:
            return True
    return starts_with_interrogative(text, interrogatives)


def starts_with_interrogative(text: str, interrogatives: list[str]) -> bool:
    for interrogative in interrogatives:
        expected_words = interrogative.split()
        if leading_words(text, len(expected_words)) == expected_words:
            return True
    return False


def leading_words(text: str, count: int) -> list[str]:
    """Primeras palabras en minúscula y sin signos, para comparar con los interrogativos."""
    cleaned_words = []
    for raw_word in text.lower().split()[:count]:
        cleaned_words.append(raw_word.strip(EDGE_PUNCTUATION))
    return cleaned_words


def ends_sentence(word_text: str) -> bool:
    return word_text.strip().endswith(SENTENCE_ENDINGS)


def words_for_segment(segment: TranscriptSegment) -> list[Word]:
    """Sin timestamps por palabra, cada oración del texto pasa a ser una sola "palabra"."""
    if segment.words:
        return segment.words

    pseudo_words = []
    for piece in SENTENCE_SPLIT_PATTERN.split(segment.text):
        text = piece.strip()
        if text:
            pseudo_words.append(
                Word(" " + text, segment.start_s, segment.end_s, UNKNOWN_WORD_PROBABILITY)
            )
    return pseudo_words


def sentence_from(spoken_words: list[SpokenWord]) -> Sentence:
    text_parts = []
    reliable = True
    for spoken_word in spoken_words:
        text_parts.append(spoken_word.word.text)
        if spoken_word.segment.low_confidence:
            reliable = False

    first = spoken_words[0]
    return Sentence(
        text="".join(text_parts).strip(),
        start_s=first.word.start_s,
        end_s=spoken_words[-1].word.end_s,
        speaker=first.segment.speaker,
        role=first.segment.role,
        reliable=reliable,
    )


class QuestionDetector:
    def __init__(self, config: QuestionsConfig) -> None:
        self.config = config

    def detect(self, segments: list[TranscriptSegment], teacher_only: bool) -> list[Question]:
        """Con diarización solo cuentan las preguntas del docente (teacher_only).

        Sin diarización se cuentan todas y quedan con speaker_known = False.
        """
        questions = []
        for sentence in self.build_sentences(segments):
            if teacher_only and sentence.role != SpeakerRole.TEACHER:
                continue
            if not is_question(sentence.text, self.config.interrogatives):
                continue
            questions.append(to_question(sentence, speaker_known=teacher_only))
        return questions

    def build_sentences(self, segments: list[TranscriptSegment]) -> list[Sentence]:
        """Arma oraciones con las palabras; una oración puede cruzar dos segmentos de Whisper."""
        sentences = []
        pending: list[SpokenWord] = []

        for segment in segments:
            if pending and self.breaks_sentence(pending[-1], segment):
                sentences.append(sentence_from(pending))
                pending = []

            for word in words_for_segment(segment):
                pending.append(SpokenWord(word, segment))
                if ends_sentence(word.text):
                    sentences.append(sentence_from(pending))
                    pending = []

        if pending:
            sentences.append(sentence_from(pending))
        return sentences

    def breaks_sentence(self, last_word: SpokenWord, next_segment: TranscriptSegment) -> bool:
        """Una oración sin cerrar no continúa si cambia el hablante o hay una pausa larga."""
        speaker_changed = last_word.segment.speaker != next_segment.speaker
        gap_s = next_segment.start_s - last_word.word.end_s
        return speaker_changed or gap_s > self.config.join_segments_max_gap_s


def to_question(sentence: Sentence, speaker_known: bool) -> Question:
    return Question(
        text=sentence.text,
        start_s=sentence.start_s,
        question_end_s=sentence.end_s,
        speaker=sentence.speaker,
        role=sentence.role,
        reliable=sentence.reliable,
        speaker_known=speaker_known,
    )
