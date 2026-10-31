"""Descarta texto que Whisper inventa en silencios o ruido (p. ej. "Gracias." al final)."""

from dataclasses import dataclass

from espejo.audio.schemas import SpeechInterval, TranscriptSegment
from espejo.common.ratios import safe_ratio
from espejo.common.text import normalize_text
from espejo.config import HallucinationConfig


@dataclass(frozen=True)
class FilteredSegments:
    kept: list[TranscriptSegment]
    discarded: list[TranscriptSegment]


def speech_overlap_s(segment: TranscriptSegment, intervals: list[SpeechInterval]) -> float:
    total_s = 0.0
    for interval in intervals:
        shared_s = min(segment.end_s, interval.end_s) - max(segment.start_s, interval.start_s)
        if shared_s > 0:
            total_s += shared_s
    return total_s


def is_typical_hallucination(segment: TranscriptSegment, phrases: list[str]) -> bool:
    """Se compara el segmento completo, sin tildes ni signos: "¡Gracias!" == "gracias"."""
    normalized = normalize_text(segment.text)
    for phrase in phrases:
        if normalized == normalize_text(phrase):
            return True
    return False


class HallucinationFilter:
    def __init__(self, config: HallucinationConfig) -> None:
        self.config = config

    def is_hallucination(self, segment: TranscriptSegment, intervals: list[SpeechInterval]) -> bool:
        """Frase típica inventada, o segmento corto casi sin voz detectada.

        La regla de la voz solo se aplica a segmentos cortos: Silero no oye bien a quien habla
        lejos o bajo, y en segmentos largos borraba habla real (p. ej. respuestas de alumnos).
        """
        if is_typical_hallucination(segment, self.config.phrases):
            return True
        if len(segment.text.split()) > self.config.max_words_for_vad_check:
            return False
        overlap_ratio = safe_ratio(speech_overlap_s(segment, intervals), segment.duration_s())
        return overlap_ratio < self.config.min_vad_overlap

    def filter(
        self, segments: list[TranscriptSegment], intervals: list[SpeechInterval]
    ) -> FilteredSegments:
        kept = []
        discarded = []
        for segment in segments:
            if self.is_hallucination(segment, intervals):
                discarded.append(segment)
            else:
                kept.append(segment)
        return FilteredSegments(kept, discarded)
