"""Marca los tramos donde el audio no alcanza para confiar en la transcripción."""

from dataclasses import replace

from espejo.audio.schemas import TranscriptSegment
from espejo.config import QualityConfig


class AudioQualityChecker:
    """Usa la confianza que reporta Whisper; así no inventamos métricas sobre audio malo."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def is_low_confidence(self, segment: TranscriptSegment) -> bool:
        uncertain_text = segment.avg_logprob < self.config.min_avg_logprob
        probably_no_speech = segment.no_speech_prob > self.config.max_no_speech_prob
        return uncertain_text or probably_no_speech

    def mark(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        marked = []
        for segment in segments:
            low_confidence = self.is_low_confidence(segment)
            marked.append(replace(segment, low_confidence=low_confidence))
        return marked


def low_confidence_duration_s(segments: list[TranscriptSegment]) -> float:
    total_s = 0.0
    for segment in segments:
        if segment.low_confidence:
            total_s += segment.duration_s()
    return total_s
