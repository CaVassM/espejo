"""Detección de voz con Silero VAD (la versión ONNX que trae faster-whisper, corre en CPU)."""

import logging
from pathlib import Path

# Debe ir antes de faster-whisper: carga primero la cuDNN de torch (ver audio/cuda_setup.py)
import espejo.audio.cuda_setup  # noqa: F401

# isort: split
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from espejo.audio.schemas import SpeechInterval
from espejo.config import VoiceActivityConfig

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Entrega los tramos con voz. No distingue quién habla: para el wait time no hace falta."""

    def __init__(self, config: VoiceActivityConfig, sample_rate_hz: int) -> None:
        self.config = config
        self.sample_rate_hz = sample_rate_hz

    def detect(self, audio_path: Path) -> list[SpeechInterval]:
        audio = decode_audio(str(audio_path), sampling_rate=self.sample_rate_hz)
        options = VadOptions(
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )
        timestamps = get_speech_timestamps(
            audio, vad_options=options, sampling_rate=self.sample_rate_hz
        )

        intervals = []
        for timestamp in timestamps:
            start_s = timestamp["start"] / self.sample_rate_hz
            end_s = timestamp["end"] / self.sample_rate_hz
            intervals.append(SpeechInterval(start_s, end_s))

        logger.info("VAD: %d intervalos de voz", len(intervals))
        return intervals
