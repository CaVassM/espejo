"""VER (audio): una sola pasada por Whisper, Silero y pyannote; guarda audio_raw.json."""

import logging
from pathlib import Path

from espejo.audio.capture_io import AUDIO_RAW_FILENAME, save_audio_capture
from espejo.audio.diarization import SpeakerDiarizer
from espejo.audio.extraction import AudioExtractor
from espejo.audio.privacy import Anonymizer
from espejo.audio.schemas import AudioCapture
from espejo.audio.transcriber import Transcriber
from espejo.audio.voice_activity import VoiceActivityDetector
from espejo.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCaptureRunner:
    def __init__(self, config: AudioConfig, model_override: str | None = None) -> None:
        sample_rate_hz = config.extraction.sample_rate_hz
        self.config = config
        self.extractor = AudioExtractor(config.extraction)
        self.transcriber = Transcriber(config.transcription, model_override)
        self.voice_detector = VoiceActivityDetector(config.voice_activity, sample_rate_hz)
        self.diarizer = SpeakerDiarizer(config.diarization, sample_rate_hz)
        self.anonymizer = None
        if config.privacy.enabled:
            self.anonymizer = Anonymizer(config.privacy, config.hallucinations)

    def run(
        self, video_path: Path, output_dir: Path, keep_audio: bool, topic: str | None = None
    ) -> AudioCapture:
        audio = self.extractor.extract(video_path, output_dir)

        # Los wav son temporales: se borran aunque algún modelo falle
        try:
            capture = AudioCapture(
                video_path=str(video_path),
                duration_s=audio.duration_s,
                language=self.config.transcription.language,
                transcriber=self.transcriber.setup,
                segments=self.transcriber.transcribe(audio.raw_path, topic),
                speech_intervals=self.voice_detector.detect(audio.vad_path),
                diarization=self.diarizer.diarize(audio.raw_path),
                topic=topic,
            )
        finally:
            if not keep_audio:
                audio.remove()

        # El texto original de los alumnos nunca llega a disco
        if self.anonymizer is not None:
            capture = self.anonymizer.anonymize(capture)
        save_audio_capture(capture, output_dir / AUDIO_RAW_FILENAME)
        return capture
