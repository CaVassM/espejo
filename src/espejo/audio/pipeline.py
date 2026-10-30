"""AudioPipeline: orquesta las etapas del módulo de audio sobre un video."""

import logging
from dataclasses import dataclass
from pathlib import Path

from espejo.audio.diarization import SpeakerDiarizer, assign_speakers
from espejo.audio.extraction import AudioExtractor, ExtractedAudio
from espejo.audio.metrics import AudioMetricsCalculator
from espejo.audio.quality import AudioQualityChecker
from espejo.audio.questions import QuestionDetector
from espejo.audio.report_writer import save_report_json, save_transcript
from espejo.audio.schemas import AudioReport
from espejo.audio.speaker_roles import assign_roles
from espejo.audio.transcriber import Transcriber
from espejo.audio.voice_activity import VoiceActivityDetector
from espejo.audio.wait_time import WaitTimeCalculator
from espejo.config import AudioConfig

logger = logging.getLogger(__name__)

REPORT_FILENAME = "audio_report.json"
TRANSCRIPT_FILENAME = "transcript.txt"


@dataclass(frozen=True)
class AudioPipelineResult:
    report: AudioReport
    report_path: Path
    transcript_path: Path


class AudioPipeline:
    def __init__(self, config: AudioConfig, model_override: str | None = None) -> None:
        self.config = config
        self.extractor = AudioExtractor(config.extraction)
        self.transcriber = Transcriber(config.transcription, model_override)
        self.voice_detector = VoiceActivityDetector(
            config.voice_activity, config.extraction.sample_rate_hz
        )
        self.diarizer = SpeakerDiarizer(config.diarization)
        self.quality_checker = AudioQualityChecker(config.quality)
        self.question_detector = QuestionDetector(config.questions)
        self.wait_time_calculator = WaitTimeCalculator(config.wait_time)
        self.metrics_calculator = AudioMetricsCalculator()

    def run(self, video_path: Path, output_root: Path, keep_audio: bool) -> AudioPipelineResult:
        output_dir = output_root / video_path.stem
        audio = self.extractor.extract(video_path, output_dir)

        # Los wav son temporales: se borran aunque el análisis falle
        try:
            report = self.analyze(video_path, audio)
        finally:
            if not keep_audio:
                audio.remove()

        report_path = output_dir / REPORT_FILENAME
        transcript_path = output_dir / TRANSCRIPT_FILENAME
        save_report_json(report, report_path)
        save_transcript(report, transcript_path)
        return AudioPipelineResult(report, report_path, transcript_path)

    def analyze(self, video_path: Path, audio: ExtractedAudio) -> AudioReport:
        segments = self.transcriber.transcribe(audio.raw_path)
        segments = self.quality_checker.mark(segments)
        speech_intervals = self.voice_detector.detect(audio.vad_path)

        diarization = self.diarizer.diarize(audio.raw_path)
        segments = assign_roles(assign_speakers(segments, diarization))

        questions = self.question_detector.detect(segments, teacher_only=diarization.succeeded())
        questions = self.wait_time_calculator.annotate(questions, speech_intervals)
        logger.info("Preguntas detectadas: %d", len(questions))

        metrics = self.metrics_calculator.calculate(
            segments,
            questions,
            speech_intervals,
            audio.duration_s,
            diarization_used=diarization.succeeded(),
        )
        return AudioReport(
            video_path=str(video_path),
            duration_s=audio.duration_s,
            transcriber=self.transcriber.setup,
            language=self.config.transcription.language,
            diarization_used=diarization.succeeded(),
            diarization_note=diarization.note,
            segments=segments,
            questions=questions,
            metrics=metrics,
        )
