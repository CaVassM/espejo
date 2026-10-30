"""INTERPRETAR (audio): convierte audio_raw.json en preguntas, esperas, respuestas y métricas."""

from espejo.audio.diarization import assign_speakers
from espejo.audio.metrics import AudioMetricsCalculator
from espejo.audio.quality import AudioQualityChecker
from espejo.audio.questions import QuestionDetector
from espejo.audio.responses import StudentResponseFinder
from espejo.audio.schemas import AudioCapture, AudioInterpretation
from espejo.audio.speaker_roles import assign_roles, find_teacher_speaker
from espejo.audio.wait_time import WaitTimeCalculator
from espejo.config import AudioConfig


class AudioInterpreter:
    def __init__(self, config: AudioConfig) -> None:
        self.quality_checker = AudioQualityChecker(config.quality)
        self.question_detector = QuestionDetector(config.questions)
        self.wait_time_calculator = WaitTimeCalculator(config.wait_time)
        self.response_finder = StudentResponseFinder(config.responses, config.wait_time.min_gap_s)
        self.metrics_calculator = AudioMetricsCalculator()

    def interpret(self, capture: AudioCapture) -> AudioInterpretation:
        diarization = capture.diarization
        speaker_known = diarization.succeeded()

        segments = self.quality_checker.mark(capture.segments)
        segments = assign_roles(assign_speakers(segments, diarization))
        teacher_speaker = find_teacher_speaker(segments)

        intervals = capture.speech_intervals
        questions = self.question_detector.detect(segments, teacher_only=speaker_known)
        questions = self.wait_time_calculator.annotate(questions, intervals)
        questions = self.response_finder.annotate(
            questions, intervals, diarization.turns, teacher_speaker
        )

        metrics = self.metrics_calculator.calculate(
            segments, questions, intervals, capture.duration_s, diarization_used=speaker_known
        )
        return AudioInterpretation(
            segments=segments,
            questions=questions,
            metrics=metrics,
            diarization_used=speaker_known,
            diarization_note=diarization.note,
            teacher_speaker=teacher_speaker,
        )
