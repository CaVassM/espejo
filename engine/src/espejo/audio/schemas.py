"""Datos que viajan entre las etapas del módulo de audio."""

from dataclasses import dataclass
from enum import StrEnum

UNKNOWN_SPEAKER = "unknown"
# Privacidad (audio/privacy.py): son parte del formato de audio_raw.json, no se configuran
STUDENT_PLACEHOLDER = "[ALUMNO responde]"
NAME_PLACEHOLDER = "[NOMBRE]"


class SpeakerRole(StrEnum):
    TEACHER = "teacher"
    STUDENT = "student"
    UNKNOWN = "unknown"


class QuestionCategory(StrEnum):
    """Clasificación del LLM para una pregunta candidata."""

    OPEN = "question_open"
    CLOSED = "question_closed"
    RHETORICAL = "question_rhetorical"


class ResponseCertainty(StrEnum):
    """Con diarización sabemos que respondió otra persona; sin ella, solo que hubo voz."""

    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float
    probability: float


@dataclass(frozen=True)
class TranscriptSegment:
    """Segmento de Whisper. speaker, role y low_confidence se completan al interpretar."""

    start_s: float
    end_s: float
    text: str
    words: list[Word]
    avg_logprob: float
    no_speech_prob: float
    speaker: str = UNKNOWN_SPEAKER
    role: SpeakerRole = SpeakerRole.UNKNOWN
    low_confidence: bool = False

    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SpeechInterval:
    """Tramo con voz según el VAD, sin importar quién habla."""

    start_s: float
    end_s: float

    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SpeakerTurn:
    """Turno de un hablante según la diarización."""

    speaker: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class DiarizationOutcome:
    """Resultado de intentar diarizar: los turnos o el motivo por el que no se hizo."""

    turns: list[SpeakerTurn] | None
    note: str

    def succeeded(self) -> bool:
        return self.turns is not None


@dataclass(frozen=True)
class TranscriberSetup:
    """Modelo y dispositivo que se usaron realmente para transcribir."""

    model_name: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class AudioCapture:
    """Todo lo que VER produce sobre el audio; INTERPRETAR trabaja solo con esto."""

    video_path: str
    duration_s: float
    language: str
    transcriber: TranscriberSetup
    segments: list[TranscriptSegment]
    speech_intervals: list[SpeechInterval]
    diarization: DiarizationOutcome
    topic: str | None = None


@dataclass(frozen=True)
class StudentResponse:
    start_s: float
    end_s: float
    speaker: str
    certainty: ResponseCertainty


@dataclass(frozen=True)
class Question:
    text: str
    start_s: float
    question_end_s: float
    speaker: str
    role: SpeakerRole
    reliable: bool
    speaker_known: bool = False
    is_tag: bool = False
    llm_category: QuestionCategory | None = None
    wait_time_s: float | None = None
    wait_time_ok: bool | None = None
    response: StudentResponse | None = None

    def is_rhetorical(self) -> bool:
        """Si el LLM la clasificó, manda el LLM; si no, la regla de muletillas."""
        if self.llm_category is not None:
            return self.llm_category == QuestionCategory.RHETORICAL
        return self.is_tag


@dataclass(frozen=True)
class AudioMetrics:
    analyzed_duration_s: float
    speech_ratio: float
    teacher_talk_ratio: float | None
    teacher_talk_ratio_reason: str | None
    teacher_question_count: int
    rhetorical_question_count: int
    reliable_question_count: int
    questions_per_minute: float
    open_question_ratio: float | None
    mean_wait_time_s: float | None
    median_wait_time_s: float | None
    wait_time_ok_ratio: float | None
    student_response_count: int
    response_rate: float | None
    low_confidence_ratio: float


@dataclass(frozen=True)
class AudioInterpretation:
    segments: list[TranscriptSegment]
    questions: list[Question]
    metrics: AudioMetrics
    diarization_used: bool
    diarization_note: str
    teacher_speaker: str | None = None
    discarded_segments: int = 0
