"""Datos que viajan entre las etapas del pipeline de audio."""

from dataclasses import dataclass, field
from enum import StrEnum

UNKNOWN_SPEAKER = "unknown"


class SpeakerRole(StrEnum):
    TEACHER = "teacher"
    STUDENT = "student"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float
    probability: float


@dataclass(frozen=True)
class TranscriptSegment:
    """Segmento de Whisper. speaker, role y low_confidence se completan en etapas posteriores."""

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
class Question:
    text: str
    start_s: float
    question_end_s: float
    speaker: str
    role: SpeakerRole
    reliable: bool
    wait_time_s: float | None = None
    wait_time_ok: bool | None = None


@dataclass(frozen=True)
class AudioMetrics:
    analyzed_duration_s: float
    speech_ratio: float
    teacher_talk_ratio: float | None
    teacher_talk_ratio_reason: str | None
    question_count: int
    questions_per_minute: float
    mean_wait_time_s: float | None
    median_wait_time_s: float | None
    wait_time_ok_ratio: float | None
    reliable_question_count: int
    low_confidence_ratio: float


@dataclass(frozen=True)
class TranscriberSetup:
    """Modelo y dispositivo que se usaron realmente para transcribir."""

    model_name: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class DiarizationOutcome:
    """Resultado de intentar diarizar: los turnos o el motivo por el que no se hizo."""

    turns: list[SpeakerTurn] | None
    note: str

    def succeeded(self) -> bool:
        return self.turns is not None


@dataclass(frozen=True)
class AudioReport:
    video_path: str
    duration_s: float
    transcriber: TranscriberSetup
    language: str
    diarization_used: bool
    diarization_note: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    metrics: AudioMetrics | None = None
