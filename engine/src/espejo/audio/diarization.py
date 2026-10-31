"""Diarización con pyannote: separa hablantes para saber cuándo habla el docente.

Si está desactivada, falta pyannote.audio, falta el token o no se aceptaron las condiciones
de los modelos en Hugging Face, el pipeline sigue y todos los segmentos quedan "unknown".
"""

import logging
from dataclasses import replace
from pathlib import Path

from espejo.audio.schemas import (
    UNKNOWN_SPEAKER,
    DiarizationOutcome,
    SpeakerTurn,
    TranscriptSegment,
)
from espejo.common.env import read_secret
from espejo.config import DiarizationConfig

logger = logging.getLogger(__name__)

GATED_MODEL_HINT = (
    "acepta las condiciones de pyannote/speaker-diarization-3.1 y pyannote/segmentation-3.0 "
    "en Hugging Face"
)


class SpeakerDiarizer:
    def __init__(self, config: DiarizationConfig, sample_rate_hz: int) -> None:
        self.config = config
        self.sample_rate_hz = sample_rate_hz

    def diarize(self, audio_path: Path) -> DiarizationOutcome:
        """Nunca lanza excepciones: si algo falla, devuelve el motivo y el pipeline sigue."""
        if not self.config.enabled:
            return self.skipped("diarización desactivada en configs/audio.yaml")

        token = read_secret(self.config.token_env_var)
        if token is None:
            return self.skipped(f"falta {self.config.token_env_var} en .env")

        try:
            turns = self.run_pyannote(audio_path, token)
        except ImportError:
            return self.skipped("pyannote.audio no está instalado")
        except Exception as error:
            # Un fallo de pyannote (red, permisos, modelo) no debe tumbar el análisis
            return self.skipped(describe_failure(error))

        logger.info("Diarización: %d turnos de habla", len(turns))
        return DiarizationOutcome(turns, "diarización con " + self.config.model)

    def skipped(self, reason: str) -> DiarizationOutcome:
        logger.info("Sin diarización: %s", reason)
        return DiarizationOutcome(None, reason)

    def run_pyannote(self, audio_path: Path, token: str) -> list[SpeakerTurn]:
        # Imports locales: pyannote y torch son pesados y la diarización es opcional
        import torch
        from faster_whisper.audio import decode_audio
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(self.config.model, use_auth_token=token)
        if pipeline is None:
            raise PermissionError(GATED_MODEL_HINT)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Se pasa la onda en memoria para no depender del backend de audio de torchaudio
        samples = decode_audio(str(audio_path), sampling_rate=self.sample_rate_hz)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": self.sample_rate_hz}
        annotation = pipeline(audio_input, min_speakers=self.config.min_speakers)

        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(SpeakerTurn(speaker, turn.start, turn.end))
        turns.sort(key=turn_start)
        return turns


def turn_start(turn: SpeakerTurn) -> float:
    return turn.start_s


def describe_failure(error: Exception) -> str:
    """Mensaje claro sin incluir el token ni detalles internos de la librería."""
    error_name = type(error).__name__
    if "Gated" in error_name or isinstance(error, PermissionError):
        return f"sin acceso a los modelos de pyannote: {GATED_MODEL_HINT}"
    return f"pyannote falló ({error_name})"


def overlap_s(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def speaker_for_segment(segment: TranscriptSegment, turns: list[SpeakerTurn]) -> str:
    """El hablante que más se solapa en el tiempo con el segmento de Whisper."""
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        shared_s = overlap_s(segment.start_s, segment.end_s, turn.start_s, turn.end_s)
        if shared_s > 0:
            previous_s = overlap_by_speaker.get(turn.speaker, 0.0)
            overlap_by_speaker[turn.speaker] = previous_s + shared_s

    best_speaker = UNKNOWN_SPEAKER
    best_overlap_s = 0.0
    for speaker, shared_s in overlap_by_speaker.items():
        if shared_s > best_overlap_s:
            best_speaker = speaker
            best_overlap_s = shared_s
    return best_speaker


def assign_speakers(
    segments: list[TranscriptSegment], outcome: DiarizationOutcome
) -> list[TranscriptSegment]:
    if outcome.turns is None:
        return segments

    labeled = []
    for segment in segments:
        speaker = speaker_for_segment(segment, outcome.turns)
        labeled.append(replace(segment, speaker=speaker))
    return labeled
