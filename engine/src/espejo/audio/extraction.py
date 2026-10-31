"""Extracción del audio del video con ffmpeg: un wav crudo y una copia filtrada para el VAD."""

import logging
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from espejo.config import ExtractionConfig

logger = logging.getLogger(__name__)

RAW_AUDIO_FILENAME = "audio_raw.wav"
VAD_AUDIO_FILENAME = "audio_vad.wav"
MONO_CHANNELS = "1"
PCM_CODEC = "pcm_s16le"
STDERR_TAIL_CHARS = 500
# Mensajes de ffmpeg cuando el video no trae pista de audio
NO_AUDIO_MESSAGES = ("does not contain any stream", "matches no streams")


class NoAudioTrackError(ValueError):
    """El video no trae pista de audio: el análisis puede seguir solo con el video."""


@dataclass(frozen=True)
class ExtractedAudio:
    """Los dos wav temporales. Whisper usa el crudo; el VAD, el filtrado."""

    raw_path: Path
    vad_path: Path
    duration_s: float

    def remove(self) -> None:
        self.raw_path.unlink(missing_ok=True)
        self.vad_path.unlink(missing_ok=True)


class AudioExtractor:
    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        self.ffmpeg_path = find_ffmpeg()

    def extract(self, video_path: Path, work_dir: Path) -> ExtractedAudio:
        if not video_path.exists():
            raise FileNotFoundError(f"No se encontró el video {video_path}.")
        work_dir.mkdir(parents=True, exist_ok=True)

        raw_path = work_dir / RAW_AUDIO_FILENAME
        vad_path = work_dir / VAD_AUDIO_FILENAME
        logger.info("Extrayendo audio de %s", video_path.name)

        # Sin filtros: el denoising empeora la transcripción de Whisper
        self.run_ffmpeg(video_path, raw_path, audio_filter=None)
        highpass_filter = f"highpass=f={self.config.vad_highpass_hz}"
        self.run_ffmpeg(video_path, vad_path, audio_filter=highpass_filter)

        return ExtractedAudio(raw_path, vad_path, read_wav_duration_s(raw_path))

    def run_ffmpeg(self, video_path: Path, output_path: Path, audio_filter: str | None) -> None:
        command = [str(self.ffmpeg_path), "-y", "-loglevel", "error", "-i", str(video_path)]
        command += ["-vn", "-ac", MONO_CHANNELS, "-ar", str(self.config.sample_rate_hz)]
        if audio_filter is not None:
            command += ["-af", audio_filter]
        command += ["-c:a", PCM_CODEC, str(output_path)]

        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if completed.returncode != 0:
            raise_ffmpeg_error(video_path, completed.stderr)


def raise_ffmpeg_error(video_path: Path, stderr: str) -> None:
    for message in NO_AUDIO_MESSAGES:
        if message in stderr:
            raise NoAudioTrackError(f"El video {video_path.name} no tiene pista de audio.")
    stderr_tail = stderr[-STDERR_TAIL_CHARS:]
    raise RuntimeError(f"ffmpeg no pudo extraer el audio de {video_path.name}: {stderr_tail}")


def find_ffmpeg() -> Path:
    """Busca ffmpeg en el PATH y, si no está, dentro del entorno conda activo."""
    on_path = shutil.which("ffmpeg")
    if on_path is not None:
        return Path(on_path)

    environment_dir = Path(sys.executable).parent
    candidates = [
        environment_dir / "Library" / "bin" / "ffmpeg.exe",
        environment_dir / "ffmpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No se encontró ffmpeg. Instálalo con: conda install -n ml_env -c conda-forge ffmpeg"
    )


def read_wav_duration_s(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()
