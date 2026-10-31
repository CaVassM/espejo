"""VER: corre los modelos una sola vez (YOLO, Whisper, Silero, pyannote) y guarda los JSON.

Necesita espejo[vision] para el video y espejo[audio] para el audio.
"""

import gc
import logging
from pathlib import Path

from espejo.analysis.runner import InterpretationResult
from espejo.audio.capture import AudioCaptureRunner
from espejo.audio.extraction import NoAudioTrackError
from espejo.config import AudioConfig, VisionConfig
from espejo.vision.debug_video import DebugVideoWriter
from espejo.vision.detection_run import VideoDetectionRunner
from espejo.vision.frame_sampler import FrameSampler

logger = logging.getLogger(__name__)

DEBUG_VIDEO_FILENAME = "debug.mp4"


def release_gpu_memory() -> None:
    """Libera la memoria de los modelos de visión antes de cargar Whisper en la misma GPU."""
    # Import local: torch solo hace falta si hay GPU que liberar
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def capture_video(video_path: Path, run_dir: Path, config: VisionConfig) -> None:
    """VER (video): genera detections.json y el fondo desenfocado del heatmap."""
    logger.info("VER video: %s", video_path.name)
    runner = VideoDetectionRunner(config)
    runner.run(video_path, run_dir)
    del runner
    release_gpu_memory()


def capture_audio(
    video_path: Path,
    run_dir: Path,
    config: AudioConfig,
    keep_audio: bool,
    model_override: str | None = None,
    topic: str | None = None,
) -> None:
    """VER (audio): genera audio_raw.json."""
    logger.info("VER audio: %s", video_path.name)
    runner = AudioCaptureRunner(config, model_override)
    runner.run(video_path, run_dir, keep_audio, topic)
    del runner
    release_gpu_memory()


def try_capture_audio(
    video_path: Path, run_dir: Path, config: AudioConfig, topic: str | None = None
) -> bool:
    """Como capture_audio, pero un video sin pista de audio no detiene el análisis."""
    try:
        capture_audio(video_path, run_dir, config, keep_audio=False, topic=topic)
    except NoAudioTrackError as error:
        logger.warning("%s Se analiza solo el video.", error)
        return False
    return True


def write_debug_video(video_path: Path, result: InterpretationResult, config: VisionConfig) -> Path:
    """Opcional: vuelve a leer el video para dibujar al docente y su estado (no va a git)."""
    if result.video is None:
        raise ValueError("El video de depuración necesita detections.json.")

    output_path = result.run_dir / DEBUG_VIDEO_FILENAME
    sampler = FrameSampler(video_path, config.sampling.sample_fps)
    observations = result.video.interpretation.observations
    DebugVideoWriter(config.debug_video).write(sampler, observations, output_path)
    return output_path
