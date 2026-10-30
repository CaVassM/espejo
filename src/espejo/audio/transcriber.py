"""Transcripción con faster-whisper sobre el audio crudo, con timestamps por palabra."""

import logging
from dataclasses import replace
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as WhisperSegment

from espejo.audio.schemas import TranscriberSetup, TranscriptSegment, Word
from espejo.config import TranscriptionConfig

logger = logging.getLogger(__name__)

DEVICE_AUTO = "auto"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
BYTES_PER_GB = 1024**3
GPU_INDEX = 0
PROGRESS_LOG_INTERVAL_SEGMENTS = 50


class Transcriber:
    def __init__(self, config: TranscriptionConfig, model_override: str | None = None) -> None:
        self.config = config
        self.setup = choose_setup(config, model_override)
        self.model = self.load_model()

    def load_model(self) -> WhisperModel:
        """Si el modelo de GPU no existe en esta versión de faster-whisper, usa el de respaldo."""
        try:
            return self.create_model(self.setup)
        except ValueError:
            if self.setup.device != DEVICE_CUDA:
                raise
            logger.warning(
                "El modelo %s no está disponible; se usa %s",
                self.setup.model_name,
                self.config.gpu_fallback_model,
            )
            self.setup = replace(self.setup, model_name=self.config.gpu_fallback_model)
            return self.create_model(self.setup)

    def create_model(self, setup: TranscriberSetup) -> WhisperModel:
        logger.info(
            "Cargando Whisper %s en %s (%s)", setup.model_name, setup.device, setup.compute_type
        )
        return WhisperModel(setup.model_name, device=setup.device, compute_type=setup.compute_type)

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        # vad_filter=False: Whisper recibe el audio completo y crudo; el VAD es una etapa aparte
        whisper_segments, _ = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            word_timestamps=True,
            beam_size=self.config.beam_size,
            condition_on_previous_text=self.config.condition_on_previous_text,
            vad_filter=False,
        )

        segments = []
        for whisper_segment in whisper_segments:
            segments.append(to_transcript_segment(whisper_segment))
            if len(segments) % PROGRESS_LOG_INTERVAL_SEGMENTS == 0:
                logger.info(
                    "Transcritos %d segmentos (t=%.0f s)", len(segments), whisper_segment.end
                )
        return segments


def to_transcript_segment(whisper_segment: WhisperSegment) -> TranscriptSegment:
    """float() convierte los tipos de numpy que entrega faster-whisper, para poder serializarlos."""
    words = []
    for whisper_word in whisper_segment.words or []:
        word = Word(
            text=whisper_word.word,
            start_s=float(whisper_word.start),
            end_s=float(whisper_word.end),
            probability=float(whisper_word.probability),
        )
        words.append(word)

    return TranscriptSegment(
        start_s=float(whisper_segment.start),
        end_s=float(whisper_segment.end),
        text=whisper_segment.text.strip(),
        words=words,
        avg_logprob=float(whisper_segment.avg_logprob),
        no_speech_prob=float(whisper_segment.no_speech_prob),
    )


def choose_setup(config: TranscriptionConfig, model_override: str | None) -> TranscriberSetup:
    device = resolve_device(config)
    if device == DEVICE_CUDA:
        setup = TranscriberSetup(config.gpu_model, DEVICE_CUDA, config.gpu_compute_type)
    else:
        setup = TranscriberSetup(config.cpu_model, DEVICE_CPU, config.cpu_compute_type)

    if model_override is not None:
        setup = replace(setup, model_name=model_override)
    return setup


def resolve_device(config: TranscriptionConfig) -> str:
    if config.device != DEVICE_AUTO:
        return config.device

    free_memory_gb = free_gpu_memory_gb()
    if free_memory_gb is None:
        logger.info("No hay GPU NVIDIA disponible; se transcribe en CPU")
        return DEVICE_CPU
    if free_memory_gb < config.min_free_gpu_memory_gb:
        logger.info("GPU ocupada (%.1f GB libres); se transcribe en CPU", free_memory_gb)
        return DEVICE_CPU
    return DEVICE_CUDA


def free_gpu_memory_gb() -> float | None:
    """Memoria libre según NVML; no crea un contexto CUDA que compita con otros procesos."""
    # Import local: NVML solo existe en equipos con driver NVIDIA
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None
    return memory_info.free / BYTES_PER_GB
