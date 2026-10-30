"""Orquesta VER (modelos, una sola vez) e INTERPRETAR (solo lee los JSON de VER)."""

import gc
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from espejo.analysis.llm_stage import with_llm_labels
from espejo.analysis.report_builder import AudioSide, ReportConfigs, VideoSide, build_class_report
from espejo.analysis.report_io import save_class_report
from espejo.analysis.rubric_evidence import evidence_as_text, evidence_times
from espejo.analysis.schemas import ClassReport
from espejo.analysis.transcript import save_transcript
from espejo.audio.capture import AudioCaptureRunner
from espejo.audio.capture_io import AUDIO_RAW_FILENAME, load_audio_capture
from espejo.audio.extraction import NoAudioTrackError
from espejo.audio.interpretation import AudioInterpreter
from espejo.config import (
    AnalysisConfig,
    AudioConfig,
    LlmConfig,
    VisionConfig,
    load_analysis_config,
    load_audio_config,
    load_llm_config,
    load_vision_config,
)
from espejo.llm.analyzer import LlmAnalyzer, without_llm
from espejo.llm.schemas import LlmAnalysis
from espejo.vision.debug_video import DebugVideoWriter
from espejo.vision.detection_run import BACKGROUND_FILENAME, VideoDetectionRunner
from espejo.vision.detections_io import DETECTIONS_FILENAME, load_detection_run
from espejo.vision.frame_sampler import FrameSampler
from espejo.vision.heatmap import HeatmapRenderer, save_image
from espejo.vision.interpretation import VisionInterpreter

logger = logging.getLogger(__name__)

REPORT_FILENAME = "class_report.json"
TRANSCRIPT_FILENAME = "transcript.txt"
HEATMAP_FILENAME = "heatmap.png"
DEBUG_VIDEO_FILENAME = "debug.mp4"


@dataclass(frozen=True)
class AllConfigs:
    vision: VisionConfig
    audio: AudioConfig
    analysis: AnalysisConfig
    llm: LlmConfig


@dataclass(frozen=True)
class InterpretationResult:
    report: ClassReport
    run_dir: Path
    video: VideoSide | None
    audio: AudioSide | None


def load_all_configs() -> AllConfigs:
    return AllConfigs(
        load_vision_config(), load_audio_config(), load_analysis_config(), load_llm_config()
    )


def run_dir_for(video_path: Path, output_root: Path) -> Path:
    return output_root / video_path.stem


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


def load_video_side(run_dir: Path, config: VisionConfig) -> VideoSide | None:
    detections_path = run_dir / DETECTIONS_FILENAME
    if not detections_path.exists():
        return None
    run = load_detection_run(detections_path)
    return VideoSide(run, VisionInterpreter(config).interpret(run))


def load_audio_side(run_dir: Path, config: AudioConfig) -> AudioSide | None:
    audio_path = run_dir / AUDIO_RAW_FILENAME
    if not audio_path.exists():
        return None
    capture = load_audio_capture(audio_path)
    return AudioSide(capture, AudioInterpreter(config).interpret(capture))


def interpret_run(
    run_dir: Path,
    configs: AllConfigs,
    include_video: bool = True,
    include_audio: bool = True,
    use_llm: bool = True,
) -> InterpretationResult:
    """INTERPRETAR: no abre el video ni el audio, solo detections.json y audio_raw.json.

    El LLM va al final, cuando YOLO y Whisper ya liberaron la GPU, y solo recibe texto.
    """
    video = None
    audio = None
    if include_video:
        video = load_video_side(run_dir, configs.vision)
    if include_audio:
        audio = load_audio_side(run_dir, configs.audio)
    if video is None and audio is None:
        raise FileNotFoundError(
            f"No hay {DETECTIONS_FILENAME} ni {AUDIO_RAW_FILENAME} en {run_dir}. "
            "Corre primero scripts/analyze_class.py"
        )

    analyzer = None
    llm_analysis = without_llm("desactivado con --no-llm")
    if audio is None:
        llm_analysis = without_llm("no hay audio")
    elif use_llm:
        analyzer = LlmAnalyzer(configs.llm)
        llm_analysis = classify_with_llm(analyzer, audio)
        audio = with_llm_labels(audio, llm_analysis)

    report_configs = ReportConfigs(configs.analysis, configs.audio)
    report = build_class_report(video, audio, report_configs, llm_analysis)
    if analyzer is not None and audio is not None:
        report = add_suggestion(analyzer, report, audio.capture.topic)
        analyzer.release()
    save_class_report(report, run_dir / REPORT_FILENAME)

    if audio is not None:
        save_transcript(audio.interpretation.segments, run_dir / TRANSCRIPT_FILENAME)
    if video is not None:
        save_heatmap(video, run_dir, configs.vision)
    return InterpretationResult(report, run_dir, video, audio)


def classify_with_llm(analyzer: LlmAnalyzer, audio: AudioSide) -> LlmAnalysis:
    interpretation = audio.interpretation
    return analyzer.classify(interpretation.segments, interpretation.questions, audio.capture.topic)


def add_suggestion(analyzer: LlmAnalyzer, report: ClassReport, topic: str | None) -> ClassReport:
    """Último llamado al LLM: un consejo basado solo en la ficha por rúbrica."""
    rubric_evidence = report.rubric_evidence
    analysis = analyzer.add_suggestion(
        report.llm, evidence_as_text(rubric_evidence), evidence_times(rubric_evidence), topic
    )
    return replace(report, llm=analysis)


def save_heatmap(video: VideoSide, run_dir: Path, config: VisionConfig) -> None:
    background_path = run_dir / BACKGROUND_FILENAME
    background = cv2.imread(str(background_path))
    if background is None:
        logger.warning("No se encontró %s; no se genera el heatmap", background_path.name)
        return

    teacher_boxes = []
    for observation in video.interpretation.observations:
        if observation.teacher_box is not None:
            teacher_boxes.append(observation.teacher_box)
    heatmap = HeatmapRenderer(config.heatmap).render(background, teacher_boxes)
    save_image(heatmap, run_dir / HEATMAP_FILENAME)


def write_debug_video(video_path: Path, result: InterpretationResult, config: VisionConfig) -> Path:
    """Opcional: vuelve a leer el video para dibujar al docente y su estado (no va a git)."""
    if result.video is None:
        raise ValueError("El video de depuración necesita detections.json.")

    output_path = result.run_dir / DEBUG_VIDEO_FILENAME
    sampler = FrameSampler(video_path, config.sampling.sample_fps)
    observations = result.video.interpretation.observations
    DebugVideoWriter(config.debug_video).write(sampler, observations, output_path)
    return output_path
