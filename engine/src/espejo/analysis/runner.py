"""INTERPRETAR: lee los JSON de VER y arma el reporte. No carga torch, YOLO ni Whisper.

VER (los modelos) está en analysis/capture.py y necesita espejo[vision,audio].
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from espejo.analysis.focus import FocusThresholds, choose_focus
from espejo.analysis.llm_stage import with_llm_labels
from espejo.analysis.report_builder import AudioSide, ReportConfigs, VideoSide, build_class_report
from espejo.analysis.report_io import save_class_report
from espejo.analysis.schemas import ClassReport
from espejo.analysis.transcript import save_transcript
from espejo.audio.capture_io import AUDIO_RAW_FILENAME, load_audio_capture
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
from espejo.vision.detections_io import (
    BACKGROUND_FILENAME,
    DETECTIONS_FILENAME,
    load_detection_run,
)
from espejo.vision.interpretation import VisionInterpreter

logger = logging.getLogger(__name__)

REPORT_FILENAME = "class_report.json"
TRANSCRIPT_FILENAME = "transcript.txt"
HEATMAP_FILENAME = "heatmap.png"


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
        report = add_suggestion(analyzer, report, audio, configs)
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


def focus_thresholds(configs: AllConfigs) -> FocusThresholds:
    return FocusThresholds(
        wait_ok_s=configs.audio.wait_time.ok_threshold_s,
        monitoring=configs.analysis.monitoring.threshold,
        min_open_question_ratio=configs.analysis.suggestion.min_open_question_ratio,
    )


def add_suggestion(
    analyzer: LlmAnalyzer, report: ClassReport, audio: AudioSide, configs: AllConfigs
) -> ClassReport:
    """Último llamado al LLM: un consejo sobre la debilidad más importante de la ficha."""
    focus = choose_focus(
        report.metrics, report.questions, report.llm.feedback, focus_thresholds(configs)
    )
    analysis = analyzer.add_suggestion(
        report.llm, focus, audio.interpretation.segments, audio.capture.topic
    )
    return replace(report, llm=analysis)


def save_heatmap(video: VideoSide, run_dir: Path, config: VisionConfig) -> None:
    try:
        # Import local: OpenCV es parte de espejo[vision]; sin él se interpreta sin heatmap
        import cv2

        from espejo.vision.heatmap import HeatmapRenderer, save_image
    except ImportError:
        logger.warning("Sin OpenCV (espejo[vision]): no se genera el heatmap")
        return

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
