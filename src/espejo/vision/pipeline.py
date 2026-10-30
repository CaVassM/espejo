"""VisionPipeline: orquesta las etapas del módulo de visión sobre un video."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from espejo.config import VisionConfig
from espejo.vision.debug_video import DebugVideoWriter
from espejo.vision.detectors import BehaviorDetector, PoseEstimator, TeacherDetector
from espejo.vision.frame_sampler import FrameSampler
from espejo.vision.geometry import BoundingBox
from espejo.vision.heatmap import HeatmapRenderer, save_image
from espejo.vision.metrics import MetricsCalculator
from espejo.vision.orientation import estimate_teacher_orientation
from espejo.vision.report_writer import save_report_json
from espejo.vision.schemas import (
    FrameDetections,
    TeacherObservation,
    VideoInfo,
    VisionReport,
)
from espejo.vision.teacher_tracker import TeacherTracker
from espejo.vision.timeline import TimedLabel, TimelineBuilder

logger = logging.getLogger(__name__)

REPORT_FILENAME = "vision_report.json"
HEATMAP_FILENAME = "heatmap.png"
DEBUG_VIDEO_FILENAME = "debug.mp4"
PROGRESS_LOG_INTERVAL_FRAMES = 100


@dataclass(frozen=True, eq=False)
class DetectionPass:
    """Resultado de recorrer el video una vez. Solo guarda el primer frame, para el heatmap."""

    frames: list[FrameDetections]
    first_image: np.ndarray


@dataclass(frozen=True)
class PipelineResult:
    report: VisionReport
    report_path: Path
    heatmap_path: Path
    debug_video_path: Path | None


class VisionPipeline:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.teacher_detector = TeacherDetector(config.models, config.teacher)
        self.behavior_detector = BehaviorDetector(config.models, config.behavior)
        self.pose_estimator = PoseEstimator(config.models, config.pose)
        self.tracker = TeacherTracker(config.teacher.teacher_fallback_conf)
        self.timeline_builder = TimelineBuilder(
            config.timeline.smoothing_window_s,
            config.timeline.min_segment_s,
        )
        self.metrics_calculator = MetricsCalculator(
            config.metrics.monitoring_threshold,
            config.heatmap.grid_columns,
            config.heatmap.grid_rows,
        )
        self.heatmap_renderer = HeatmapRenderer(config.heatmap)

    def run(self, video_path: Path, output_root: Path, with_debug_video: bool) -> PipelineResult:
        sampler = FrameSampler(video_path, self.config.sampling.sample_fps)
        detection_pass = self.detect_all_frames(sampler)
        observations = self.build_observations(detection_pass.frames)
        report = self.build_report(sampler, observations)

        output_dir = output_root / video_path.stem
        report_path = output_dir / REPORT_FILENAME
        save_report_json(report, report_path)

        heatmap_path = output_dir / HEATMAP_FILENAME
        teacher_boxes = visible_teacher_boxes(observations)
        heatmap = self.heatmap_renderer.render(detection_pass.first_image, teacher_boxes)
        save_image(heatmap, heatmap_path)

        debug_video_path = None
        if with_debug_video:
            debug_video_path = output_dir / DEBUG_VIDEO_FILENAME
            debug_sampler = FrameSampler(video_path, self.config.sampling.sample_fps)
            DebugVideoWriter(self.config.debug_video).write(
                debug_sampler, observations, debug_video_path
            )

        return PipelineResult(report, report_path, heatmap_path, debug_video_path)

    def detect_all_frames(self, sampler: FrameSampler) -> DetectionPass:
        """Una sola pasada por el video corriendo los tres modelos; no guarda imágenes."""
        frames = []
        first_image = None

        for frame in sampler.frames():
            if first_image is None:
                first_image = frame.image.copy()

            frame_detections = FrameDetections(
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                teacher_detections=self.teacher_detector.track(frame),
                behavior_detections=self.behavior_detector.detect(frame),
                poses=self.pose_estimator.estimate(frame),
            )
            frames.append(frame_detections)

            if len(frames) % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                logger.info("Procesados %d frames (t=%.0f s)", len(frames), frame.timestamp_s)

        if first_image is None:
            raise ValueError(f"No se pudo leer ningún frame de {sampler.video_path}.")
        return DetectionPass(frames, first_image)

    def build_observations(self, frames: list[FrameDetections]) -> list[TeacherObservation]:
        """Segunda etapa: con el docente ya identificado, asigna conducta y orientación."""
        detections_per_frame = []
        for frame_detections in frames:
            detections_per_frame.append(frame_detections.teacher_detections)
        teacher_boxes = self.tracker.select_teacher_boxes(detections_per_frame)

        observations = []
        for frame_detections, teacher_box in zip(frames, teacher_boxes, strict=True):
            observations.append(self.observe(frame_detections, teacher_box))
        return observations

    def observe(
        self, frame_detections: FrameDetections, teacher_box: BoundingBox | None
    ) -> TeacherObservation:
        behavior_result = self.behavior_detector.assign(
            frame_detections.behavior_detections, teacher_box
        )
        orientation = estimate_teacher_orientation(
            teacher_box, frame_detections.poses, self.config.pose
        )
        return TeacherObservation(
            frame_index=frame_detections.frame_index,
            timestamp_s=frame_detections.timestamp_s,
            teacher_box=teacher_box,
            behavior=behavior_result.behavior,
            orientation=orientation,
            participation_events=behavior_result.participation_events,
        )

    def build_report(
        self, sampler: FrameSampler, observations: list[TeacherObservation]
    ) -> VisionReport:
        end_s = analysis_end_s(observations, sampler.info, sampler.sample_fps)
        behavior_timeline = self.timeline_builder.build(behavior_labels(observations), end_s)
        orientation_timeline = self.timeline_builder.build(orientation_labels(observations), end_s)

        metrics = self.metrics_calculator.calculate(
            observations,
            behavior_timeline,
            orientation_timeline,
            sampler.info.width,
            sampler.info.height,
        )
        return VisionReport(
            video=sampler.info,
            sample_fps=sampler.sample_fps,
            models=self.model_names(),
            behavior_timeline=behavior_timeline,
            orientation_timeline=orientation_timeline,
            metrics=metrics,
        )

    def model_names(self) -> dict[str, str]:
        """Solo el nombre del archivo: no exponemos rutas locales en el reporte."""
        models = self.config.models
        return {
            "teacher": Path(models.teacher_weights).name,
            "teacher_class": self.config.teacher.class_name,
            "behavior": Path(models.behavior_weights).name,
            "pose": Path(models.pose_weights).name,
            "tracker": models.tracker_config,
        }


def behavior_labels(observations: list[TeacherObservation]) -> list[TimedLabel]:
    labels = []
    for observation in observations:
        labels.append(TimedLabel(observation.timestamp_s, observation.behavior))
    return labels


def orientation_labels(observations: list[TeacherObservation]) -> list[TimedLabel]:
    labels = []
    for observation in observations:
        labels.append(TimedLabel(observation.timestamp_s, observation.orientation))
    return labels


def visible_teacher_boxes(observations: list[TeacherObservation]) -> list[BoundingBox]:
    boxes = []
    for observation in observations:
        if observation.teacher_box is not None:
            boxes.append(observation.teacher_box)
    return boxes


def analysis_end_s(
    observations: list[TeacherObservation], video: VideoInfo, sample_fps: float
) -> float:
    """El último frame muestreado representa un intervalo de muestreo, sin pasarse del video.

    Algunos celulares reportan mal la cantidad de frames; si la duración informada es menor
    que el último frame leído, confiamos en los frames.
    """
    if not observations:
        return video.duration_s

    last_timestamp_s = observations[-1].timestamp_s
    last_sample_end_s = last_timestamp_s + 1 / sample_fps
    if video.duration_s > last_timestamp_s:
        return min(last_sample_end_s, video.duration_s)
    return last_sample_end_s
