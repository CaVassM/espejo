"""VER (video): pasa el video una sola vez por los modelos y guarda detections.json."""

import logging
from pathlib import Path

from espejo.config import VisionConfig
from espejo.vision.detections_io import (
    BACKGROUND_FILENAME,
    DETECTIONS_FILENAME,
    save_detection_run,
)
from espejo.vision.detectors import BehaviorDetector, PoseEstimator, TeacherDetector
from espejo.vision.frame_sampler import FrameSampler
from espejo.vision.heatmap import HeatmapRenderer, save_image
from espejo.vision.schemas import DetectionRun, Frame, FrameDetections

logger = logging.getLogger(__name__)

PROGRESS_LOG_INTERVAL_FRAMES = 100


class VideoDetectionRunner:
    """Corre los modelos de visión y guarda solo coordenadas, nunca frames ni recortes."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.teacher_detector = TeacherDetector(config.models, config.teacher)
        self.behavior_detector = BehaviorDetector(config.models, config.behavior)
        self.heatmap_renderer = HeatmapRenderer(config.heatmap)

        self.pose_estimator = None
        if config.pose.enabled:
            self.pose_estimator = PoseEstimator(config.models, config.pose)

    def run(self, video_path: Path, output_dir: Path) -> DetectionRun:
        sampler = FrameSampler(video_path, self.config.sampling.sample_fps)
        self.teacher_detector.reset_tracking()
        frames = []

        for frame in sampler.frames():
            if not frames:
                self.save_background(frame, output_dir)
            frames.append(self.detect_frame(frame))
            if len(frames) % PROGRESS_LOG_INTERVAL_FRAMES == 0:
                logger.info("Procesados %d frames (t=%.0f s)", len(frames), frame.timestamp_s)

        if not frames:
            raise ValueError(f"No se pudo leer ningún frame de {video_path}.")

        run = DetectionRun(sampler.info, sampler.sample_fps, self.model_names(), frames)
        save_detection_run(run, output_dir / DETECTIONS_FILENAME)
        return run

    def detect_frame(self, frame: Frame) -> FrameDetections:
        poses = []
        if self.pose_estimator is not None:
            poses = self.pose_estimator.estimate(frame)

        return FrameDetections(
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            teacher_detections=self.teacher_detector.track(frame),
            behavior_detections=self.behavior_detector.detect(frame),
            poses=poses,
        )

    def save_background(self, frame: Frame, output_dir: Path) -> None:
        """Se guarda solo el primer frame y ya desenfocado: es el fondo del heatmap."""
        blurred = self.heatmap_renderer.blur_background(frame.image)
        save_image(blurred, output_dir / BACKGROUND_FILENAME)

    def model_names(self) -> dict[str, str]:
        """Solo el nombre del archivo: no exponemos rutas locales en los reportes."""
        models = self.config.models
        names = {
            "teacher": Path(models.teacher_weights).name,
            "teacher_class": self.config.teacher.class_name,
            "behavior": Path(models.behavior_weights).name,
            "tracker": models.tracker_config,
        }
        if self.config.pose.enabled:
            names["pose"] = Path(models.pose_weights).name
        return names
