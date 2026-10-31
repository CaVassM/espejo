"""Lectura del video y muestreo de frames a una frecuencia fija."""

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2

from espejo.vision.schemas import Frame, VideoInfo

logger = logging.getLogger(__name__)


class FrameSampler:
    """Entrega solo los frames necesarios para muestrear a sample_fps."""

    def __init__(self, video_path: Path, sample_fps: float) -> None:
        if not video_path.exists():
            raise FileNotFoundError(f"No se encontró el video {video_path}.")
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.info = read_video_info(video_path)

        if self.sample_fps > self.info.native_fps:
            logger.warning(
                "sample_fps (%.1f) supera los fps del video (%.1f); se usarán todos los frames",
                self.sample_fps,
                self.info.native_fps,
            )

    def frames(self) -> Iterator[Frame]:
        capture = open_capture(self.video_path)
        frames_per_sample = max(1.0, self.info.native_fps / self.sample_fps)
        next_sample_index = 0.0
        frame_index = 0

        try:
            while capture.grab():
                # grab() avanza sin decodificar; solo decodificamos los frames que usamos
                if frame_index >= next_sample_index:
                    success, image = capture.retrieve()
                    if success:
                        timestamp_s = frame_index / self.info.native_fps
                        yield Frame(frame_index, timestamp_s, image)
                    next_sample_index += frames_per_sample
                frame_index += 1
        finally:
            capture.release()


def open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV no pudo abrir el video {video_path}. ¿Está dañado el archivo?")
    return capture


def read_video_info(video_path: Path) -> VideoInfo:
    capture = open_capture(video_path)
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if native_fps <= 0:
        raise ValueError(f"El video {video_path} no informa sus fps; conviértelo a mp4 primero.")

    return VideoInfo(
        path=str(video_path),
        duration_s=frame_count / native_fps,
        native_fps=native_fps,
        width=width,
        height=height,
    )
