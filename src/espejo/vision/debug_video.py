"""Video de depuración con la caja, la conducta y la orientación del docente dibujadas."""

import logging
from pathlib import Path

import cv2
import numpy as np

from espejo.config import DebugVideoConfig
from espejo.vision.frame_sampler import FrameSampler
from espejo.vision.schemas import TeacherObservation

logger = logging.getLogger(__name__)

TEACHER_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_ORIGIN = (20, 40)
TEXT_MARGIN_PX = 8
VIDEO_CODEC = "mp4v"
STAGE_INTERACTION_CAPTION = " | alumno al frente"


class DebugVideoWriter:
    """Vuelve a leer el video y dibuja las observaciones ya calculadas, frame a frame."""

    def __init__(self, config: DebugVideoConfig) -> None:
        self.config = config

    def write(
        self,
        sampler: FrameSampler,
        observations: list[TeacherObservation],
        output_path: Path,
    ) -> None:
        observations_by_index = {}
        for observation in observations:
            observations_by_index[observation.frame_index] = observation

        frame_size = (sampler.info.width, sampler.info.height)
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(str(output_path), fourcc, sampler.sample_fps, frame_size)

        try:
            for frame in sampler.frames():
                observation = observations_by_index.get(frame.index)
                if observation is not None:
                    self.draw_observation(frame.image, observation)
                writer.write(frame.image)
        finally:
            writer.release()
        logger.info("Video de depuración guardado en %s", output_path)

    def draw_observation(self, image: np.ndarray, observation: TeacherObservation) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = self.config.box_thickness
        caption = str(observation.state)
        if observation.stage_interaction_present:
            caption += STAGE_INTERACTION_CAPTION
        header = f"t={observation.timestamp_s:.1f}s  {caption}"
        cv2.putText(image, header, TEXT_ORIGIN, font, self.config.font_scale, TEXT_COLOR, thickness)

        box = observation.teacher_box
        if box is None:
            return

        top_left = (int(box.x1), int(box.y1))
        bottom_right = (int(box.x2), int(box.y2))
        label_origin = (int(box.x1), max(int(box.y1) - TEXT_MARGIN_PX, 0))
        cv2.rectangle(image, top_left, bottom_right, TEACHER_COLOR, thickness)
        cv2.putText(
            image, "docente", label_origin, font, self.config.font_scale, TEACHER_COLOR, thickness
        )
