"""VER con el motor completo (espejo[vision,audio]). Modelos cargados una vez por contenedor."""

import logging
from collections.abc import Callable
from pathlib import Path

from espejo.analysis.runner import load_all_configs, load_video_side, save_heatmap
from espejo.audio.capture import AudioCaptureRunner
from espejo.audio.extraction import NoAudioTrackError
from espejo.vision.detection_run import VideoDetectionRunner
from espejo.vision.detections_io import BACKGROUND_FILENAME
from espejo_worker.jobs import Stage

logger = logging.getLogger(__name__)


class VerPipeline:
    def __init__(self) -> None:
        """Carga YOLO, Whisper, Silero, spaCy y pyannote: es lo lento del arranque en frío."""
        self.configs = load_all_configs()
        self.video_runner = VideoDetectionRunner(self.configs.vision)
        self.audio_runner = AudioCaptureRunner(self.configs.audio)
        self.audio_runner.diarizer.preload()

    def run(
        self, video_path: Path, run_dir: Path, topic: str | None, on_stage: Callable[[Stage], None]
    ) -> None:
        """Deja en run_dir detections.json, audio_raw.json (ya anonimizado) y heatmap.png."""
        on_stage(Stage.VIDEO)
        self.video_runner.run(video_path, run_dir)

        on_stage(Stage.AUDIO)
        try:
            self.audio_runner.run(video_path, run_dir, keep_audio=False, topic=topic)
        except NoAudioTrackError as error:
            logger.warning("%s Se entrega solo el video.", error)

        video = load_video_side(run_dir, self.configs.vision)
        if video is not None:
            save_heatmap(video, run_dir, self.configs.vision)
        # El primer frame (desenfocado) solo sirve de fondo del heatmap: no sale del worker
        (run_dir / BACKGROUND_FILENAME).unlink(missing_ok=True)
