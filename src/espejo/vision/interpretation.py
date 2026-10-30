"""INTERPRETAR (video): convierte detections.json en estados, línea de tiempo y métricas."""

from espejo.config import VisionConfig
from espejo.vision.metrics import VisionMetricsCalculator
from espejo.vision.schemas import (
    Behavior,
    DetectionRun,
    FrameDetections,
    Segment,
    TeacherObservation,
    VideoInfo,
    VisionInterpretation,
)
from espejo.vision.teacher_state import assign_teacher_state, has_stage_interaction
from espejo.vision.teacher_tracker import TeacherSelection, TeacherTracker
from espejo.vision.timeline import TimedLabel, TimelineBuilder, relabel_short_segments

STAGE_INTERACTION_LABEL = "stage_interaction"
NO_STAGE_INTERACTION_LABEL = "none"


class VisionInterpreter:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.tracker = TeacherTracker(config.teacher)
        self.timeline_builder = TimelineBuilder(
            config.timeline.smoothing_window_s,
            config.timeline.min_segment_s,
        )
        self.metrics_calculator = VisionMetricsCalculator(config.path)

    def interpret(self, run: DetectionRun) -> VisionInterpretation:
        observations = self.build_observations(run.frames)
        end_s = analysis_end_s(observations, run.video, run.sample_fps)

        timeline = self.timeline_builder.build(state_labels(observations), end_s)
        timeline = relabel_short_segments(
            timeline, Behavior.GUIDE, self.config.behavior.min_guide_s, Behavior.OTHER
        )
        stage_events = self.stage_interaction_events(observations, end_s)
        metrics = self.metrics_calculator.calculate(
            observations, timeline, stage_events, run.video.width
        )
        return VisionInterpretation(observations, timeline, stage_events, metrics)

    def build_observations(self, frames: list[FrameDetections]) -> list[TeacherObservation]:
        """Primero se decide quién es el docente en cada frame; luego, su estado."""
        selections = self.tracker.select(frames)

        observations: list[TeacherObservation] = []
        for frame, selection in zip(frames, selections, strict=True):
            previous = None
            if observations:
                previous = observations[-1]
            observations.append(self.observe(frame, selection, previous))
        return observations

    def observe(
        self,
        frame: FrameDetections,
        selection: TeacherSelection,
        previous: TeacherObservation | None,
    ) -> TeacherObservation:
        """En un frame interpolado el docente conserva el estado del frame anterior."""
        behavior_config = self.config.behavior
        state = assign_teacher_state(frame.behavior_detections, selection.box, behavior_config)
        if selection.interpolated and previous is not None:
            state = previous.state

        return TeacherObservation(
            frame_index=frame.frame_index,
            timestamp_s=frame.timestamp_s,
            teacher_box=selection.box,
            state=state,
            stage_interaction_present=has_stage_interaction(
                frame.behavior_detections, behavior_config
            ),
            interpolated=selection.interpolated,
        )

    def stage_interaction_events(
        self, observations: list[TeacherObservation], end_s: float
    ) -> list[Segment]:
        """Cada tramo continuo con un alumno al frente es UN evento, no uno por frame."""
        presence_timeline = self.timeline_builder.build(stage_labels(observations), end_s)
        events = []
        for segment in presence_timeline:
            if segment.label == STAGE_INTERACTION_LABEL:
                events.append(segment)
        return events


def state_labels(observations: list[TeacherObservation]) -> list[TimedLabel]:
    labels = []
    for observation in observations:
        labels.append(TimedLabel(observation.timestamp_s, observation.state))
    return labels


def stage_labels(observations: list[TeacherObservation]) -> list[TimedLabel]:
    labels = []
    for observation in observations:
        label = NO_STAGE_INTERACTION_LABEL
        if observation.stage_interaction_present:
            label = STAGE_INTERACTION_LABEL
        labels.append(TimedLabel(observation.timestamp_s, label))
    return labels


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
