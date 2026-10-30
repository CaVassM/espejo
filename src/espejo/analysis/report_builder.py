"""Arma el reporte de la clase a partir de las interpretaciones de video y audio."""

from dataclasses import dataclass
from pathlib import Path

from espejo.analysis.monitoring import (
    MonitoringResult,
    compute_monitoring,
    guide_intervals,
    question_episodes,
)
from espejo.analysis.rubric_metrics import build_rubric_metrics
from espejo.analysis.schemas import ClassReport, DataQuality, ReportMetadata
from espejo.audio.schemas import AudioCapture, AudioInterpretation
from espejo.config import AnalysisConfig, AudioConfig
from espejo.vision.schemas import DetectionRun, VisionInterpretation

VAD_MODEL_NAME = "silero-vad (incluido en faster-whisper)"
NO_AUDIO_NOTE = "Sin audio: el monitoreo se calcula solo con el video."
NO_VIDEO_NOTE = "Sin video: el monitoreo se calcula solo con el audio."
NO_DIARIZATION_NOTE = (
    "Sin diarización: se cuentan las preguntas de cualquier hablante y las respuestas "
    "tienen certeza baja; el monitoreo por audio es menos preciso."
)


@dataclass(frozen=True)
class VideoSide:
    run: DetectionRun
    interpretation: VisionInterpretation


@dataclass(frozen=True)
class ReportConfigs:
    analysis: AnalysisConfig
    audio: AudioConfig


@dataclass(frozen=True)
class AudioSide:
    capture: AudioCapture
    interpretation: AudioInterpretation


def session_duration_s(video: VideoSide | None, audio: AudioSide | None) -> float:
    if video is not None:
        return video.interpretation.metrics.analyzed_duration_s
    if audio is not None:
        return audio.capture.duration_s
    raise ValueError("No hay ni video ni audio para armar el reporte.")


def monitoring_for(
    video: VideoSide | None, audio: AudioSide | None, duration_s: float, configs: ReportConfigs
) -> MonitoringResult:
    video_intervals = None
    audio_intervals = None
    if video is not None:
        video_intervals = guide_intervals(video.interpretation.timeline)
    if audio is not None:
        max_wait_s = configs.audio.wait_time.max_wait_s
        questions = audio.interpretation.questions
        audio_intervals = question_episodes(questions, max_wait_s, duration_s)

    threshold = configs.analysis.monitoring.threshold
    return compute_monitoring(video_intervals, audio_intervals, duration_s, threshold)


def build_models(video: VideoSide | None, audio: AudioSide | None) -> dict[str, str]:
    models: dict[str, str] = {}
    if video is not None:
        models.update(video.run.models)
    if audio is not None:
        transcriber = audio.capture.transcriber
        models["whisper"] = f"{transcriber.model_name} ({transcriber.device})"
        models["vad"] = VAD_MODEL_NAME
        models["diarization"] = audio.interpretation.diarization_note
    return models


def build_metadata(video: VideoSide | None, audio: AudioSide | None) -> ReportMetadata:
    source_path = ""
    sample_fps = None
    if video is not None:
        source_path = video.run.video.path
        sample_fps = video.run.sample_fps
    elif audio is not None:
        source_path = audio.capture.video_path

    diarization_used = audio is not None and audio.interpretation.diarization_used
    return ReportMetadata(
        video=Path(source_path).name,
        duration_s=session_duration_s(video, audio),
        sample_fps=sample_fps,
        models=build_models(video, audio),
        video_available=video is not None,
        audio_available=audio is not None,
        diarization_used=diarization_used,
    )


def build_data_quality(video: VideoSide | None, audio: AudioSide | None) -> DataQuality:
    notes = []
    teacher_visible_ratio = None
    discarded_jumps = None
    if video is None:
        notes.append(NO_VIDEO_NOTE)
    else:
        teacher_visible_ratio = video.interpretation.metrics.teacher_visible_ratio
        discarded_jumps = video.interpretation.metrics.path.discarded_jumps

    low_confidence_ratio = None
    speech_ratio = None
    diarization_used = False
    diarization_note = NO_AUDIO_NOTE
    if audio is None:
        notes.append(NO_AUDIO_NOTE)
    else:
        low_confidence_ratio = audio.interpretation.metrics.low_confidence_ratio
        speech_ratio = audio.interpretation.metrics.speech_ratio
        diarization_used = audio.interpretation.diarization_used
        diarization_note = audio.interpretation.diarization_note
        if not diarization_used:
            notes.append(NO_DIARIZATION_NOTE)

    return DataQuality(
        teacher_visible_ratio=teacher_visible_ratio,
        discarded_path_jumps=discarded_jumps,
        low_confidence_audio_ratio=low_confidence_ratio,
        speech_ratio=speech_ratio,
        diarization_used=diarization_used,
        diarization_note=diarization_note,
        notes=notes,
    )


def build_class_report(
    video: VideoSide | None, audio: AudioSide | None, configs: ReportConfigs
) -> ClassReport:
    duration_s = session_duration_s(video, audio)
    monitoring = monitoring_for(video, audio, duration_s, configs)

    vision_interpretation = None
    timeline = []
    stage_events = []
    if video is not None:
        vision_interpretation = video.interpretation
        timeline = video.interpretation.timeline
        stage_events = video.interpretation.stage_interaction_events

    audio_interpretation = None
    questions = []
    if audio is not None:
        audio_interpretation = audio.interpretation
        questions = audio.interpretation.questions

    metrics = build_rubric_metrics(
        vision_interpretation,
        audio_interpretation,
        monitoring,
        configs.analysis.monitoring.threshold,
        configs.audio.wait_time.ok_threshold_s,
    )
    return ClassReport(
        metadata=build_metadata(video, audio),
        video_timeline=timeline,
        stage_interaction_events=stage_events,
        questions=questions,
        metrics=metrics,
        data_quality=build_data_quality(video, audio),
    )
