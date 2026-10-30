"""Carga de la configuración YAML en dataclasses tipadas."""

from dataclasses import dataclass, replace
from pathlib import Path

from espejo.common.config_io import CONFIGS_DIR, read_yaml, resolve_project_path

DEFAULT_VISION_CONFIG = CONFIGS_DIR / "vision.yaml"
DEFAULT_DATASET_CONFIG = CONFIGS_DIR / "dataset.yaml"


@dataclass(frozen=True)
class SamplingConfig:
    sample_fps: float


@dataclass(frozen=True)
class ModelsConfig:
    teacher_weights: Path
    behavior_weights: Path
    pose_weights: str
    tracker_config: str
    image_size: int


@dataclass(frozen=True)
class TeacherConfig:
    detection_conf: float
    teacher_fallback_conf: float
    class_name: str


@dataclass(frozen=True)
class BehaviorConfig:
    detection_conf: float
    behavior_iou_threshold: float
    guide_class: str
    board_writing_class: str
    participation_classes: list[str]


@dataclass(frozen=True)
class PoseConfig:
    detection_conf: float
    match_iou_threshold: float
    face_kp_conf: float
    shoulder_kp_conf: float


@dataclass(frozen=True)
class TimelineConfig:
    smoothing_window_s: float
    min_segment_s: float


@dataclass(frozen=True)
class MetricsConfig:
    monitoring_threshold: float


@dataclass(frozen=True)
class HeatmapConfig:
    grid_columns: int
    grid_rows: int
    background_blur_kernel: int
    heat_blur_kernel: int
    overlay_alpha: float


@dataclass(frozen=True)
class DebugVideoConfig:
    box_thickness: int
    font_scale: float


@dataclass(frozen=True)
class VisionConfig:
    sampling: SamplingConfig
    models: ModelsConfig
    teacher: TeacherConfig
    behavior: BehaviorConfig
    pose: PoseConfig
    timeline: TimelineConfig
    metrics: MetricsConfig
    heatmap: HeatmapConfig
    debug_video: DebugVideoConfig


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    subset_folder: str
    raw_dir: Path
    processed_dir: Path
    reference_class: str
    preview_image_count: int
    preview_seed: int
    derived_datasets: dict[str, dict[str, str]]

    def subset_dir(self) -> Path:
        return self.raw_dir / self.subset_folder

    def source_yaml(self) -> Path:
        return self.subset_dir() / f"{self.subset_folder}.yaml"

    def processed_dataset_dir(self, dataset_name: str) -> Path:
        return self.processed_dir / dataset_name

    def data_yaml_path(self, dataset_name: str) -> Path:
        return CONFIGS_DIR / f"data_{dataset_name}.yaml"


def load_dataset_config(path: Path = DEFAULT_DATASET_CONFIG) -> DatasetConfig:
    raw = read_yaml(path)
    raw["raw_dir"] = resolve_project_path(raw["raw_dir"])
    raw["processed_dir"] = resolve_project_path(raw["processed_dir"])
    return DatasetConfig(**raw)


def load_vision_config(path: Path = DEFAULT_VISION_CONFIG) -> VisionConfig:
    raw = read_yaml(path)

    models_raw = dict(raw["models"])
    models_raw["teacher_weights"] = resolve_project_path(models_raw["teacher_weights"])
    models_raw["behavior_weights"] = resolve_project_path(models_raw["behavior_weights"])

    return VisionConfig(
        sampling=SamplingConfig(**raw["sampling"]),
        models=ModelsConfig(**models_raw),
        teacher=TeacherConfig(**raw["teacher"]),
        behavior=BehaviorConfig(**raw["behavior"]),
        pose=PoseConfig(**raw["pose"]),
        timeline=TimelineConfig(**raw["timeline"]),
        metrics=MetricsConfig(**raw["metrics"]),
        heatmap=HeatmapConfig(**raw["heatmap"]),
        debug_video=DebugVideoConfig(**raw["debug_video"]),
    )


def with_model_overrides(
    config: VisionConfig,
    teacher_weights: str | None,
    teacher_class: str | None,
    behavior_weights: str | None,
) -> VisionConfig:
    """Permite probar el pipeline con otros pesos (p. ej. yolov8s.pt y la clase "person")."""
    models = config.models
    teacher = config.teacher
    if teacher_weights is not None:
        models = replace(models, teacher_weights=Path(teacher_weights))
    if behavior_weights is not None:
        models = replace(models, behavior_weights=Path(behavior_weights))
    if teacher_class is not None:
        teacher = replace(teacher, class_name=teacher_class)
    return replace(config, models=models, teacher=teacher)


DEFAULT_AUDIO_CONFIG = CONFIGS_DIR / "audio.yaml"


@dataclass(frozen=True)
class ExtractionConfig:
    sample_rate_hz: int
    vad_highpass_hz: int


@dataclass(frozen=True)
class TranscriptionConfig:
    language: str
    device: str
    gpu_model: str
    gpu_fallback_model: str
    gpu_compute_type: str
    cpu_model: str
    cpu_compute_type: str
    min_free_gpu_memory_gb: float
    beam_size: int
    condition_on_previous_text: bool


@dataclass(frozen=True)
class VoiceActivityConfig:
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int


@dataclass(frozen=True)
class DiarizationConfig:
    enabled: bool
    model: str
    token_env_var: str


@dataclass(frozen=True)
class QuestionsConfig:
    interrogatives: list[str]
    join_segments_max_gap_s: float


@dataclass(frozen=True)
class WaitTimeConfig:
    min_gap_s: float
    max_wait_s: float
    ok_threshold_s: float


@dataclass(frozen=True)
class QualityConfig:
    min_avg_logprob: float
    max_no_speech_prob: float


@dataclass(frozen=True)
class AudioConfig:
    extraction: ExtractionConfig
    transcription: TranscriptionConfig
    voice_activity: VoiceActivityConfig
    diarization: DiarizationConfig
    questions: QuestionsConfig
    wait_time: WaitTimeConfig
    quality: QualityConfig


def load_audio_config(path: Path = DEFAULT_AUDIO_CONFIG) -> AudioConfig:
    raw = read_yaml(path)
    return AudioConfig(
        extraction=ExtractionConfig(**raw["extraction"]),
        transcription=TranscriptionConfig(**raw["transcription"]),
        voice_activity=VoiceActivityConfig(**raw["voice_activity"]),
        diarization=DiarizationConfig(**raw["diarization"]),
        questions=QuestionsConfig(**raw["questions"]),
        wait_time=WaitTimeConfig(**raw["wait_time"]),
        quality=QualityConfig(**raw["quality"]),
    )
