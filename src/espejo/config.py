"""Carga de la configuración YAML en dataclasses tipadas."""

from dataclasses import dataclass, replace
from pathlib import Path

from espejo.common.config_io import CONFIGS_DIR, read_yaml, resolve_project_path

DEFAULT_VISION_CONFIG = CONFIGS_DIR / "vision.yaml"
DEFAULT_AUDIO_CONFIG = CONFIGS_DIR / "audio.yaml"
DEFAULT_ANALYSIS_CONFIG = CONFIGS_DIR / "analysis.yaml"
DEFAULT_DATASET_CONFIG = CONFIGS_DIR / "dataset.yaml"


# --- Visión ---------------------------------------------------------------------------------


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
    teacher_min_conf: float
    similar_conf_margin: float
    gap_fill_s: float
    class_name: str


@dataclass(frozen=True)
class BehaviorConfig:
    detection_conf: float
    behavior_iou_threshold: float
    guide_class: str
    guide_min_conf: float
    min_guide_s: float
    board_writing_class: str
    stage_interaction_class: str


@dataclass(frozen=True)
class PoseConfig:
    enabled: bool
    detection_conf: float
    match_iou_threshold: float
    face_kp_conf: float
    shoulder_kp_conf: float


@dataclass(frozen=True)
class TimelineConfig:
    smoothing_window_s: float
    min_segment_s: float


@dataclass(frozen=True)
class PathConfig:
    median_window_frames: int
    max_step_ratio: float


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
    path: PathConfig
    heatmap: HeatmapConfig
    debug_video: DebugVideoConfig


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
        path=PathConfig(**raw["path"]),
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


# --- Audio ----------------------------------------------------------------------------------


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
    min_speakers: int


@dataclass(frozen=True)
class QuestionsConfig:
    interrogatives: list[str]
    join_segments_max_gap_s: float
    tag_words: list[str]


@dataclass(frozen=True)
class WaitTimeConfig:
    min_gap_s: float
    max_wait_s: float
    ok_threshold_s: float


@dataclass(frozen=True)
class ResponsesConfig:
    response_window_s: float


@dataclass(frozen=True)
class HallucinationConfig:
    min_vad_overlap: float
    max_words_for_vad_check: int
    phrases: list[str]


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
    responses: ResponsesConfig
    hallucinations: HallucinationConfig
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
        responses=ResponsesConfig(**raw["responses"]),
        hallucinations=HallucinationConfig(**raw["hallucinations"]),
        quality=QualityConfig(**raw["quality"]),
    )


# --- Unión de visión y audio ----------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringConfig:
    threshold: float
    max_episode_s: float


@dataclass(frozen=True)
class SuggestionFocusConfig:
    min_open_question_ratio: float


@dataclass(frozen=True)
class AnalysisConfig:
    monitoring: MonitoringConfig
    suggestion: SuggestionFocusConfig


def load_analysis_config(path: Path = DEFAULT_ANALYSIS_CONFIG) -> AnalysisConfig:
    raw = read_yaml(path)
    return AnalysisConfig(
        monitoring=MonitoringConfig(**raw["monitoring"]),
        suggestion=SuggestionFocusConfig(**raw["suggestion"]),
    )


# --- Dataset --------------------------------------------------------------------------------


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


# --- LLM ------------------------------------------------------------------------------------

DEFAULT_LLM_CONFIG = CONFIGS_DIR / "llm.yaml"


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str
    large_model: str
    small_model: str
    min_vram_gb_for_large: float
    temperature: float
    context_tokens: int
    request_timeout_s: float


@dataclass(frozen=True)
class AnthropicConfig:
    model: str
    api_key_env_var: str
    max_tokens: int
    fallback_beta: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


@dataclass(frozen=True)
class WindowConfig:
    window_s: float
    overlap_s: float


@dataclass(frozen=True)
class LlmValidationConfig:
    timestamp_tolerance_s: float
    whole_class_words: list[str]


@dataclass(frozen=True)
class SuggestionConfig:
    context_s: float


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    definitions_path: Path
    max_attempts: int
    ollama: OllamaConfig
    anthropic: AnthropicConfig
    windows: WindowConfig
    validation: LlmValidationConfig
    suggestion: SuggestionConfig


def load_llm_config(path: Path = DEFAULT_LLM_CONFIG) -> LlmConfig:
    raw = read_yaml(path)
    return LlmConfig(
        provider=raw["provider"],
        definitions_path=resolve_project_path(raw["definitions_path"]),
        max_attempts=raw["max_attempts"],
        ollama=OllamaConfig(**raw["ollama"]),
        anthropic=AnthropicConfig(**raw["anthropic"]),
        windows=WindowConfig(**raw["windows"]),
        validation=LlmValidationConfig(**raw["validation"]),
        suggestion=SuggestionConfig(**raw["suggestion"]),
    )
