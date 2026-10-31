"""Métricas finales, cada una con la rúbrica del Minedu a la que aporta evidencia."""

from espejo.analysis.monitoring import MonitoringResult
from espejo.analysis.schemas import MetricValue, RubricMetric
from espejo.audio.schemas import AudioInterpretation
from espejo.vision.schemas import Behavior, VisionInterpretation

R1 = "R1"  # Involucra activamente a los estudiantes en el proceso de aprendizaje
R2 = "R2"  # Promueve el razonamiento, la creatividad y/o el pensamiento crítico
R3 = "R3"  # Evalúa el progreso de los aprendizajes
R4 = "R4"  # Propicia un ambiente de respeto y proximidad
NO_VIDEO_NOTE = "sin video"
NO_AUDIO_NOTE = "sin audio"
NO_LLM_NOTE = "sin análisis LLM"
EXPERIMENTAL_NOTE = "experimental"
NOT_AVAILABLE_NOTE = "no disponible: el modelo no tuvo detecciones confiables de {what}"


def frames_in_state(vision: VisionInterpretation, state: Behavior) -> int:
    count = 0
    for observation in vision.observations:
        if observation.state == state:
            count += 1
    return count


def experimental_value(
    vision: VisionInterpretation | None, state: Behavior, value: MetricValue, what: str
) -> tuple[MetricValue, str]:
    """Una conducta experimental sin ninguna detección confiable no es 0 %: es "no disponible"."""
    if vision is None:
        return None, NO_VIDEO_NOTE
    if frames_in_state(vision, state) == 0:
        return None, NOT_AVAILABLE_NOTE.format(what=what)
    return value, EXPERIMENTAL_NOTE


def video_metrics(vision: VisionInterpretation | None) -> list[RubricMetric]:
    board_writing = None
    stage_count = None
    path_length = None
    note = NO_VIDEO_NOTE
    if vision is not None:
        board_writing = vision.metrics.board_writing_ratio
        stage_count = vision.metrics.stage_interaction_count
        path_length = vision.metrics.path.length_normalized
        note = None
    board_writing, board_note = experimental_value(
        vision, Behavior.BOARD_WRITING, board_writing, "escribir en la pizarra"
    )

    return [
        RubricMetric(
            "board_writing_ratio",
            board_writing,
            [R1],
            "Tiempo escribiendo en la pizarra, de espaldas a la clase",
            board_note,
            experimental=True,
        ),
        RubricMetric(
            "stage_interaction_count",
            stage_count,
            [R1],
            "Veces que un alumno pasó al frente",
            note,
        ),
        RubricMetric(
            "path_length_normalized",
            path_length,
            [R4],
            "Recorrido del docente en anchos de frame (evidencia visual: heatmap.png)",
            note,
        ),
    ]


def monitoring_metrics(
    monitoring: MonitoringResult, threshold: float, vision: VisionInterpretation | None
) -> list[RubricMetric]:
    video_ratio, video_note = experimental_value(
        vision, Behavior.GUIDE, monitoring.video_ratio, "guiar en las carpetas"
    )
    return [
        RubricMetric(
            "monitoring_ratio",
            monitoring.ratio,
            [R3],
            "Tiempo monitoreando: desde cada pregunta hasta la respuesta de un alumno (audio)",
            None if monitoring.ratio is not None else NO_AUDIO_NOTE,
        ),
        RubricMetric(
            "monitoring_video_ratio",
            video_ratio,
            [R3],
            "Complemento experimental: tiempo guiando en las carpetas (video)",
            video_note,
            experimental=True,
        ),
        RubricMetric(
            "monitoring_audio_ratio",
            monitoring.audio_ratio,
            [R3],
            "Parte del monitoreo que se oye: desde cada pregunta hasta su respuesta",
        ),
        RubricMetric(
            "meets_monitoring_threshold",
            monitoring.meets_threshold,
            [R3],
            f"Monitorea al menos el {threshold:.0%} de la sesión, como pide el manual",
        ),
    ]


def audio_metrics(
    audio: AudioInterpretation | None, wait_ok_threshold_s: float
) -> list[RubricMetric]:
    if audio is None:
        return empty_audio_metrics()

    metrics = audio.metrics
    return [
        RubricMetric(
            "teacher_question_count", metrics.teacher_question_count, [R2, R3], "Preguntas"
        ),
        RubricMetric(
            "questions_per_minute", metrics.questions_per_minute, [R2, R3], "Preguntas por minuto"
        ),
        RubricMetric(
            "rhetorical_question_count",
            metrics.rhetorical_question_count,
            [R2],
            "Muletillas o preguntas retóricas (no cuentan como preguntas)",
        ),
        RubricMetric(
            "open_question_ratio",
            metrics.open_question_ratio,
            [R2],
            "Preguntas abiertas / (abiertas + cerradas), según el LLM",
            open_ratio_note(metrics.open_question_ratio),
        ),
        RubricMetric(
            "mean_wait_time_s",
            metrics.mean_wait_time_s,
            [R2],
            "Espera promedio después de preguntar (solo audio confiable)",
        ),
        RubricMetric(
            "median_wait_time_s",
            metrics.median_wait_time_s,
            [R2],
            "Espera mediana después de preguntar (solo audio confiable)",
        ),
        RubricMetric(
            "wait_time_ok_ratio",
            metrics.wait_time_ok_ratio,
            [R2],
            f"Preguntas con al menos {wait_ok_threshold_s:g} s de espera (umbral de Rowe)",
        ),
        RubricMetric(
            "student_response_count",
            metrics.student_response_count,
            [R1],
            "Preguntas que recibieron respuesta",
        ),
        RubricMetric("response_rate", metrics.response_rate, [R1], "Respuestas / preguntas"),
        RubricMetric(
            "teacher_talk_ratio",
            metrics.teacher_talk_ratio,
            [R1],
            "Fracción de la clase en que habla el docente",
            metrics.teacher_talk_ratio_reason,
        ),
    ]


def open_ratio_note(open_ratio: float | None) -> str | None:
    if open_ratio is None:
        return NO_LLM_NOTE
    return None


def empty_audio_metrics() -> list[RubricMetric]:
    names_and_rubrics = [
        ("teacher_question_count", [R2, R3]),
        ("questions_per_minute", [R2, R3]),
        ("rhetorical_question_count", [R2]),
        ("open_question_ratio", [R2]),
        ("mean_wait_time_s", [R2]),
        ("median_wait_time_s", [R2]),
        ("wait_time_ok_ratio", [R2]),
        ("student_response_count", [R1]),
        ("response_rate", [R1]),
        ("teacher_talk_ratio", [R1]),
    ]
    metrics = []
    for name, rubrics in names_and_rubrics:
        metrics.append(RubricMetric(name, None, rubrics, "No disponible", NO_AUDIO_NOTE))
    return metrics


def build_rubric_metrics(
    vision: VisionInterpretation | None,
    audio: AudioInterpretation | None,
    monitoring: MonitoringResult,
    monitoring_threshold: float,
    wait_ok_threshold_s: float,
) -> list[RubricMetric]:
    metrics = video_metrics(vision)
    metrics += monitoring_metrics(monitoring, monitoring_threshold, vision)
    metrics += audio_metrics(audio, wait_ok_threshold_s)
    return metrics
