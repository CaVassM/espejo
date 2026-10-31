"""Traduce las métricas de la clase a frases que el LLM usa para recomendar uso de IA.

Vive en analysis/ y no en llm/ porque es la capa que conoce las métricas: el módulo del LLM
solo recibe una lista de frases ya escritas, y así no queda dependiendo del reporte.
"""

from espejo.analysis.schemas import RubricMetric

LOW_OPEN_RATIO = 0.5
MANY_RHETORICAL = 3


def metric_value(metrics: list[RubricMetric], name: str) -> float | int | None:
    for metric in metrics:
        if metric.name == name and isinstance(metric.value, int | float):
            return metric.value
    return None


def observations_from(metrics: list[RubricMetric], wait_ok_s: float, threshold: float) -> list[str]:
    """Frases en lenguaje simple sobre lo que pasó en la clase, sin números crudos sueltos."""
    observations = []

    questions = metric_value(metrics, "teacher_question_count")
    if questions is not None:
        observations.append(f"El docente hizo {int(questions)} preguntas en la clase.")

    wait = metric_value(metrics, "mean_wait_time_s")
    if wait is not None and wait < wait_ok_s:
        observations.append(
            f"Esperó {wait:.1f} s en promedio después de preguntar, menos de los "
            f"{wait_ok_s:.0f} s que dan tiempo a pensar."
        )

    open_ratio = metric_value(metrics, "open_question_ratio")
    if open_ratio is not None:
        if open_ratio < LOW_OPEN_RATIO:
            observations.append(
                "La mayoría de sus preguntas pedían un dato o un sí/no, no explicar ni razonar."
            )
        else:
            observations.append(
                "Buena parte de sus preguntas ya pedían explicar o razonar; conviene sostenerlas "
                "con repreguntas."
            )

    monitoring = metric_value(metrics, "monitoring_ratio")
    if monitoring is not None and monitoring < threshold:
        observations.append(
            f"Dedicó {monitoring * 100:.0f} % de la clase a recoger evidencia de lo que "
            f"entendían los estudiantes, por debajo del {threshold * 100:.0f} % que pide el manual."
        )

    rhetorical = metric_value(metrics, "rhetorical_question_count")
    if rhetorical is not None and rhetorical >= MANY_RHETORICAL:
        observations.append(
            f'Usó {int(rhetorical)} muletillas del tipo "¿no?" o "¿ya?" que no esperan respuesta.'
        )

    response_rate = metric_value(metrics, "response_rate")
    if response_rate is not None:
        observations.append(f"Respondieron {response_rate * 100:.0f} % de sus preguntas.")

    board = metric_value(metrics, "board_writing_ratio")
    if board is not None:
        observations.append(
            f"Estuvo de espaldas escribiendo en la pizarra {board * 100:.0f} % del tiempo."
        )
    return observations
