from espejo.analysis.ai_observations import observations_from
from espejo.analysis.schemas import RubricMetric

WAIT_OK_S = 3.0
MONITORING_THRESHOLD = 0.25


def metric(name: str, value: float | int | None) -> RubricMetric:
    return RubricMetric(name=name, value=value, rubrics=["R2"], description="")


def joined(metrics: list[RubricMetric]) -> str:
    return " ".join(observations_from(metrics, WAIT_OK_S, MONITORING_THRESHOLD))


def test_sin_metricas_no_hay_observaciones():
    assert observations_from([], WAIT_OK_S, MONITORING_THRESHOLD) == []


def test_una_metrica_sin_valor_no_inventa_una_frase():
    assert observations_from([metric("mean_wait_time_s", None)], WAIT_OK_S, MONITORING_THRESHOLD) == []


def test_la_espera_corta_se_menciona_y_la_suficiente_no():
    assert "menos de los 3 s" in joined([metric("mean_wait_time_s", 1.2)])
    assert joined([metric("mean_wait_time_s", 4.0)]) == ""


def test_el_monitoreo_se_menciona_solo_si_queda_bajo_el_umbral():
    assert "por debajo del 25 %" in joined([metric("monitoring_ratio", 0.18)])
    assert joined([metric("monitoring_ratio", 0.40)]) == ""


def test_las_preguntas_cerradas_y_abiertas_dan_frases_distintas():
    cerradas = joined([metric("open_question_ratio", 0.2)])
    abiertas = joined([metric("open_question_ratio", 0.8)])

    assert "no explicar ni razonar" in cerradas
    assert "repreguntas" in abiertas
    assert cerradas != abiertas


def test_las_muletillas_se_mencionan_solo_si_son_varias():
    assert "3 muletillas" in joined([metric("rhetorical_question_count", 3)])
    assert joined([metric("rhetorical_question_count", 1)]) == ""


def test_una_clase_real_produce_varias_observaciones_con_sus_cifras():
    """Las cifras del prompt salen de las métricas, no las elige el modelo."""
    observations = observations_from(
        [
            metric("teacher_question_count", 33),
            metric("mean_wait_time_s", 1.22),
            metric("open_question_ratio", 0.848),
            metric("monitoring_ratio", 0.243),
            metric("rhetorical_question_count", 11),
            metric("response_rate", 0.727),
            metric("board_writing_ratio", 0.013),
        ],
        WAIT_OK_S,
        MONITORING_THRESHOLD,
    )
    texto = " ".join(observations)

    assert len(observations) == 7
    assert "33 preguntas" in texto
    assert "1.2 s" in texto
    assert "24 %" in texto
    assert "11 muletillas" in texto
    assert "73 %" in texto
