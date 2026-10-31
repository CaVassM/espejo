"""Servidor MCP del docente por HTTP, como lo usa un asistente: claves, formato y tope."""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_api import WORKER_AUTH, World, raw_payload

from app.mcp_server import RedactTeacherKeyFilter
from app.services.assistant import load_assistant_settings
from app.settings import API_CONFIG_PATH

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
LIMITS = load_assistant_settings(API_CONFIG_PATH)
CLASS_45_MIN_S = 45 * 60
LONG_QUOTE = "y entonces, ¿qué creen ustedes que pasará con los hijos del labrador cuando " * 2


def citation(minute_s: int, label: str, quote: str) -> dict[str, Any]:
    return {
        "timestamp": f"{minute_s // 60:02d}:{minute_s % 60:02d}",
        "label": label,
        "quote": quote,
        "reason": "pide predecir lo que pasará en el cuento",
    }


def make_report(citation_count: int, duration_s: float) -> dict[str, Any]:
    """Un class_report con el formato real; con muchas citas simula una clase de 45 min."""
    citations = []
    llm_items = []
    for index in range(citation_count):
        item = citation(index * 7, "question_open", f"¿Pregunta {index}? {LONG_QUOTE}")
        citations.append(item)
        confidence = "alta" if index % 3 == 0 else "media"
        llm_items.append({**item, "confidence": confidence})
    rubrics = []
    for rubric in ["R1", "R2", "R3", "R4", "R5"]:
        rubrics.append(
            {
                "rubric": rubric,
                "title": f"Rúbrica {rubric}",
                "status": "con evidencia",
                "metrics": [],
                "citations": citations if rubric == "R2" else [],
            }
        )
    return {
        "metadata": {"duration_s": duration_s},
        "metrics": {
            "teacher_question_count": {"value": 15},
            "questions_per_minute": {"value": 3.0},
            "open_question_ratio": {"value": 0.4},
            "mean_wait_time_s": {"value": 1.2},
            "response_rate": {"value": 0.93},
            "monitoring_ratio": {"value": 0.2},
            "teacher_talk_ratio": {"value": 0.57},
        },
        "rubric_evidence": rubrics,
        "llm_analysis": {
            "question_labels": llm_items,
            "feedback": [{"label": "feedback_elemental"}],
        },
        "one_suggestion": {
            "timestamp": "01:22",
            "what_we_saw": "Esperaste 1,2 s.",
            "why_it_matters": "Pensar toma tiempo.",
            "try_this": "Cuenta hasta tres.",
            "example_phrase": "Piénsalo un momento.",
        },
    }


def make_audio_raw(duration_s: float) -> dict[str, Any]:
    """Segmentos de 2 s alternando docente y alumno, ya anonimizados como en el worker."""
    segments = []
    turns = []
    start_s = 0.0
    index = 0
    while start_s < duration_s:
        teacher = index % 2 == 0
        text = f"Bien, [NOMBRE], {LONG_QUOTE}" if teacher else "[ALUMNO responde]"
        segments.append(
            {
                "start_s": start_s,
                "end_s": start_s + 2.0,
                "text": text,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.05,
                "words": [],
            }
        )
        # El docente habla más tiempo que el alumno: es el que la regla elige como docente
        end_s = start_s + (2.0 if teacher else 1.5)
        speaker = "SPEAKER_00" if teacher else "SPEAKER_01"
        turns.append({"speaker": speaker, "start_s": start_s, "end_s": end_s})
        start_s += 2.0
        index += 1
    return {"segments": segments, "diarization": {"used": True, "note": "prueba", "turns": turns}}


class Teacher:
    """Un docente con su clave y un análisis ya terminado."""

    def __init__(self, world: World, reuse_key: str | None = None, duration_s: float = 300) -> None:
        body: dict[str, Any] = {"access_code": "codigo", "topic": "Cuento"}
        if reuse_key:
            body["teacher_key"] = reuse_key
        created = world.client.post("/api/analyses", json=body).json()
        self.key = created["teacher_key"]
        self.analysis_id = created["analysis_id"]
        self.assistant_url = created["assistant_url"]
        payload = raw_payload(self.analysis_id)
        payload["audio_raw"] = make_audio_raw(duration_s)
        world.client.post(
            f"/api/analyses/{self.analysis_id}/raw", json=payload, headers=WORKER_AUTH
        )


def call(client: TestClient, key: str, name: str, arguments: dict[str, Any]) -> str:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = client.post(f"/mcp/{key}", headers=MCP_HEADERS, json=request)
    assert response.status_code == 200
    return response.json()["result"]["content"][0]["text"]


def rpc(client: TestClient, key: str, method: str, params: dict[str, Any]) -> Any:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return client.post(f"/mcp/{key}", headers=MCP_HEADERS, json=request).json()["result"]


def plan_text(client: TestClient, key: str, analysis_id: str) -> str:
    params = {"name": "planificar_proxima_clase", "arguments": {"id": analysis_id}}
    return rpc(client, key, "prompts/get", params)["messages"][0]["content"]["text"]


@pytest.fixture
def world() -> World:
    return World(report=make_report(citation_count=12, duration_s=300))


def test_creating_returns_the_key_once_with_the_assistant_url(world: World) -> None:
    teacher = Teacher(world)
    assert len(teacher.key) >= 32
    assert teacher.assistant_url.endswith(f"/mcp/{teacher.key}")
    view = world.client.get(f"/api/analyses/{teacher.analysis_id}").json()
    assert "teacher_key" not in view


def test_reusing_a_key_groups_classes_and_the_list_shows_the_10_newest(world: World) -> None:
    teacher = Teacher(world)
    for _ in range(11):
        Teacher(world, reuse_key=teacher.key)
    body = {"access_code": "codigo", "teacher_key": "x" * 40}
    assert world.client.post("/api/analyses", json=body).status_code == 401

    with TestClient(world.client.app) as client:
        text = call(client, teacher.key, "listar_mis_clases", {})
    rows = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(rows) == LIMITS.max_classes
    assert "Cuento | 5 min" in rows[0]


def test_a_key_does_not_see_another_teachers_classes(world: World) -> None:
    ana = Teacher(world)
    beto = Teacher(world)
    with TestClient(world.client.app) as client:
        listing = call(client, ana.key, "listar_mis_clases", {})
        stolen = call(client, ana.key, "ver_reporte", {"id": beto.analysis_id})
        fragment = call(client, ana.key, "ver_fragmento", {"id": beto.analysis_id, "minuto": "1"})
    assert ana.analysis_id in listing
    assert beto.analysis_id not in listing
    assert "No existe ese análisis entre tus clases." in stolen
    assert "No existe ese análisis entre tus clases." in fragment


def test_an_invalid_key_sees_nothing(world: World) -> None:
    teacher = Teacher(world)
    with TestClient(world.client.app) as client:
        for key in ["clave-inventada-" + "0" * 30, "corta"]:
            assert call(client, key, "listar_mis_clases", {}) == "No hay clases para esta clave."
            denied = call(client, key, "ver_reporte", {"id": teacher.analysis_id})
            assert "No existe ese análisis" in denied


def test_each_tool_answers_short_and_readable(world: World) -> None:
    teacher = Teacher(world)
    other = Teacher(world, reuse_key=teacher.key)
    report_id = {"id": teacher.analysis_id}
    with TestClient(world.client.app) as client:
        tools = rpc(client, teacher.key, "tools/list", {})["tools"]
        report = call(client, teacher.key, "ver_reporte", report_id)
        page_1 = call(client, teacher.key, "ver_evidencias", {**report_id, "rubrica": "r2"})
        page_2 = call(
            client, teacher.key, "ver_evidencias", {**report_id, "rubrica": "R2", "pagina": 2}
        )
        comparison = call(
            client,
            teacher.key,
            "comparar_clases",
            {"id_a": teacher.analysis_id, "id_b": other.analysis_id},
        )
        fragment = call(client, teacher.key, "ver_fragmento", {**report_id, "minuto": "02:00"})
        plan = plan_text(client, teacher.key, teacher.analysis_id)

    names = {tool["name"] for tool in tools}
    assert names == {
        "listar_mis_clases",
        "ver_reporte",
        "ver_evidencias",
        "comparar_clases",
        "ver_fragmento",
    }
    for tool in tools:
        assert tool["annotations"]["readOnlyHint"] is True

    # Resumen: métricas clave, consejo y número de citas por rúbrica, sin las citas
    assert "Espera promedio después de preguntar: 1,2 s" in report
    assert "Prueba esto: Cuenta hasta tres." in report
    assert "R2 Rúbrica R2: con evidencia, 12 citas" in report
    assert "¿Pregunta 0?" not in report

    # Evidencias: 8 por página, primero las de confianza alta, y aviso de la siguiente página
    cited = [line for line in page_1.splitlines() if line.startswith("- [")]
    assert len(cited) == LIMITS.evidence_page_size
    assert all("confianza alta" in line for line in cited[:4])
    assert page_1.endswith("Hay 4 más. Pide pagina=2.")
    assert "Pide pagina" not in page_2

    # Las dos clases tienen el mismo reporte: no hay cambios
    assert "Sin cambios" in comparison

    # Fragmento: solo ±60 s alrededor del minuto, ya anonimizado
    assert "de 01:00 a 03:00" in fragment
    assert "[04:00]" not in fragment
    assert "ALUMNO: [ALUMNO responde]" in fragment
    assert "DOCENTE: Bien, [NOMBRE]" in fragment
    assert "No califiques" in plan


def test_comparison_lists_only_what_changed() -> None:
    world = World(report=make_report(citation_count=2, duration_s=300))
    first = Teacher(world)
    changed = make_report(citation_count=2, duration_s=300)
    changed["metrics"]["mean_wait_time_s"]["value"] = 2.8
    changed["llm_analysis"]["feedback"].append({"label": "feedback_descriptive"})
    world.report = changed
    second = Teacher(world, reuse_key=first.key)
    with TestClient(world.client.app) as client:
        text = call(
            client,
            first.key,
            "comparar_clases",
            {"id_a": first.analysis_id, "id_b": second.analysis_id},
        )
    assert "- Espera promedio: 1,2 s → 2,8 s" in text
    assert "- Retroalimentación descriptiva: 0 → 1" in text
    assert "Preguntas abiertas" not in text


def test_no_tool_passes_the_limit_even_with_a_45_minute_class() -> None:
    world = World(report=make_report(citation_count=300, duration_s=CLASS_45_MIN_S))
    teacher = Teacher(world, duration_s=CLASS_45_MIN_S)
    other = Teacher(world, reuse_key=teacher.key, duration_s=CLASS_45_MIN_S)
    calls = [
        ("listar_mis_clases", {}),
        ("ver_reporte", {"id": teacher.analysis_id}),
        ("ver_evidencias", {"id": teacher.analysis_id, "rubrica": "R2"}),
        ("comparar_clases", {"id_a": teacher.analysis_id, "id_b": other.analysis_id}),
        ("ver_fragmento", {"id": teacher.analysis_id, "minuto": "20:00"}),
    ]
    with TestClient(world.client.app) as client:
        answers = {}
        for name, arguments in calls:
            answers[name] = call(client, teacher.key, name, arguments)
        answers["planificar_proxima_clase"] = plan_text(client, teacher.key, teacher.analysis_id)
    for name, text in answers.items():
        assert len(text) <= LIMITS.max_response_chars, name
    # El fragmento de ±60 s con mucho texto se recorta y lo avisa
    assert "respuesta recortada" in answers["ver_fragmento"]


def test_the_key_never_reaches_the_logs() -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s"',
        ("1.2.3.4", "POST", "/mcp/abcDEF123_-xyz"),
        None,
    )
    RedactTeacherKeyFilter().filter(record)
    assert "abcDEF123" not in record.getMessage()
    assert "/mcp/[clave]" in record.getMessage()
