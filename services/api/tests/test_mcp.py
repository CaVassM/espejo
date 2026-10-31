"""Servidor MCP del docente, por HTTP como lo usa un asistente: claves, aislamiento y formatos."""

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_api import WORKER_AUTH, World, raw_payload

from app.mcp_server import RedactTeacherKeyFilter

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
REPORT = {
    "metadata": {"duration_s": 300.0},
    "metrics": {
        "teacher_question_count": {"value": 15},
        "open_question_ratio": {"value": 0.4},
        "mean_wait_time_s": {"value": 1.2},
        "monitoring_ratio": {"value": 0.2},
        "board_writing_ratio": {"value": None, "experimental": True},
    },
    "rubric_evidence": [
        {
            "rubric": "R2",
            "title": "Promueve el razonamiento",
            "status": "con evidencia",
            "metrics": [{"name": "open_question_ratio", "value": 0.4, "experimental": False}],
            "citations": [
                {
                    "timestamp": "01:22",
                    "label": "question_open",
                    "quote": "¿De qué tratará?",
                    "reason": "pide predecir",
                }
            ],
        }
    ],
    "llm_analysis": {"feedback": [{"label": "feedback_elemental"}]},
    "one_suggestion": {
        "timestamp": "01:22",
        "what_we_saw": "Esperaste 1,2 s.",
        "why_it_matters": "Pensar toma tiempo.",
        "try_this": "Cuenta hasta tres.",
        "example_phrase": "Piénsalo un momento.",
    },
    # Lo que nunca debe salir por MCP
    "audio_events": [{"text": "pregunta"}],
}


class Teacher:
    """Un docente con su clave y sus análisis ya terminados."""

    def __init__(self, world: World, reuse_key: str | None = None) -> None:
        body: dict[str, Any] = {"access_code": "codigo", "topic": "Cuento"}
        if reuse_key:
            body["teacher_key"] = reuse_key
        created = world.client.post("/api/analyses", json=body).json()
        self.key = created["teacher_key"]
        self.analysis_id = created["analysis_id"]
        self.assistant_url = created["assistant_url"]
        world.client.post(
            f"/api/analyses/{self.analysis_id}/raw",
            json=raw_payload(self.analysis_id),
            headers=WORKER_AUTH,
        )


@pytest.fixture
def world() -> World:
    return World(report=REPORT)


def call_tool(client: TestClient, key: str, name: str, arguments: dict[str, Any]) -> Any:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = client.post(f"/mcp/{key}", headers=MCP_HEADERS, json=request)
    assert response.status_code == 200
    result = response.json()["result"]
    if result.get("isError"):
        return {"error": result["content"][0]["text"]}
    if "structuredContent" in result:
        return result["structuredContent"]
    return json.loads(result["content"][0]["text"])


def rpc(client: TestClient, key: str, method: str, params: dict[str, Any]) -> Any:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return client.post(f"/mcp/{key}", headers=MCP_HEADERS, json=request).json()["result"]


def test_creating_returns_the_key_once_with_the_assistant_url(world: World) -> None:
    teacher = Teacher(world)
    assert len(teacher.key) >= 32
    assert teacher.assistant_url.endswith(f"/mcp/{teacher.key}")
    view = world.client.get(f"/api/analyses/{teacher.analysis_id}").json()
    assert "teacher_key" not in view


def test_reusing_a_key_groups_classes_and_a_made_up_key_is_rejected(world: World) -> None:
    teacher = Teacher(world)
    Teacher(world, reuse_key=teacher.key)
    body = {"access_code": "codigo", "teacher_key": "x" * 40}
    assert world.client.post("/api/analyses", json=body).status_code == 401

    with TestClient(world.client.app) as client:
        classes = call_tool(client, teacher.key, "listar_mis_clases", {})["result"]
    assert len(classes) == 2
    assert set(classes[0]) == {"id", "fecha", "tema", "duracion_min", "estado"}
    assert classes[0]["duracion_min"] == 5.0


def test_a_key_does_not_see_another_teachers_classes(world: World) -> None:
    ana = Teacher(world)
    beto = Teacher(world)
    with TestClient(world.client.app) as client:
        anas = call_tool(client, ana.key, "listar_mis_clases", {})["result"]
        stolen = call_tool(client, ana.key, "ver_reporte", {"id": beto.analysis_id})
    assert [item["id"] for item in anas] == [ana.analysis_id]
    assert "No existe ese análisis entre tus clases." in stolen["error"]


def test_an_invalid_key_sees_nothing(world: World) -> None:
    teacher = Teacher(world)
    with TestClient(world.client.app) as client:
        for key in ["clave-inventada-" + "0" * 30, "corta"]:
            assert call_tool(client, key, "listar_mis_clases", {}) == {"result": []}
            denied = call_tool(client, key, "ver_reporte", {"id": teacher.analysis_id})
            assert "error" in denied


def test_each_tool_returns_its_format_and_nothing_raw(world: World) -> None:
    teacher = Teacher(world)
    other = Teacher(world, reuse_key=teacher.key)
    with TestClient(world.client.app) as client:
        tools = rpc(client, teacher.key, "tools/list", {})["tools"]
        report = call_tool(client, teacher.key, "ver_reporte", {"id": teacher.analysis_id})
        evidence = call_tool(
            client, teacher.key, "ver_evidencias", {"id": teacher.analysis_id, "rubrica": "r2"}
        )
        comparison = call_tool(
            client,
            teacher.key,
            "comparar_clases",
            {"id_a": teacher.analysis_id, "id_b": other.analysis_id},
        )
        prompt = rpc(
            client,
            teacher.key,
            "prompts/get",
            {"name": "planificar_proxima_clase", "arguments": {"id": teacher.analysis_id}},
        )

    names = {tool["name"] for tool in tools}
    assert names == {"listar_mis_clases", "ver_reporte", "ver_evidencias", "comparar_clases"}
    for tool in tools:
        assert tool["annotations"]["readOnlyHint"] is True

    assert set(report) == {"id", "ficha", "metricas_principales", "retroalimentaciones", "consejo"}
    assert report["consejo"]["try_this"] == "Cuenta hasta tres."
    assert "board_writing_ratio" not in report["metricas_principales"]
    assert evidence["citas"] == [
        {
            "minuto": "01:22",
            "tipo": "question_open",
            "cita": "¿De qué tratará?",
            "por_que": "pide predecir",
        }
    ]
    assert comparison["diferencia_b_menos_a"]["mean_wait_time_s"] == 0.0
    assert comparison["clase_a"]["retroalimentaciones"] == {"elemental": 1}
    text = prompt["messages"][0]["content"]["text"]
    assert "No califiques" in text

    everything = json.dumps([report, evidence, comparison, prompt], ensure_ascii=False)
    assert "audio_events" not in everything
    assert "detections" not in everything


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
