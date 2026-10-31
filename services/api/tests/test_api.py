"""La API con base de datos en memoria, worker simulado e interpretación simulada."""

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.repositories.analyses import AnalysisRepository, create_db_engine, create_tables
from app.services.analyses import AnalysisService
from app.services.tickets import sign_ticket
from app.settings import ApiSettings, sqlalchemy_url

SETTINGS = ApiSettings(
    database_url="sqlite://",
    worker_url="https://worker.example",
    worker_token="token-de-prueba",
    upload_access_code="codigo",
    upload_ticket_ttl_s=3600,
    cors_origin_regex=r"https://.*\.vercel\.app",
    frontend_origins=[],
    worker_timeout_s=10,
)
WORKER_AUTH = {"Authorization": "Bearer token-de-prueba"}
REPORT = {"metrics": {"teacher_question_count": {"value": 30}}}


class FakeWorker:
    """Responde como el worker de Modal; jobs se cambia desde el test."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.result_calls = 0

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def job_result(self, job_id: str) -> dict[str, Any]:
        self.result_calls += 1
        return raw_payload(job_id)


def raw_payload(analysis_id: str) -> dict[str, Any]:
    return {
        "job_id": analysis_id,
        "analysis_id": analysis_id,
        "detections": {"frames": []},
        "audio_raw": {"segments": [{"text": "[ALUMNO responde]"}]},
        "heatmap_png_base64": base64.b64encode(b"\x89PNG").decode("ascii"),
    }


class World:
    def __init__(
        self, fail_interpretation: bool = False, report: dict[str, Any] | None = None
    ) -> None:
        engine = create_db_engine(SETTINGS.database_url)
        create_tables(engine)
        repository = AnalysisRepository(sessionmaker(engine, expire_on_commit=False))
        self.worker = FakeWorker()
        self.interpreted: list[Any] = []
        # El reporte que "arma" la interpretación simulada (se puede cambiar entre análisis)
        self.report = report or REPORT

        def interpreter(detections: Any, audio_raw: Any) -> dict[str, Any]:
            if fail_interpretation:
                raise RuntimeError("sin LLM")
            self.interpreted.append((detections, audio_raw))
            return self.report

        service = AnalysisService(
            repository, self.worker, interpreter, SETTINGS, now=lambda: 1_000.0
        )
        self.client = TestClient(create_app(SETTINGS, service))

    def create(self, code: str = "codigo") -> Any:
        return self.client.post("/api/analyses", json={"access_code": code, "topic": "Cuento"})


@pytest.fixture
def world() -> World:
    return World()


def test_health(world: World) -> None:
    assert world.client.get("/api/health").json() == {"status": "ok"}


def test_create_needs_the_access_code(world: World) -> None:
    assert world.create(code="otro").status_code == 401


def test_create_returns_a_signed_upload_ticket(world: World) -> None:
    created = world.create().json()
    analysis_id = created["analysis_id"]

    assert created["status"] == "waiting_upload"
    assert created["stage"] == "Esperando el video"
    assert created["upload"]["url"] == "https://worker.example/jobs"
    assert created["upload"]["fields"] == {"analysis_id": analysis_id, "topic": "Cuento"}
    expected = sign_ticket(analysis_id, 1_000 + 3600, "token-de-prueba")
    assert created["upload"]["ticket"] == expected


def test_status_follows_the_worker_and_pulls_the_result(world: World) -> None:
    analysis_id = world.create().json()["analysis_id"]

    world.worker.jobs[analysis_id] = {"status": "processing", "stage": "Mirando el video"}
    view = world.client.get(f"/api/analyses/{analysis_id}").json()
    assert (view["status"], view["stage"]) == ("processing", "Mirando el video")

    # Render estaba dormido y no recibió el aviso: lo va a buscar al consultar
    world.worker.jobs[analysis_id] = {"status": "done", "stage": "Listo"}
    world.client.get(f"/api/analyses/{analysis_id}")

    view = world.client.get(f"/api/analyses/{analysis_id}").json()
    assert view["status"] == "done"
    assert view["report_ready"] is True
    assert world.client.get(f"/api/analyses/{analysis_id}/report").json() == REPORT
    heatmap = world.client.get(f"/api/analyses/{analysis_id}/heatmap.png")
    assert heatmap.headers["content-type"] == "image/png"
    assert len(world.interpreted) == 1


def test_worker_callback_needs_the_token_and_interprets_once(world: World) -> None:
    analysis_id = world.create().json()["analysis_id"]
    url = f"/api/analyses/{analysis_id}/raw"

    assert world.client.post(url, json=raw_payload(analysis_id)).status_code == 401
    first = world.client.post(url, json=raw_payload(analysis_id), headers=WORKER_AUTH)
    second = world.client.post(url, json=raw_payload(analysis_id), headers=WORKER_AUTH)

    assert first.status_code == 202
    assert second.json()["status"] == "done"
    assert len(world.interpreted) == 1


def test_worker_error_is_shown(world: World) -> None:
    analysis_id = world.create().json()["analysis_id"]
    error = "No se pudo procesar el video (ValueError)."
    world.worker.jobs[analysis_id] = {"status": "error", "stage": "Error", "error": error}

    view = world.client.get(f"/api/analyses/{analysis_id}").json()

    assert view["status"] == "error"
    assert view["error"] == error


def test_failed_interpretation_ends_in_error() -> None:
    world = World(fail_interpretation=True)
    analysis_id = world.create().json()["analysis_id"]
    world.client.post(
        f"/api/analyses/{analysis_id}/raw", json=raw_payload(analysis_id), headers=WORKER_AUTH
    )

    view = world.client.get(f"/api/analyses/{analysis_id}").json()

    assert view["status"] == "error"
    assert "RuntimeError" in view["error"]


def test_report_before_done_is_a_conflict(world: World) -> None:
    analysis_id = world.create().json()["analysis_id"]
    assert world.client.get(f"/api/analyses/{analysis_id}/report").status_code == 409


def test_detections_are_served_once_the_worker_termina(world: World) -> None:
    """Las cajas del video, para dibujarlas sobre el reproductor."""
    analysis_id = world.create().json()["analysis_id"]

    # Antes de que el worker entregue nada no hay cajas que servir.
    assert world.client.get(f"/api/analyses/{analysis_id}/detections").status_code == 409

    world.client.post(
        f"/api/analyses/{analysis_id}/raw", json=raw_payload(analysis_id), headers=WORKER_AUTH
    )

    response = world.client.get(f"/api/analyses/{analysis_id}/detections")
    assert response.status_code == 200
    assert response.json() == raw_payload(analysis_id)["detections"]


def test_detections_of_an_unknown_analysis_are_not_found(world: World) -> None:
    assert world.client.get("/api/analyses/no-existe/detections").status_code == 404


def test_unknown_analysis_is_not_found(world: World) -> None:
    assert world.client.get("/api/analyses/no-existe").status_code == 404


def test_cors_allows_vercel_previews(world: World) -> None:
    headers = {"Origin": "https://espejo.vercel.app", "Access-Control-Request-Method": "POST"}
    response = world.client.options("/api/analyses", headers=headers)
    assert response.headers["access-control-allow-origin"] == "https://espejo.vercel.app"


def test_render_database_url_uses_psycopg() -> None:
    assert sqlalchemy_url("postgresql://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert sqlalchemy_url("sqlite://") == "sqlite://"
