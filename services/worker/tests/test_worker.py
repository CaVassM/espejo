"""Endpoints y procesamiento con dependencias simuladas: sin Modal, sin GPU y sin modelos."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from espejo_worker.jobs import Stage, with_stage
from espejo_worker.processing import ContainerInfo, ProcessingDeps, process_job, read_local
from espejo_worker.settings import WorkerSecrets, load_settings
from espejo_worker.tickets import signature
from espejo_worker.web import WebDeps, create_app

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS = load_settings(REPO_ROOT / "configs" / "worker.yaml")
SECRETS = WorkerSecrets(
    worker_token="token-de-prueba",
    upload_access_code="codigo",
    api_base_url=None,
    frontend_origins=[],
)
AUTH = {"Authorization": "Bearer token-de-prueba"}


class Clock:
    def __init__(self) -> None:
        self.now_s = 1_000.0

    def __call__(self) -> float:
        self.now_s += 10.0
        return self.now_s


class FakePipeline:
    """Escribe resultados de mentira, o falla si se le pide."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def run(
        self, video_path: Path, run_dir: Path, topic: str | None, on_stage: Callable[[Stage], None]
    ) -> None:
        assert video_path.exists()
        on_stage(Stage.VIDEO)
        if self.fail:
            raise RuntimeError("video dañado")
        (run_dir / "detections.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
        on_stage(Stage.AUDIO)
        segments = [{"text": "[ALUMNO responde]"}]
        audio = {"segments": segments, "topic": topic}
        (run_dir / "audio_raw.json").write_text(json.dumps(audio), encoding="utf-8")
        (run_dir / "heatmap.png").write_bytes(b"\x89PNG")


class World:
    """El Dict, el Volume y el spawn de Modal, simulados en memoria y en una carpeta temporal."""

    def __init__(self, tmp_path: Path, pipeline: FakePipeline) -> None:
        self.store: dict = {}
        self.jobs_dir = tmp_path / "jobs"
        self.clock = Clock()
        self.spawned: list[str] = []
        self.processing = ProcessingDeps(
            store=self.store,
            jobs_dir=self.jobs_dir,
            pipeline=pipeline,
            reload=lambda: None,
            commit=lambda: None,
            settings=SETTINGS,
            secrets=SECRETS,
            now=self.clock,
        )
        web = WebDeps(
            store=self.store,
            jobs_dir=self.jobs_dir,
            spawn=self.spawned.append,
            commit=lambda: None,
            read_file=read_local(self.jobs_dir),
            settings=SETTINGS,
            secrets=SECRETS,
            now=self.clock,
        )
        self.client = TestClient(create_app(web))

    def upload(self, filename: str = "clase.mp4", code: str = "codigo") -> dict:
        files = {"video": (filename, b"video de prueba", "video/mp4")}
        response = self.client.post(
            "/jobs", files=files, data={"topic": "Cuento"}, headers={"X-Access-Code": code}
        )
        return {"status_code": response.status_code, "body": response.json()}

    def run_spawned(self) -> None:
        for job_id in self.spawned:
            process_job(job_id, self.processing, ContainerInfo(cold_start=True, models_load_s=42.0))


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path, FakePipeline())


def test_health(world: World) -> None:
    assert world.client.get("/health").json() == {"status": "ok"}


def test_upload_needs_the_access_code(world: World) -> None:
    assert world.upload(code="otro")["status_code"] == 401
    assert world.spawned == []


def test_upload_rejects_other_formats(world: World) -> None:
    assert world.upload(filename="notas.txt")["status_code"] == 415


def test_job_goes_from_queue_to_done_and_deletes_the_video(world: World) -> None:
    created = world.upload()
    job_id = created["body"]["job_id"]
    assert created["status_code"] == 202
    assert created["body"]["stage"] == "En cola"

    world.run_spawned()

    view = world.client.get(f"/jobs/{job_id}").json()
    assert view["status"] == "done"
    assert view["stage"] == "Listo"
    assert view["timings"]["cold_start"] is True
    assert "video_s" in view["timings"] and "audio_s" in view["timings"]
    assert not (world.jobs_dir / job_id / "video.mp4").exists()


def test_result_needs_the_worker_token(world: World) -> None:
    job_id = world.upload()["body"]["job_id"]
    world.run_spawned()

    assert world.client.get(f"/jobs/{job_id}/result").status_code == 401
    result = world.client.get(f"/jobs/{job_id}/result", headers=AUTH).json()
    assert result["detections"] == {"frames": []}
    assert result["audio_raw"]["segments"][0]["text"] == "[ALUMNO responde]"
    assert result["heatmap_png_base64"] is not None


def test_result_before_finishing_is_a_conflict(world: World) -> None:
    job_id = world.upload()["body"]["job_id"]
    response = world.client.get(f"/jobs/{job_id}/result", headers=AUTH)
    assert response.status_code == 409


def test_failure_is_reported_and_the_video_is_deleted(tmp_path: Path) -> None:
    world = World(tmp_path, FakePipeline(fail=True))
    job_id = world.upload()["body"]["job_id"]

    world.run_spawned()

    view = world.client.get(f"/jobs/{job_id}").json()
    assert view["status"] == "error"
    assert view["stage"] == "Error"
    assert "RuntimeError" in view["error"]
    assert not (world.jobs_dir / job_id / "video.mp4").exists()


def test_unknown_job_is_not_found(world: World) -> None:
    assert world.client.get("/jobs/no-existe").status_code == 404


def test_stage_timings() -> None:
    job = {
        "stage": "queued",
        "stage_started_at_s": 0.0,
        "created_at_s": 0.0,
        "timings": {},
    }
    job = with_stage(job, Stage.VIDEO, 5.0)
    job = with_stage(job, Stage.AUDIO, 65.0)
    job = with_stage(job, Stage.DONE, 95.0)
    assert job["timings"] == {"queued_s": 5.0, "video_s": 60.0, "audio_s": 30.0, "total_s": 95.0}


def ticket_for(analysis_id: str, expires_at_s: int, secret: str = "token-de-prueba") -> str:
    return f"{analysis_id}.{expires_at_s}.{signature(analysis_id, expires_at_s, secret)}"


def upload_with_ticket(world: World, ticket: str, analysis_id: str = "abc123") -> object:
    files = {"video": ("clase.mp4", b"video de prueba", "video/mp4")}
    return world.client.post(
        "/jobs",
        files=files,
        data={"analysis_id": analysis_id},
        headers={"X-Upload-Ticket": ticket},
    )


def test_ticket_from_render_uses_the_analysis_id_as_job_id(world: World) -> None:
    response = upload_with_ticket(world, ticket_for("abc123", 9_999_999_999))

    assert response.status_code == 202
    assert response.json()["job_id"] == "abc123"
    assert response.json()["analysis_id"] == "abc123"
    assert world.spawned == ["abc123"]


def test_ticket_is_single_use(world: World) -> None:
    ticket = ticket_for("abc123", 9_999_999_999)
    upload_with_ticket(world, ticket)
    assert upload_with_ticket(world, ticket).status_code == 409


def test_expired_forged_or_foreign_tickets_are_rejected(world: World) -> None:
    expired = ticket_for("abc123", 5)
    forged = ticket_for("abc123", 9_999_999_999, secret="otro-secreto")
    foreign = ticket_for("otro", 9_999_999_999)

    assert upload_with_ticket(world, expired).status_code == 401
    assert upload_with_ticket(world, forged).status_code == 401
    assert upload_with_ticket(world, foreign, analysis_id="abc123").status_code == 401
    assert world.spawned == []


def test_cors_lets_vercel_upload(world: World) -> None:
    headers = {
        "Origin": "https://espejo.vercel.app",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-upload-ticket",
    }
    response = world.client.options("/jobs", headers=headers)
    assert response.headers["access-control-allow-origin"] == "https://espejo.vercel.app"
