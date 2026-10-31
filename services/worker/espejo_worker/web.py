"""Endpoints del worker. Las dependencias (Dict, Volume, spawn) se inyectan: en Modal o en tests."""

import hmac
import logging
import time
import uuid
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status

from espejo_worker.jobs import JobStatus, new_job, public_view
from espejo_worker.results import ReadFile, build_result
from espejo_worker.settings import BYTES_PER_MB, WorkerSecrets, WorkerSettings

logger = logging.getLogger(__name__)

VIDEO_BASENAME = "video"
BEARER_PREFIX = "Bearer "
STATUS_OK = "ok"


@dataclass
class WebDeps:
    store: MutableMapping[str, Any]
    jobs_dir: Path
    spawn: Callable[[str], None]
    commit: Callable[[], None]
    read_file: ReadFile
    settings: WorkerSettings
    secrets: WorkerSecrets
    now: Callable[[], float] = field(default=time.time)


def same_secret(given: str | None, expected: str) -> bool:
    """Comparación en tiempo constante, para no dar pistas sobre el secreto."""
    if not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def bearer_token(authorization: str | None) -> str | None:
    if authorization is None or not authorization.startswith(BEARER_PREFIX):
        return None
    return authorization[len(BEARER_PREFIX) :]


async def save_upload(video: UploadFile, target: Path, settings: WorkerSettings) -> int:
    """Copia por partes y corta si pasa del máximo. Devuelve los bytes escritos."""
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("wb") as output:
        while chunk := await video.read(settings.upload_chunk_bytes()):
            written += len(chunk)
            if written > settings.max_upload_bytes():
                break
            output.write(chunk)
    if written > settings.max_upload_bytes():
        target.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"El video pasa de {settings.max_upload_mb} MB.",
        )
    return written


def video_extension(filename: str | None, settings: WorkerSettings) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Formato no admitido. Usa: {allowed}."
        )
    return extension


def create_app(deps: WebDeps) -> FastAPI:
    app = FastAPI(title="Espejo worker")

    def get_job(job_id: str) -> dict[str, Any]:
        job = deps.store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No existe ese trabajo.")
        return job

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": STATUS_OK}

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        video: Annotated[UploadFile, File()],
        x_access_code: Annotated[str | None, Header()] = None,
        topic: Annotated[str | None, Form()] = None,
        analysis_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        if not same_secret(x_access_code, deps.secrets.upload_access_code):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Código de acceso inválido.")
        extension = video_extension(video.filename, deps.settings)

        job_id = uuid.uuid4().hex
        video_file = VIDEO_BASENAME + extension
        size = await save_upload(video, deps.jobs_dir / job_id / video_file, deps.settings)
        deps.commit()

        job = new_job(job_id, video_file, topic or None, analysis_id or None, deps.now())
        deps.store[job_id] = job
        deps.spawn(job_id)
        logger.info("Trabajo %s en cola (%.1f MB)", job_id, size / BYTES_PER_MB)
        return public_view(job)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        return public_view(get_job(job_id))

    @app.get("/jobs/{job_id}/result")
    def job_result(
        job_id: str, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        if not same_secret(bearer_token(authorization), deps.secrets.worker_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token inválido.")
        job = get_job(job_id)
        if public_view(job)["status"] != JobStatus.DONE:
            raise HTTPException(status.HTTP_409_CONFLICT, public_view(job))
        return build_result(job_id, job["analysis_id"], deps.read_file)

    return app
