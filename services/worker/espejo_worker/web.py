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
from fastapi.middleware.cors import CORSMiddleware

from espejo_worker.jobs import JobStatus, new_job, public_view
from espejo_worker.results import ReadFile, build_result
from espejo_worker.settings import BYTES_PER_MB, WorkerSecrets, WorkerSettings
from espejo_worker.tickets import verify_ticket

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


def upload_job_id(
    ticket: str | None, access_code: str | None, analysis_id: str | None, deps: WebDeps
) -> str:
    """Con permiso de Render, el trabajo usa el id del análisis; con código de acceso, uno nuevo.

    El permiso es de un solo uso: si ya hay un trabajo con ese id, se rechaza.
    """
    if ticket is not None:
        ticket_analysis_id = verify_ticket(ticket, deps.secrets.worker_token, deps.now())
        if ticket_analysis_id is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Permiso inválido o vencido.")
        if analysis_id and analysis_id != ticket_analysis_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "El permiso es de otro análisis.")
        if deps.store.get(ticket_analysis_id) is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Este análisis ya tiene un video.")
        return ticket_analysis_id
    if not same_secret(access_code, deps.secrets.upload_access_code):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Código de acceso inválido.")
    return uuid.uuid4().hex


def create_app(deps: WebDeps) -> FastAPI:
    app = FastAPI(title="Espejo worker")
    # El frontend (Vercel) sube el video directo desde el navegador
    app.add_middleware(
        CORSMiddleware,
        allow_origins=deps.secrets.frontend_origins,
        allow_origin_regex=deps.settings.cors_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        x_upload_ticket: Annotated[str | None, Header()] = None,
        topic: Annotated[str | None, Form()] = None,
        analysis_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        job_id = upload_job_id(x_upload_ticket, x_access_code, analysis_id, deps)
        if x_upload_ticket is not None:
            analysis_id = job_id
        extension = video_extension(video.filename, deps.settings)

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
