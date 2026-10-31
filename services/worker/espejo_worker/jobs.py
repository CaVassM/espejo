"""Estado de un trabajo: etapas en español simple, tiempos y vencimiento de los resultados."""

import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

SECONDS_PER_DAY = 24 * 60 * 60
TIME_DECIMALS = 1


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class Stage(StrEnum):
    QUEUED = "queued"
    VIDEO = "video"
    AUDIO = "audio"
    DONE = "done"
    ERROR = "error"


STAGE_LABELS = {
    Stage.QUEUED: "En cola",
    Stage.VIDEO: "Mirando el video",
    Stage.AUDIO: "Escuchando el audio",
    Stage.DONE: "Listo",
    Stage.ERROR: "Error",
}
STATUS_BY_STAGE = {
    Stage.QUEUED: JobStatus.QUEUED,
    Stage.VIDEO: JobStatus.PROCESSING,
    Stage.AUDIO: JobStatus.PROCESSING,
    Stage.DONE: JobStatus.DONE,
    Stage.ERROR: JobStatus.ERROR,
}


def iso_time(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=UTC).isoformat(timespec="seconds")


def new_job(
    job_id: str, video_file: str, topic: str | None, analysis_id: str | None, now_s: float
) -> dict[str, Any]:
    """Solo tipos simples: el estado vive en un Modal Dict."""
    return {
        "job_id": job_id,
        "analysis_id": analysis_id,
        "video_file": video_file,
        "topic": topic,
        "stage": Stage.QUEUED.value,
        "error": None,
        "created_at_s": now_s,
        "stage_started_at_s": now_s,
        "updated_at_s": now_s,
        "timings": {},
        "callback": None,
    }


def with_stage(
    job: dict[str, Any], stage: Stage, now_s: float, error: str | None = None
) -> dict[str, Any]:
    """Pasa a otra etapa y guarda cuánto duró la anterior (p. ej. timings["video_s"])."""
    updated = dict(job)
    timings = dict(job["timings"])
    previous = job["stage"]
    timings[f"{previous}_s"] = round(now_s - job["stage_started_at_s"], TIME_DECIMALS)
    if stage in (Stage.DONE, Stage.ERROR):
        timings["total_s"] = round(now_s - job["created_at_s"], TIME_DECIMALS)
    updated.update(
        stage=stage.value,
        error=error,
        stage_started_at_s=now_s,
        updated_at_s=now_s,
        timings=timings,
    )
    return updated


def public_view(job: dict[str, Any]) -> dict[str, Any]:
    """Lo que ve quien consulta el estado: sin rutas ni detalles internos."""
    stage = Stage(job["stage"])
    return {
        "job_id": job["job_id"],
        "analysis_id": job["analysis_id"],
        "status": STATUS_BY_STAGE[stage].value,
        "stage": STAGE_LABELS[stage],
        "created_at": iso_time(job["created_at_s"]),
        "updated_at": iso_time(job["updated_at_s"]),
        "error": job["error"],
        "timings": job["timings"],
        "callback": job["callback"],
    }


def remove_expired_jobs(jobs_dir: Path, ttl_days: int, now_s: float) -> list[str]:
    """Borra las carpetas de trabajos con más de ttl_days. Devuelve los ids borrados."""
    removed = []
    if not jobs_dir.exists():
        return removed
    for job_dir in jobs_dir.iterdir():
        age_s = now_s - job_dir.stat().st_mtime
        if job_dir.is_dir() and age_s > ttl_days * SECONDS_PER_DAY:
            shutil.rmtree(job_dir, ignore_errors=True)
            removed.append(job_dir.name)
    return removed
