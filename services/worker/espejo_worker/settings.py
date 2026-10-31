"""Configuración del worker (configs/worker.yaml) y secretos (Modal Secret "espejo-secrets")."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class WorkerSettings:
    app_name: str
    gpu: str
    timeout_s: int
    scaledown_window_s: int
    max_containers: int
    web_timeout_s: int
    web_scaledown_window_s: int
    web_max_inputs: int
    max_upload_mb: int
    upload_chunk_mb: int
    allowed_extensions: list[str]
    cors_origin_regex: str
    job_ttl_days: int
    callback_timeout_s: float
    callback_attempts: int
    callback_backoff_s: float
    models_volume: str
    jobs_volume: str
    jobs_dict: str
    secret_name: str

    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * BYTES_PER_MB

    def upload_chunk_bytes(self) -> int:
        return self.upload_chunk_mb * BYTES_PER_MB


@dataclass(frozen=True)
class WorkerSecrets:
    """Vienen del Secret de Modal como variables de entorno. Nunca se registran en logs."""

    worker_token: str
    upload_access_code: str
    api_base_url: str | None
    frontend_origins: list[str]


def load_settings(path: Path) -> WorkerSettings:
    with path.open(encoding="utf-8") as config_file:
        return WorkerSettings(**yaml.safe_load(config_file))


def load_secrets() -> WorkerSecrets:
    api_base_url = os.environ.get("API_BASE_URL") or None
    if api_base_url is not None:
        api_base_url = api_base_url.rstrip("/")
    origins = []
    for origin in os.environ.get("FRONTEND_ORIGINS", "").split(","):
        if origin.strip():
            origins.append(origin.strip().rstrip("/"))
    return WorkerSecrets(
        worker_token=os.environ["WORKER_TOKEN"],
        upload_access_code=os.environ["UPLOAD_ACCESS_CODE"],
        api_base_url=api_base_url,
        frontend_origins=origins,
    )
