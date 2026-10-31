"""Configuración de la API: configs/api.yaml y variables de entorno (secretos de Render)."""

import os
from dataclasses import dataclass
from pathlib import Path

from espejo.common.config_io import CONFIGS_DIR, read_yaml

API_CONFIG_PATH = CONFIGS_DIR / "api.yaml"
# Render entrega postgresql://; SQLAlchemy necesita saber que el driver es psycopg (v3)
POSTGRES_PREFIXES = ("postgresql://", "postgres://")
PSYCOPG_PREFIX = "postgresql+psycopg://"


@dataclass(frozen=True)
class ApiSettings:
    database_url: str
    worker_url: str
    worker_token: str
    upload_access_code: str
    upload_ticket_ttl_s: int
    cors_origin_regex: str
    frontend_origins: list[str]
    worker_timeout_s: float


def sqlalchemy_url(database_url: str) -> str:
    for prefix in POSTGRES_PREFIXES:
        if database_url.startswith(prefix):
            return PSYCOPG_PREFIX + database_url[len(prefix) :]
    return database_url


def split_origins(raw: str | None) -> list[str]:
    origins = []
    for origin in (raw or "").split(","):
        if origin.strip():
            origins.append(origin.strip().rstrip("/"))
    return origins


def load_settings(config_path: Path = API_CONFIG_PATH) -> ApiSettings:
    """WORKER_TOKEN y UPLOAD_ACCESS_CODE son obligatorios: sin ellos la API no arranca."""
    raw = read_yaml(config_path)
    database_url = os.environ.get("DATABASE_URL") or raw["local_database_url"]
    return ApiSettings(
        database_url=sqlalchemy_url(database_url),
        worker_url=(os.environ.get("WORKER_URL") or raw["worker_url"]).rstrip("/"),
        worker_token=os.environ["WORKER_TOKEN"],
        upload_access_code=os.environ["UPLOAD_ACCESS_CODE"],
        upload_ticket_ttl_s=raw["upload_ticket_ttl_s"],
        cors_origin_regex=raw["cors_origin_regex"],
        frontend_origins=split_origins(os.environ.get("FRONTEND_ORIGINS")),
        worker_timeout_s=raw["worker_timeout_s"],
    )
