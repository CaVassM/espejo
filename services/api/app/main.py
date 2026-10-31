"""Punto de entrada de la API: arma la app, la base de datos, CORS, las rutas y el servidor MCP.

En Render:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from app.mcp_server import (
    RedactTeacherKeyFilter,
    TeacherKeyMcpMiddleware,
    build_mcp_app,
    build_mcp_server,
)
from app.repositories.analyses import AnalysisRepository, create_db_engine, create_tables
from app.routers import analyses, health
from app.services.analyses import AnalysisService
from app.services.assistant import AssistantQueries, load_assistant_settings
from app.services.interpretation import interpret_raw
from app.services.worker_client import WorkerClient
from app.settings import API_CONFIG_PATH, ApiSettings, load_settings

API_PREFIX = "/api"
ACCESS_LOGGER = "uvicorn.access"


def build_service(settings: ApiSettings) -> AnalysisService:
    engine = create_db_engine(settings.database_url)
    create_tables(engine)
    repository = AnalysisRepository(sessionmaker(engine, expire_on_commit=False))
    worker = WorkerClient(settings.worker_url, settings.worker_token, settings.worker_timeout_s)
    return AnalysisService(repository, worker, interpret_raw, settings)


def create_app(
    settings: ApiSettings | None = None, service: AnalysisService | None = None
) -> FastAPI:
    """settings y service se inyectan en los tests (base en memoria y worker simulado)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # La URL del asistente lleva la clave del docente: nunca debe quedar en los logs
    logging.getLogger(ACCESS_LOGGER).addFilter(RedactTeacherKeyFilter())
    settings = settings or load_settings()
    service = service or build_service(settings)

    assistant_settings = load_assistant_settings(API_CONFIG_PATH)
    mcp_server = build_mcp_server(AssistantQueries(service.repository, assistant_settings))
    mcp_app = build_mcp_app(mcp_server)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with mcp_server.session_manager.run():
            yield

    app = FastAPI(title="Espejo API", lifespan=lifespan)
    app.state.analysis_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TeacherKeyMcpMiddleware, mcp_app=mcp_app)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(analyses.router, prefix=API_PREFIX)
    return app
