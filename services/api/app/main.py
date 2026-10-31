"""Punto de entrada de la API: arma la app y monta las rutas."""

from fastapi import FastAPI

from app.routers import health

API_PREFIX = "/api"

app = FastAPI(title="Espejo API")
app.include_router(health.router, prefix=API_PREFIX)
