"""Estado de la API, para el monitoreo de Render."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse

STATUS_OK = "ok"

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status=STATUS_OK)
