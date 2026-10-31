"""Entrada y salida de /api/analyses (contrato con el frontend: docs/api_contract.md)."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class AnalysisStatus(StrEnum):
    WAITING_UPLOAD = "waiting_upload"
    PROCESSING = "processing"
    INTERPRETING = "interpreting"
    DONE = "done"
    ERROR = "error"


STAGE_WAITING_UPLOAD = "Esperando el video"
STAGE_INTERPRETING = "Preparando el reporte"
STAGE_DONE = "Listo"
STAGE_ERROR = "Error"


class CreateAnalysisRequest(BaseModel):
    access_code: str
    topic: str | None = None
    # Opcional: la clave de un análisis anterior, para agruparlos en el mismo asistente
    teacher_key: str | None = None


class UploadInstructions(BaseModel):
    """Cómo subir el video directo al worker de Modal (POST multipart)."""

    url: str
    ticket: str
    expires_at: datetime
    fields: dict[str, str]


class AnalysisView(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    stage: str
    error: str | None
    topic: str | None
    created_at: datetime
    updated_at: datetime
    report_ready: bool


class CreatedAnalysis(AnalysisView):
    upload: UploadInstructions
    # Se muestra una sola vez y solo al docente: no se puede volver a pedir
    teacher_key: str
    assistant_url: str


class RawReceived(BaseModel):
    analysis_id: str
    status: AnalysisStatus
