"""Lógica de un análisis: crearlo, seguir al worker de Modal y armar el reporte.

Estados: waiting_upload -> processing (en Modal) -> interpreting (en Render) -> done | error.
El resultado del worker llega por su aviso (callback) o, si Render estaba dormido, se va a
buscar la próxima vez que el frontend consulta el estado.
"""

import base64
import hmac
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from app.repositories.analyses import AnalysisRecord, AnalysisRepository
from app.schemas.analyses import (
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_INTERPRETING,
    STAGE_WAITING_UPLOAD,
    AnalysisStatus,
    AnalysisView,
    CreatedAnalysis,
    UploadInstructions,
)
from app.services.teacher_keys import hash_teacher_key, looks_like_key, new_teacher_key
from app.services.tickets import sign_ticket
from app.services.worker_client import WorkerGateway
from app.settings import ApiSettings

logger = logging.getLogger(__name__)

WORKER_JOBS_PATH = "/jobs"
MCP_PATH_PREFIX = "/mcp/"
WORKER_STATUS_DONE = "done"
WORKER_STATUS_ERROR = "error"
REPORT_ERROR = "No se pudo armar el reporte ({error_name})."
WAITING_FOR_WORKER = [AnalysisStatus.WAITING_UPLOAD, AnalysisStatus.PROCESSING]

Interpreter = Callable[[Any, Any], dict[str, Any]]


class InvalidAccessCodeError(Exception):
    """El código de acceso no coincide con UPLOAD_ACCESS_CODE."""


class AnalysisNotFoundError(Exception):
    pass


class InvalidTeacherKeyError(Exception):
    """La clave a reutilizar no tiene el formato o no pertenece a ningún análisis."""


def same_secret(given: str | None, expected: str) -> bool:
    if not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def to_view(record: AnalysisRecord) -> AnalysisView:
    return AnalysisView(
        analysis_id=record.id,
        status=AnalysisStatus(record.status),
        stage=record.stage,
        error=record.error,
        topic=record.topic,
        created_at=record.created_at,
        updated_at=record.updated_at,
        report_ready=record.report is not None,
    )


@dataclass
class AnalysisService:
    repository: AnalysisRepository
    worker: WorkerGateway
    interpreter: Interpreter
    settings: ApiSettings
    now: Callable[[], float] = field(default=time.time)
    new_id: Callable[[], str] = field(default=lambda: uuid.uuid4().hex)

    def get_record(self, analysis_id: str) -> AnalysisRecord:
        record = self.repository.get(analysis_id)
        if record is None:
            raise AnalysisNotFoundError(analysis_id)
        return record

    def teacher_key_for(self, reused_key: str | None) -> str:
        """Una clave nueva, o la de un análisis anterior del mismo docente (si existe)."""
        if reused_key is None:
            return new_teacher_key()
        if not looks_like_key(reused_key) or not self.repository.key_exists(
            hash_teacher_key(reused_key)
        ):
            raise InvalidTeacherKeyError()
        return reused_key

    def create(
        self,
        access_code: str,
        topic: str | None,
        api_base_url: str,
        reused_key: str | None = None,
    ) -> CreatedAnalysis:
        """Valida el código y entrega el permiso para subir el video directo a Modal."""
        if not same_secret(access_code, self.settings.upload_access_code):
            raise InvalidAccessCodeError()
        teacher_key = self.teacher_key_for(reused_key)
        analysis_id = self.new_id()
        self.repository.add(
            AnalysisRecord(
                id=analysis_id,
                topic=topic or None,
                status=AnalysisStatus.WAITING_UPLOAD.value,
                stage=STAGE_WAITING_UPLOAD,
                teacher_key_hash=hash_teacher_key(teacher_key),
            )
        )
        expires_at_s = int(self.now()) + self.settings.upload_ticket_ttl_s
        fields = {"analysis_id": analysis_id}
        if topic:
            fields["topic"] = topic
        upload = UploadInstructions(
            url=self.settings.worker_url + WORKER_JOBS_PATH,
            ticket=sign_ticket(analysis_id, expires_at_s, self.settings.worker_token),
            expires_at=datetime.fromtimestamp(expires_at_s, tz=UTC),
            fields=fields,
        )
        view = to_view(self.get_record(analysis_id))
        return CreatedAnalysis(
            **view.model_dump(),
            upload=upload,
            teacher_key=teacher_key,
            assistant_url=f"{api_base_url}{MCP_PATH_PREFIX}{teacher_key}",
        )

    def receive_raw(self, analysis_id: str, payload: dict[str, Any]) -> bool:
        """Guarda lo que entregó el worker. True si hay que interpretar y le toca a este llamado."""
        record = self.get_record(analysis_id)
        if record.status not in WAITING_FOR_WORKER:
            return False
        heatmap = None
        if payload.get("heatmap_png_base64"):
            heatmap = base64.b64decode(payload["heatmap_png_base64"])
        self.repository.update_fields(
            analysis_id,
            detections=payload["detections"],
            audio_raw=payload["audio_raw"],
            heatmap_png=heatmap,
        )
        return self.repository.claim(
            analysis_id,
            WAITING_FOR_WORKER,
            status=AnalysisStatus.INTERPRETING.value,
            stage=STAGE_INTERPRETING,
        )

    def refresh(self, analysis_id: str) -> tuple[AnalysisView, bool]:
        """Si el análisis sigue en Modal, pregunta al worker. True si hay que interpretar."""
        record = self.get_record(analysis_id)
        must_interpret = False
        if record.status in WAITING_FOR_WORKER:
            must_interpret = self.follow_worker(analysis_id)
        return to_view(self.get_record(analysis_id)), must_interpret

    def follow_worker(self, analysis_id: str) -> bool:
        try:
            job = self.worker.job_status(analysis_id)
            if job is None:
                return False
            if job["status"] == WORKER_STATUS_DONE:
                return self.receive_raw(analysis_id, self.worker.job_result(analysis_id))
        except httpx.HTTPError as error:
            # Modal no respondió: se vuelve a intentar en la próxima consulta
            logger.warning("No se pudo consultar el worker: %s", type(error).__name__)
            return False

        if job["status"] == WORKER_STATUS_ERROR:
            self.repository.claim(
                analysis_id,
                WAITING_FOR_WORKER,
                status=AnalysisStatus.ERROR.value,
                stage=STAGE_ERROR,
                error=job["error"],
            )
        else:
            self.repository.claim(
                analysis_id,
                WAITING_FOR_WORKER,
                status=AnalysisStatus.PROCESSING.value,
                stage=job["stage"],
            )
        return False

    def interpret(self, analysis_id: str) -> None:
        """En segundo plano: nunca lanza; si falla, el análisis queda en error."""
        record = self.get_record(analysis_id)
        try:
            report = self.interpreter(record.detections, record.audio_raw)
        except Exception as error:
            logger.exception("Falló la interpretación de %s", analysis_id)
            self.repository.update_fields(
                analysis_id,
                status=AnalysisStatus.ERROR.value,
                stage=STAGE_ERROR,
                error=REPORT_ERROR.format(error_name=type(error).__name__),
            )
            return
        self.repository.update_fields(
            analysis_id, status=AnalysisStatus.DONE.value, stage=STAGE_DONE, report=report
        )
