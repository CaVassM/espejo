"""Rutas de /api/analyses: delgadas, toda la lógica está en services/analyses.py."""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response, status

from app.schemas.analyses import (
    AnalysisStatus,
    AnalysisView,
    CreateAnalysisRequest,
    CreatedAnalysis,
    RawReceived,
)
from app.services.analyses import (
    AnalysisNotFoundError,
    AnalysisService,
    InvalidAccessCodeError,
    InvalidTeacherKeyError,
    same_secret,
)

BEARER_PREFIX = "Bearer "
PNG_MEDIA_TYPE = "image/png"

router = APIRouter(prefix="/analyses", tags=["analyses"])


def service_of(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "No existe ese análisis.")


@router.post("", status_code=status.HTTP_201_CREATED)
def create_analysis(body: CreateAnalysisRequest, request: Request) -> CreatedAnalysis:
    service = service_of(request)
    api_base_url = service.settings.public_api_url or str(request.base_url).rstrip("/")
    try:
        return service.create(body.access_code, body.topic, api_base_url, body.teacher_key)
    except InvalidAccessCodeError as error:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Código de acceso inválido.") from error
    except InvalidTeacherKeyError as error:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clave del docente inválida.") from error


@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: str, request: Request, background_tasks: BackgroundTasks
) -> AnalysisView:
    service = service_of(request)
    try:
        view, must_interpret = service.refresh(analysis_id)
    except AnalysisNotFoundError as error:
        raise not_found() from error
    if must_interpret:
        background_tasks.add_task(service.interpret, analysis_id)
    return view


@router.get("/{analysis_id}/report")
def get_report(analysis_id: str, request: Request) -> dict[str, Any]:
    try:
        record = service_of(request).get_record(analysis_id)
    except AnalysisNotFoundError as error:
        raise not_found() from error
    if record.report is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "El reporte aún no está listo.")
    return record.report


@router.get("/{analysis_id}/detections")
def get_detections(analysis_id: str, request: Request) -> dict[str, Any]:
    """Las cajas que detectaron los modelos de visión, para dibujarlas sobre el video.

    El worker ya las entrega en /raw y quedan guardadas: aquí solo se sirven. Se
    responde en cuanto existen, sin esperar al reporte, porque VER termina antes
    que INTERPRETAR.
    """
    try:
        record = service_of(request).get_record(analysis_id)
    except AnalysisNotFoundError as error:
        raise not_found() from error
    if record.detections is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "El análisis del video aún no está listo.")
    return record.detections


@router.get("/{analysis_id}/heatmap.png")
def get_heatmap(analysis_id: str, request: Request) -> Response:
    try:
        record = service_of(request).get_record(analysis_id)
    except AnalysisNotFoundError as error:
        raise not_found() from error
    if record.heatmap_png is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Este análisis no tiene heatmap.")
    return Response(content=record.heatmap_png, media_type=PNG_MEDIA_TYPE)


@router.post("/{analysis_id}/raw", status_code=status.HTTP_202_ACCEPTED)
def receive_raw(
    analysis_id: str,
    payload: dict[str, Any],
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> RawReceived:
    """El aviso del worker de Modal al terminar VER (Authorization: Bearer WORKER_TOKEN)."""
    service = service_of(request)
    token = None
    if authorization and authorization.startswith(BEARER_PREFIX):
        token = authorization[len(BEARER_PREFIX) :]
    if not same_secret(token, service.settings.worker_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token inválido.")
    try:
        must_interpret = service.receive_raw(analysis_id, payload)
    except AnalysisNotFoundError as error:
        raise not_found() from error
    if must_interpret:
        background_tasks.add_task(service.interpret, analysis_id)
        return RawReceived(analysis_id=analysis_id, status=AnalysisStatus.INTERPRETING)
    record = service.get_record(analysis_id)
    return RawReceived(analysis_id=analysis_id, status=AnalysisStatus(record.status))
