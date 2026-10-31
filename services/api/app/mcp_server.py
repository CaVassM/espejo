"""Servidor MCP de solo lectura: el asistente de IA del docente consulta sus clases.

Se publica en /mcp/{teacher_key} (transporte streamable HTTP). La clave de la URL decide qué
análisis se ven: este intermediario guarda su hash en el pedido y quita la clave de la ruta,
y las herramientas leen ese hash. Una clave inválida no ve nada. La clave nunca va a los logs.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from app.services.assistant import AssistantError, AssistantQueries
from app.services.teacher_keys import hash_teacher_key, looks_like_key

MCP_INNER_PATH = "/mcp"
MCP_KEY_PATH = re.compile(r"^/mcp/([^/]+)/?$")
MCP_KEY_IN_TEXT = re.compile(r"/mcp/[^/\s\"?]+")
REDACTED_PATH = "/mcp/[clave]"
KEY_HASH_SCOPE = "espejo.teacher_key_hash"

INSTRUCTIONS = """\
Espejo analiza clases grabadas y entrega evidencias según las rúbricas de observación de aula del
Minedu (Perú): R1 involucra, R2 razonamiento, R3 evalúa el progreso, R4 respeto y proximidad,
R5 regula el comportamiento. Estas herramientas solo leen los análisis del docente dueño de la
clave. No califiques al docente ni pongas notas: describe evidencias y sugiere con tono cálido.
Las métricas marcadas "experimental" (pizarra, guiar en carpetas) son poco confiables.
"""

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)


def key_hash_of(ctx: Context) -> str | None:
    request = ctx.request_context.request
    if request is None:
        return None
    return request.scope.get(KEY_HASH_SCOPE)


def answer(query: Callable[..., Any], *args: Any) -> Any:
    """Los errores previstos (no existe, no está listo) llegan al asistente como texto claro."""
    try:
        return query(*args)
    except AssistantError as error:
        raise ToolError(str(error)) from error


def build_mcp_server(queries: AssistantQueries) -> MCPServer:
    server = MCPServer(name="Espejo", instructions=INSTRUCTIONS)

    @server.tool(
        name="listar_mis_clases",
        description=(
            "Lista las clases analizadas del docente: id, fecha, tema, duración en minutos y "
            "estado. Úsala primero para saber qué id pasar a las otras herramientas."
        ),
        annotations=READ_ONLY,
    )
    def listar_mis_clases(ctx: Context) -> list[dict[str, Any]]:
        return queries.list_classes(key_hash_of(ctx))

    @server.tool(
        name="ver_reporte",
        description=(
            "Muestra el reporte de una clase: la ficha por rúbrica (R1 a R5, con su estado y "
            "métricas), las métricas principales (preguntas, % abiertas, espera, monitoreo, "
            "respuestas) y el consejo en 3 partes (lo que vimos, por qué importa, prueba esto)."
        ),
        annotations=READ_ONLY,
    )
    def ver_reporte(id: str, ctx: Context) -> dict[str, Any]:
        return answer(queries.report, key_hash_of(ctx), id)

    @server.tool(
        name="ver_evidencias",
        description=(
            "Muestra las citas del docente, con su minuto, que respaldan una rúbrica (R1, R2, "
            "R3, R4 o R5) en una clase. Úsala para explicar con ejemplos concretos."
        ),
        annotations=READ_ONLY,
    )
    def ver_evidencias(id: str, rubrica: str, ctx: Context) -> dict[str, Any]:
        return answer(queries.evidence, key_hash_of(ctx), id, rubrica)

    @server.tool(
        name="comparar_clases",
        description=(
            "Compara dos clases del mismo docente: espera promedio después de preguntar, % de "
            "preguntas abiertas, monitoreo y tipos de retroalimentación. Úsala para ver avances."
        ),
        annotations=READ_ONLY,
    )
    def comparar_clases(id_a: str, id_b: str, ctx: Context) -> dict[str, Any]:
        return answer(queries.compare, key_hash_of(ctx), id_a, id_b)

    @server.prompt(
        name="planificar_proxima_clase",
        description=(
            "Pide un plan breve para la próxima clase, basado en el consejo y las evidencias de "
            "una clase analizada, sin calificar al docente."
        ),
    )
    def planificar_proxima_clase(id: str, ctx: Context) -> str:
        return answer(queries.plan_prompt, key_hash_of(ctx), id)

    return server


def build_mcp_app(server: MCPServer) -> Starlette:
    """Sin estado (cada pedido es independiente): sirve con varias instancias y tras dormir.

    La protección contra "DNS rebinding" es para servidores en la PC del usuario; este es
    público y lo protege la clave.
    """
    return server.streamable_http_app(
        streamable_http_path=MCP_INNER_PATH,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )


class TeacherKeyMcpMiddleware:
    """Lleva /mcp/{teacher_key} al servidor MCP, con el hash de la clave y sin la clave."""

    def __init__(self, app: ASGIApp, mcp_app: ASGIApp) -> None:
        self.app = app
        self.mcp_app = mcp_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            match = MCP_KEY_PATH.match(scope["path"])
            if match is not None:
                key = match.group(1)
                inner = dict(scope, path=MCP_INNER_PATH, raw_path=MCP_INNER_PATH.encode())
                inner[KEY_HASH_SCOPE] = hash_teacher_key(key) if looks_like_key(key) else None
                await self.mcp_app(inner, receive, send)
                return
        await self.app(scope, receive, send)


class RedactTeacherKeyFilter(logging.Filter):
    """Tapa la clave en los logs de acceso de uvicorn (que registran la ruta pedida)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            redacted = []
            for arg in record.args:
                if isinstance(arg, str):
                    arg = MCP_KEY_IN_TEXT.sub(REDACTED_PATH, arg)
                redacted.append(arg)
            record.args = tuple(redacted)
        if isinstance(record.msg, str):
            record.msg = MCP_KEY_IN_TEXT.sub(REDACTED_PATH, record.msg)
        return True
