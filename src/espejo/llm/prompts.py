"""Prompts y esquemas JSON del LLM. La parte fija son las definiciones del manual del Minedu.

Cada ventana se clasifica con dos llamados: uno para los eventos (retroalimentación,
regulación, participación y respeto) y otro para las preguntas. Con un solo llamado, un modelo
local de 7B se concentraba en la lista de preguntas y devolvía vacíos los eventos.
"""

from typing import Any

from espejo.llm.schemas import (
    CONFIDENCE_LABELS,
    FEEDBACK_LABELS,
    PARTICIPATION_LABELS,
    QUESTION_LABELS,
    REGULATION_LABELS,
    RESPECT_LABELS,
)
from espejo.llm.transcript_view import TranscriptWindow, format_candidate, format_line

NO_TOPIC = "(no indicado)"
NONE_TEXT = "(ninguna)"


def evidence_item_schema(labels: list[str], with_question_id: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "label": {"type": "string", "enum": labels},
        "quote": {"type": "string"},
        "timestamp": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": CONFIDENCE_LABELS},
    }
    required = ["label", "quote", "timestamp", "reason", "confidence"]
    if with_question_id:
        properties["question_id"] = {"type": "integer"}
        required = ["question_id", *required]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def array_of(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item_schema}


EVENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "feedback": array_of(evidence_item_schema(FEEDBACK_LABELS, with_question_id=False)),
        "regulation": array_of(evidence_item_schema(REGULATION_LABELS, with_question_id=False)),
        "participation": array_of(
            evidence_item_schema(PARTICIPATION_LABELS, with_question_id=False)
        ),
        "respect": array_of(evidence_item_schema(RESPECT_LABELS, with_question_id=False)),
    },
    "required": ["feedback", "regulation", "participation", "respect"],
    "additionalProperties": False,
}

QUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": array_of(evidence_item_schema(QUESTION_LABELS, with_question_id=True)),
    },
    "required": ["questions"],
    "additionalProperties": False,
}

SUGGESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggestion": {"type": "string"},
        "timestamp": {"type": "string"},
    },
    "required": ["suggestion", "timestamp"],
    "additionalProperties": False,
}

ROLE_TEXT = """\
Eres un observador pedagógico que apoya a docentes del Perú. Lees la transcripción de un
tramo de una clase. NO pones notas ni niveles: solo marcas evidencias, cada una con una cita.
Usa las definiciones del manual que están abajo para decidir. Responde siempre en español.
"""

RULES_TEXT = """\
Reglas obligatorias:
- "quote" es una cita TEXTUAL copiada de la transcripción, sin cambiar ninguna palabra.
- "timestamp" es el [mm:ss] de la línea donde está la cita.
- "reason" es una sola línea breve, en español. "confidence" es alta, media o baja.
- Si no hay elementos de un tipo, devuelve una lista vacía. No inventes nada.
"""

EVENTS_TASK = """\
Tarea: revisa TODA la transcripción y marca:
- feedback: lo que dice el docente justo DESPUÉS de una respuesta de un alumno, según los
  cuatro tipos del manual: feedback_incorrect, feedback_elemental, feedback_descriptive,
  feedback_discovery. Por ejemplo, "Muy bien" tras una respuesta es feedback_elemental.
- regulation: mecanismos para regular el comportamiento: regulation_formative,
  regulation_external_control, regulation_mistreatment.
- participation: participation_invitation, cuando el docente invita a participar o dirige una
  pregunta a quien no ha participado (por ejemplo, llama a un alumno por su nombre).
- respect: respect_issue, solo si hay apodos, burlas o insultos (del docente o entre alumnos).
"""

QUESTIONS_TASK = """\
Tarea: clasifica CADA pregunta candidata (usa su question_id) mirando las líneas de antes y
de después:
- question_open: pide razonar, inferir, predecir, explicar, justificar, comparar o crear.
- question_closed: pide afirmar o negar, un dato puntual o recordar información ya dada.
- question_rhetorical: muletilla ("¿no?", "¿ya?", "¿cierto?") o pregunta que no busca
  respuesta.
"""


def joined_or_none(items: list[str]) -> str:
    if not items:
        return NONE_TEXT
    return "\n".join(items)


def transcript_block(window: TranscriptWindow) -> str:
    lines = []
    for line in window.lines:
        lines.append(format_line(line))
    return joined_or_none(lines)


def shared_context(definitions: str, topic: str | None, window: TranscriptWindow) -> str:
    return (
        f"DEFINICIONES DEL MANUAL DE RÚBRICAS DEL MINEDU:\n{definitions}\n\n"
        f"TEMA DE LA CLASE: {topic or NO_TOPIC}\n\n"
        f"TRANSCRIPCIÓN:\n{transcript_block(window)}\n"
    )


def build_events_prompt(window: TranscriptWindow, definitions: str, topic: str | None) -> str:
    return f"{ROLE_TEXT}\n{EVENTS_TASK}\n{RULES_TEXT}\n{shared_context(definitions, topic, window)}"


def build_questions_prompt(window: TranscriptWindow, definitions: str, topic: str | None) -> str:
    candidates = []
    for candidate in window.candidates:
        candidates.append(format_candidate(candidate))
    return (
        f"{ROLE_TEXT}\n{QUESTIONS_TASK}\n{RULES_TEXT}\n"
        f"{shared_context(definitions, topic, window)}\n"
        f"PREGUNTAS CANDIDATAS (question_id | [mm:ss] | texto):\n{joined_or_none(candidates)}\n"
    )


SUGGESTION_INSTRUCTIONS = """\
Eres un acompañante pedagógico cálido y profesional. Con SOLO la evidencia de abajo, escribe
UN consejo concreto para la próxima clase del docente (dos o tres oraciones como máximo), en
español. No califiques ni pongas notas. En "timestamp" pon el [mm:ss] de la evidencia que
motiva el consejo; debe ser uno de los minutos que aparecen en la evidencia.
"""


def build_suggestion_prompt(evidence_text: str, topic: str | None) -> str:
    return (
        f"{SUGGESTION_INSTRUCTIONS}\n"
        f"TEMA DE LA CLASE: {topic or NO_TOPIC}\n\n"
        f"EVIDENCIA DE LA CLASE:\n{evidence_text}\n"
    )
