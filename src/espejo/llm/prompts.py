"""Prompts y esquemas JSON del LLM. La parte fija son las definiciones del manual del Minedu."""

from typing import Any

from espejo.llm.schemas import (
    CONFIDENCE_LABELS,
    FEEDBACK_LABELS,
    QUESTION_LABELS,
    REGULATION_LABELS,
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


CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": array_of(evidence_item_schema(QUESTION_LABELS, with_question_id=True)),
        "feedback": array_of(evidence_item_schema(FEEDBACK_LABELS, with_question_id=False)),
        "regulation": array_of(evidence_item_schema(REGULATION_LABELS, with_question_id=False)),
    },
    "required": ["questions", "feedback", "regulation"],
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

CLASSIFICATION_INSTRUCTIONS = """\
Eres un observador pedagógico que apoya a docentes del Perú. Lees la transcripción de un
tramo de una clase. NO pones notas ni niveles: solo marcas evidencias, cada una con una cita.

Tareas:
a) Clasifica CADA pregunta candidata (usa su question_id) como:
   - question_open: pide razonar, explicar, predecir, justificar o crear.
   - question_closed: pide recordar, repetir, completar o responder sí/no.
   - question_rhetorical: muletilla ("¿no?", "¿ya?", "¿cierto?") o pregunta que no busca
     respuesta. Mira las líneas de antes y de después para decidir.
b) Retroalimentación del docente justo DESPUÉS de una respuesta de un alumno:
   feedback_incorrect, feedback_elemental, feedback_descriptive, feedback_discovery.
c) Regulación del comportamiento: regulation_formative, regulation_external_control,
   regulation_mistreatment.
Usa las definiciones del manual que están abajo para decidir.

Reglas obligatorias:
- "quote" es una cita TEXTUAL copiada de la transcripción, sin cambiar ninguna palabra.
- "timestamp" es el [mm:ss] de la línea donde está la cita.
- "reason" es una sola línea breve. "confidence" es alta, media o baja.
- Si no hay eventos de un tipo, devuelve una lista vacía. No inventes nada.
"""

SUGGESTION_INSTRUCTIONS = """\
Eres un acompañante pedagógico cálido y profesional. Con SOLO la evidencia de abajo, escribe
UN consejo concreto para la próxima clase del docente (dos o tres oraciones como máximo).
No califiques ni pongas notas. En "timestamp" pon el [mm:ss] de la evidencia que motiva el
consejo; debe ser uno de los minutos que aparecen en la evidencia.
"""


def joined_or_none(items: list[str]) -> str:
    if not items:
        return NONE_TEXT
    return "\n".join(items)


def build_classification_prompt(
    window: TranscriptWindow, definitions: str, topic: str | None
) -> str:
    lines = []
    for line in window.lines:
        lines.append(format_line(line))
    candidates = []
    for candidate in window.candidates:
        candidates.append(format_candidate(candidate))

    return (
        f"{CLASSIFICATION_INSTRUCTIONS}\n"
        f"DEFINICIONES DEL MANUAL DE RÚBRICAS DEL MINEDU:\n{definitions}\n\n"
        f"TEMA DE LA CLASE: {topic or NO_TOPIC}\n\n"
        f"TRANSCRIPCIÓN:\n{joined_or_none(lines)}\n\n"
        f"PREGUNTAS CANDIDATAS (question_id | [mm:ss] | texto):\n{joined_or_none(candidates)}\n"
    )


def build_suggestion_prompt(evidence_text: str, topic: str | None) -> str:
    return (
        f"{SUGGESTION_INSTRUCTIONS}\n"
        f"TEMA DE LA CLASE: {topic or NO_TOPIC}\n\n"
        f"EVIDENCIA DE LA CLASE:\n{evidence_text}\n"
    )
