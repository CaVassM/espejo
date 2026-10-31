"""Prompts y esquemas JSON del LLM. La parte fija son las definiciones del manual del Minedu.

Cada ventana se clasifica con dos llamados: uno para los eventos (retroalimentación,
regulación, participación y respeto) y otro para las preguntas. Con un solo llamado, un modelo
local de 7B se concentraba en la lista de preguntas y devolvía vacíos los eventos.
"""

from typing import Any

from espejo.llm.schemas import (
    ANSWERED_QUESTION_LABELS,
    CONFIDENCE_LABELS,
    FEEDBACK_LABELS,
    PARTICIPATION_LABELS,
    QUESTION_LABELS,
    REGULATION_LABELS,
    RESPECT_LABELS,
    CandidateQuestion,
    SuggestionFocus,
    TranscriptLine,
    Weakness,
)
from espejo.llm.transcript_view import TranscriptWindow, format_candidate, format_line

NO_TOPIC = "(no indicado)"
NONE_TEXT = "(ninguna)"


def evidence_item_schema(labels: list[str], with_addressee: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "label": {"type": "string", "enum": labels},
        "quote": {"type": "string"},
        "timestamp": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": CONFIDENCE_LABELS},
    }
    required = ["label", "quote", "timestamp", "reason", "confidence"]
    if with_addressee:
        properties["addressee"] = {"type": "string"}
        required = [*required, "addressee"]
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
        "feedback": array_of(evidence_item_schema(FEEDBACK_LABELS)),
        "regulation": array_of(evidence_item_schema(REGULATION_LABELS)),
        "participation": array_of(evidence_item_schema(PARTICIPATION_LABELS, with_addressee=True)),
        "respect": array_of(evidence_item_schema(RESPECT_LABELS)),
    },
    "required": ["feedback", "regulation", "participation", "respect"],
    "additionalProperties": False,
}


# Las preguntas no llevan cita ni minuto: el question_id ya apunta a la pregunta real y el código
# pone su texto y su minuto. Un 7B copiaba mal la cita (le pegaba el minuto) y se perdían.
def question_item_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question_id": {"type": "integer"},
            "label": {"type": "string", "enum": labels},
            "reason": {"type": "string"},
            "confidence": {"type": "string", "enum": CONFIDENCE_LABELS},
        },
        "required": ["question_id", "label", "reason", "confidence"],
        "additionalProperties": False,
    }


def questions_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"questions": array_of(question_item_schema(labels))},
        "required": ["questions"],
        "additionalProperties": False,
    }


QUESTIONS_SCHEMA = questions_schema(QUESTION_LABELS)
# Segundo llamado, solo para preguntas que un alumno respondió y que el LLM marcó como
# retóricas: el esquema solo deja elegir entre abierta y cerrada.
ANSWERED_QUESTIONS_SCHEMA = questions_schema(ANSWERED_QUESTION_LABELS)

SUGGESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "why_it_matters": {"type": "string", "minLength": 1},
        "try_this": {"type": "string", "minLength": 1},
        "example_phrase": {"type": "string", "minLength": 1},
    },
    "required": ["why_it_matters", "try_this", "example_phrase"],
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
  tipos del manual: feedback_elemental, feedback_descriptive, feedback_discovery. Por ejemplo,
  "Muy bien" tras una respuesta es feedback_elemental. Lo que dicen los alumnos aparece como
  "[ALUMNO responde]": no juzgues si su respuesta era correcta.
- regulation: mecanismos para regular el comportamiento: regulation_formative,
  regulation_external_control, regulation_mistreatment.
- participation: participation_invitation SOLO cuando el docente se dirige a UN estudiante en
  particular: lo llama por su nombre, le dice "a ver tú", "a ver usted", "hija", "hijo", o le da
  la palabra. En "addressee" copia el nombre o la palabra con que se dirige a ese estudiante;
  debe estar dentro de la cita. Las preguntas a toda la clase ("¿qué pasará?", "niños...",
  "ustedes...") NO son invitaciones: no las marques.
- respect: respect_issue, solo si hay apodos, burlas o insultos (del docente o entre alumnos).
"""

QUESTIONS_TASK = """\
Tarea: clasifica CADA pregunta candidata (usa su question_id) mirando las líneas de antes y
de después:
- question_open: pide razonar, inferir, predecir, explicar, justificar, comparar o crear.
- question_closed: pide afirmar o negar, un dato puntual o recordar información ya dada.
- question_rhetorical: muletilla ("¿no?", "¿ya?", "¿cierto?") o pregunta que no busca
  respuesta.
Clasifica TODAS las preguntas candidatas: un elemento por cada question_id. "reason" es una
sola línea breve, en español. "confidence" es alta, media o baja.
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


def candidate_block(candidates: list[CandidateQuestion]) -> str:
    formatted = []
    for candidate in candidates:
        formatted.append(format_candidate(candidate))
    return joined_or_none(formatted)


def build_questions_prompt(window: TranscriptWindow, definitions: str, topic: str | None) -> str:
    return (
        f"{ROLE_TEXT}\n{QUESTIONS_TASK}\n"
        f"{shared_context(definitions, topic, window)}\n"
        "PREGUNTAS CANDIDATAS (question_id | [mm:ss] | texto | si respondió un alumno):\n"
        f"{candidate_block(window.candidates)}\n"
    )


ANSWERED_QUESTIONS_TASK = """\
Tarea: un alumno respondió cada una de estas preguntas, así que NO son retóricas (el manual
define la retórica como la que se hace "sin esperar una respuesta del estudiante"). Mirando las
líneas de antes y de después, elige SOLO entre:
- question_open: pide razonar, inferir, predecir, explicar, justificar, comparar o crear.
- question_closed: pide afirmar o negar, un dato puntual o recordar información ya dada.
Un elemento por cada question_id. "reason" es una sola línea breve, en español. "confidence"
es alta, media o baja.
"""


def build_answered_questions_prompt(
    window: TranscriptWindow,
    definitions: str,
    topic: str | None,
    candidates: list[CandidateQuestion],
) -> str:
    return (
        f"{ROLE_TEXT}\n{ANSWERED_QUESTIONS_TASK}\n"
        f"{shared_context(definitions, topic, window)}\n"
        "PREGUNTAS QUE UN ALUMNO RESPONDIÓ (question_id | [mm:ss] | texto | si respondió):\n"
        f"{candidate_block(candidates)}\n"
    )


SUGGESTION_INSTRUCTIONS = """\
Eres un colega docente con experiencia que acompaña a otro docente del Perú. Escribe en español,
en tono cálido y cercano, de colega a colega (trátalo de "tú"). Usa palabras simples, sin
términos técnicos, sin siglas y sin poner notas ni niveles.
Ya elegimos el punto a trabajar y ya escribimos lo que vimos en la clase. Tú escribe:
- "why_it_matters": por qué importa para el aprendizaje de los estudiantes, en una o dos
  oraciones cortas. Empieza directo con la idea (no con "Importa porque").
- "try_this": UNA sola acción concreta para la próxima clase, en una oración que empiece con
  un verbo (por ejemplo "Después de preguntar, ...").
- "example_phrase": una frase corta que el docente diría en su clase MIENTRAS hace esa acción,
  sobre el tema de la clase. Sin comillas.
No repitas lo que vimos y no menciones otros momentos de la clase.
"""

# Lo que se sabe de cada debilidad, en palabras simples, para que el LLM no lo invente.
WEAKNESS_TEXT = {
    Weakness.SHORT_WAIT: (
        "Esperar muy poco después de preguntar. Esperar al menos 3 segundos en silencio da "
        "tiempo para pensar: responden más estudiantes y con respuestas más completas."
    ),
    Weakness.ELEMENTAL_FEEDBACK: (
        'La retroalimentación solo confirma ("Muy bien"). El manual valora pedirle al '
        "estudiante que explique cómo llegó a su respuesta, o decirle qué hizo bien y qué le "
        "falta."
    ),
    Weakness.LOW_MONITORING: (
        "Poco tiempo recogiendo evidencia de lo que entienden los estudiantes (preguntándoles o "
        "recorriendo sus lugares para ver su trabajo). El manual pide al menos un 25 % de la clase."
    ),
    Weakness.CLOSED_QUESTIONS: (
        "La mayoría de preguntas piden un dato o un sí/no. Las preguntas que piden explicar, "
        "predecir o dar razones hacen pensar más."
    ),
}


def build_suggestion_prompt(
    focus: SuggestionFocus, context_lines: list[TranscriptLine], topic: str | None
) -> str:
    context = []
    for line in context_lines:
        context.append(format_line(line))
    return (
        f"{SUGGESTION_INSTRUCTIONS}\n"
        f"PUNTO A TRABAJAR: {WEAKNESS_TEXT[focus.weakness]}\n\n"
        f"LO QUE VIMOS: {focus.what_we_saw}\n\n"
        f"TEMA DE LA CLASE: {topic or NO_TOPIC}\n\n"
        f"LO QUE SE DIJO CERCA DE ESE MOMENTO:\n{joined_or_none(context)}\n"
    )


AI_TIPS_COUNT = 3

AI_TIPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "what_for": {"type": "string"},
                    "how": {"type": "string"},
                    "example_prompt": {"type": "string"},
                },
                "required": ["title", "what_for", "how", "example_prompt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tips"],
    "additionalProperties": False,
}

AI_TIPS_INSTRUCTIONS = f"""\
Eres un colega docente del Perú que ya usa asistentes de IA (ChatGPT, Claude, Gemini) para
preparar sus clases y le enseña a otro docente a hacerlo. Escribe en español, tratándolo de
"tú", con palabras simples y sin tecnicismos.

Propón exactamente {AI_TIPS_COUNT} formas de usar un asistente de IA para PREPARAR LA PRÓXIMA
CLASE, elegidas por lo que se observó en esta clase y sobre el tema de esta clase.
Para cada una:
- "title": un nombre corto, de 3 a 6 palabras, que diga qué se logra.
- "what_for": en una oración, qué problema de ESTA clase ayuda a resolver.
- "how": en una o dos oraciones, qué hace el docente paso a paso. Empieza con un verbo.
- "example_prompt": el texto EXACTO que el docente copia y pega en el asistente de IA. Escríbelo
  en primera persona, ya completo con el tema y el grado de esta clase, listo para usar sin
  editar nada. De 2 a 4 oraciones.

Reglas:
- La IA prepara materiales ANTES de la clase; no reemplaza al docente ni evalúa a los estudiantes.
- No propongas subir datos, nombres, fotos ni grabaciones de estudiantes a ninguna herramienta.
- Nada de notas, niveles ni puntajes.
- Las tres propuestas deben ser distintas entre sí.
"""


def build_ai_tips_prompt(observations: list[str], topic: str | None) -> str:
    lines = []
    for observation in observations:
        lines.append(f"- {observation}")
    return (
        f"{AI_TIPS_INSTRUCTIONS}\n"
        f"TEMA DE LA CLASE: {topic or NO_TOPIC}\n\n"
        f"LO QUE SE OBSERVÓ EN ESTA CLASE:\n{joined_or_none(lines)}\n"
    )
