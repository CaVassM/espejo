"""Un solo consejo para la próxima clase, sobre la debilidad que eligieron las reglas.

"Lo que vimos" (minuto y cita) viene de los datos; el LLM solo escribe "Por qué importa" y
"Prueba esto" (una acción y una frase de ejemplo).
"""

import logging
from dataclasses import dataclass

from espejo.config import LlmConfig
from espejo.llm.clients import LlmClient, LlmError, complete_json
from espejo.llm.prompts import SUGGESTION_SCHEMA, build_suggestion_prompt
from espejo.llm.schemas import LlmUsage, Suggestion, SuggestionFocus, TranscriptLine
from espejo.llm.transcript_view import lines_between

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuggestionOutcome:
    suggestion: Suggestion | None
    usage: LlmUsage
    note: str | None = None


def lines_near(
    lines: list[TranscriptLine], timestamp_s: float | None, context_s: float
) -> list[TranscriptLine]:
    """Lo que se dijo alrededor del momento citado, para que el ejemplo encaje con la clase."""
    if timestamp_s is None:
        return []
    return lines_between(lines, timestamp_s - context_s, timestamp_s + context_s)


class SuggestionWriter:
    def __init__(self, client: LlmClient, config: LlmConfig) -> None:
        self.client = client
        self.config = config

    def write(
        self, focus: SuggestionFocus, lines: list[TranscriptLine], topic: str | None
    ) -> SuggestionOutcome:
        context = lines_near(lines, focus.timestamp_s, self.config.suggestion.context_s)
        prompt = build_suggestion_prompt(focus, context, topic)
        try:
            reply = complete_json(self.client, prompt, SUGGESTION_SCHEMA, self.config.max_attempts)
        except LlmError as error:
            logger.error("No se pudo generar el consejo: %s", error)
            return SuggestionOutcome(None, LlmUsage(), "el LLM no devolvió un consejo válido")

        content = reply.content
        suggestion = Suggestion(
            focus=focus,
            why_it_matters=content["why_it_matters"].strip(),
            try_this=content["try_this"].strip(),
            example_phrase=content["example_phrase"].strip().strip('"«»'),
        )
        return SuggestionOutcome(suggestion, reply.usage)
