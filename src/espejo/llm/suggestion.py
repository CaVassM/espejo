"""Un solo consejo para la próxima clase, basado solo en la evidencia de la ficha."""

import logging
from dataclasses import dataclass

from espejo.config import LlmConfig
from espejo.llm.clients import LlmClient, LlmError, complete_json
from espejo.llm.prompts import SUGGESTION_SCHEMA, build_suggestion_prompt
from espejo.llm.schemas import LlmUsage, Suggestion
from espejo.llm.validation import parse_mmss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuggestionOutcome:
    suggestion: Suggestion | None
    usage: LlmUsage
    note: str | None = None


def cites_known_minute(timestamp_s: float, evidence_times: list[float], tolerance_s: float) -> bool:
    """El minuto del consejo debe ser el de alguna evidencia: así no inventa un momento."""
    for evidence_time_s in evidence_times:
        if abs(evidence_time_s - timestamp_s) <= tolerance_s:
            return True
    return False


class SuggestionWriter:
    def __init__(self, client: LlmClient, config: LlmConfig) -> None:
        self.client = client
        self.config = config

    def write(
        self, evidence_text: str, evidence_times: list[float], topic: str | None
    ) -> SuggestionOutcome:
        if not evidence_times:
            return SuggestionOutcome(
                None, LlmUsage(), "sin evidencia con minuto para basar un consejo"
            )

        prompt = build_suggestion_prompt(evidence_text, topic)
        try:
            reply = complete_json(self.client, prompt, SUGGESTION_SCHEMA, self.config.max_attempts)
        except LlmError as error:
            logger.error("No se pudo generar el consejo: %s", error)
            return SuggestionOutcome(None, LlmUsage(), "el LLM no devolvió un consejo válido")

        timestamp_s = parse_mmss(reply.content["timestamp"])
        tolerance_s = self.config.validation.timestamp_tolerance_s
        if timestamp_s is None or not cites_known_minute(timestamp_s, evidence_times, tolerance_s):
            return SuggestionOutcome(
                None, reply.usage, "consejo rechazado: cita un minuto sin evidencia"
            )
        return SuggestionOutcome(Suggestion(reply.content["suggestion"], timestamp_s), reply.usage)
