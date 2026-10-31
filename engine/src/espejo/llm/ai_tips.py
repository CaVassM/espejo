"""Recomendaciones de cómo usar IA para preparar la próxima clase.

No son consejos genéricos: las observaciones que alimentan el prompt salen de las métricas
reales de la clase (las arma analysis/ai_observations.py), así que las tres propuestas hablan
de lo que pasó en ESTA sesión.

Es la única parte del LLM que no cita la transcripción, porque no afirma nada sobre la clase:
propone material para la siguiente. Por eso no pasa por la validación anti-invento.
"""

import logging
from dataclasses import dataclass

from espejo.config import LlmConfig
from espejo.llm.clients import LlmClient, LlmError, complete_json
from espejo.llm.prompts import AI_TIPS_SCHEMA, build_ai_tips_prompt
from espejo.llm.schemas import AiTip, LlmUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AiTipsOutcome:
    tips: list[AiTip]
    usage: LlmUsage
    note: str | None = None


class AiTipsWriter:
    def __init__(self, client: LlmClient, config: LlmConfig) -> None:
        self.client = client
        self.config = config

    def write(self, observations: list[str], topic: str | None) -> AiTipsOutcome:
        if not observations:
            return AiTipsOutcome([], LlmUsage(), "sin métricas para recomendar uso de IA")

        prompt = build_ai_tips_prompt(observations, topic)
        try:
            reply = complete_json(self.client, prompt, AI_TIPS_SCHEMA, self.config.max_attempts)
        except LlmError as error:
            logger.error("No se pudieron generar las recomendaciones de IA: %s", error)
            return AiTipsOutcome([], LlmUsage(), "el LLM no devolvió recomendaciones válidas")

        tips = []
        for raw in reply.content["tips"]:
            tips.append(
                AiTip(
                    title=raw["title"].strip(),
                    what_for=raw["what_for"].strip(),
                    how=raw["how"].strip(),
                    example_prompt=raw["example_prompt"].strip().strip('"«»'),
                )
            )
        return AiTipsOutcome(tips, reply.usage)
