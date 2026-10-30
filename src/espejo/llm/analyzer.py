"""Orquesta el LLM: clasifica la transcripción y aplica sus clases a las preguntas del audio."""

import logging
from dataclasses import replace

from espejo.audio.schemas import Question, QuestionCategory, TranscriptSegment
from espejo.config import LlmConfig
from espejo.llm.classification import LlmClassifier
from espejo.llm.clients import ClientChoice, build_llm_client
from espejo.llm.schemas import Evidence, LlmAnalysis
from espejo.llm.suggestion import SuggestionWriter
from espejo.llm.transcript_view import candidates_from_questions, lines_from_segments

logger = logging.getLogger(__name__)

NO_LLM_PREFIX = "sin análisis LLM: "


def without_llm(reason: str) -> LlmAnalysis:
    """El pipeline sigue con las reglas de respaldo y el reporte lo dice."""
    logger.info("%s%s", NO_LLM_PREFIX, reason)
    return LlmAnalysis(used=False, note=NO_LLM_PREFIX + reason)


def apply_question_labels(questions: list[Question], labels: dict[int, Evidence]) -> list[Question]:
    """Las preguntas que el LLM clasificó usan su clase; las demás siguen con la regla."""
    labeled = []
    for question_id, question in enumerate(questions):
        evidence = labels.get(question_id)
        if evidence is None:
            labeled.append(question)
        else:
            labeled.append(replace(question, llm_category=QuestionCategory(evidence.label)))
    return labeled


class LlmAnalyzer:
    def __init__(self, config: LlmConfig, choice: ClientChoice | None = None) -> None:
        """choice permite inyectar un cliente (p. ej. un mock en los tests)."""
        self.config = config
        self.choice = choice or build_llm_client(config)

    def read_definitions(self) -> str | None:
        path = self.config.definitions_path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def classify(
        self,
        segments: list[TranscriptSegment],
        questions: list[Question],
        topic: str | None,
    ) -> LlmAnalysis:
        client = self.choice.client
        if client is None:
            return without_llm(self.choice.note)
        definitions = self.read_definitions()
        if definitions is None:
            return without_llm(f"falta {self.config.definitions_path.name}")
        if not segments:
            return without_llm("no hay transcripción")

        classifier = LlmClassifier(client, self.config, definitions)
        result = classifier.classify(
            lines_from_segments(segments), candidates_from_questions(questions), topic
        )
        note = f"{client.provider} con {client.model}"
        if result.failed_windows:
            note += f" ({result.failed_windows} ventanas sin clasificar)"
        return LlmAnalysis(
            used=True,
            note=note,
            provider=client.provider,
            model=client.model,
            question_labels=result.question_labels,
            feedback=result.feedback,
            regulation=result.regulation,
            participation=result.participation,
            respect=result.respect,
            rejected=result.rejected,
            usage=result.usage,
        )

    def add_suggestion(
        self,
        analysis: LlmAnalysis,
        evidence_text: str,
        evidence_times: list[float],
        topic: str | None,
    ) -> LlmAnalysis:
        client = self.choice.client
        if not analysis.used or client is None:
            return analysis

        outcome = SuggestionWriter(client, self.config).write(evidence_text, evidence_times, topic)
        note = analysis.note
        if outcome.note:
            note += f"; {outcome.note}"
        return replace(
            analysis,
            suggestion=outcome.suggestion,
            usage=analysis.usage.plus(outcome.usage),
            note=note,
        )
