"""Clasificación por ventanas: preguntas, retroalimentación y regulación, ya validadas."""

import logging
from dataclasses import dataclass, field
from typing import Any

from espejo.config import LlmConfig
from espejo.llm.clients import LlmClient, LlmError, complete_json
from espejo.llm.prompts import CLASSIFICATION_SCHEMA, build_classification_prompt
from espejo.llm.schemas import (
    CandidateQuestion,
    Evidence,
    EvidenceKind,
    JsonReply,
    LlmUsage,
    RejectedEvidence,
    TranscriptLine,
)
from espejo.llm.transcript_view import TranscriptWindow, build_windows
from espejo.llm.validation import (
    deduplicate,
    first_label_per_question,
    parse_mmss,
    validate_evidence,
)

logger = logging.getLogger(__name__)

INVALID_TIME_REASON = "minuto con formato inválido"
UNKNOWN_QUESTION_REASON = "question_id que no es una pregunta candidata de esa ventana"
REPLY_KEYS = {
    EvidenceKind.QUESTION: "questions",
    EvidenceKind.FEEDBACK: "feedback",
    EvidenceKind.REGULATION: "regulation",
    EvidenceKind.PARTICIPATION: "participation",
    EvidenceKind.RESPECT: "respect",
}


@dataclass(frozen=True)
class ParsedItems:
    items: list[Evidence] = field(default_factory=list)
    rejected: list[RejectedEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class ClassificationResult:
    question_labels: dict[int, Evidence]
    feedback: list[Evidence]
    regulation: list[Evidence]
    participation: list[Evidence]
    respect: list[Evidence]
    rejected: list[RejectedEvidence]
    usage: LlmUsage
    failed_windows: int


def evidence_from_raw(raw: dict[str, Any], kind: EvidenceKind, timestamp_s: float) -> Evidence:
    return Evidence(
        kind=kind,
        label=raw["label"],
        timestamp_s=timestamp_s,
        quote=raw["quote"],
        reason=raw["reason"],
        confidence=raw["confidence"],
        question_id=raw.get("question_id"),
    )


def parse_reply(content: dict[str, Any], window: TranscriptWindow) -> ParsedItems:
    """Convierte el JSON en evidencias; descarta minutos ilegibles y preguntas desconocidas."""
    candidate_ids = set()
    for candidate in window.candidates:
        candidate_ids.add(candidate.question_id)

    items = []
    rejected = []
    for kind, key in REPLY_KEYS.items():
        for raw in content[key]:
            timestamp_s = parse_mmss(raw["timestamp"])
            evidence = evidence_from_raw(raw, kind, timestamp_s or 0.0)
            if timestamp_s is None:
                rejected.append(RejectedEvidence(evidence, INVALID_TIME_REASON))
            elif kind == EvidenceKind.QUESTION and evidence.question_id not in candidate_ids:
                rejected.append(RejectedEvidence(evidence, UNKNOWN_QUESTION_REASON))
            else:
                items.append(evidence)
    return ParsedItems(items, rejected)


def items_of_kind(items: list[Evidence], kind: EvidenceKind) -> list[Evidence]:
    selected = []
    for item in items:
        if item.kind == kind:
            selected.append(item)
    return selected


class LlmClassifier:
    def __init__(self, client: LlmClient, config: LlmConfig, definitions: str) -> None:
        self.client = client
        self.config = config
        self.definitions = definitions

    def classify(
        self,
        lines: list[TranscriptLine],
        candidates: list[CandidateQuestion],
        topic: str | None,
    ) -> ClassificationResult:
        accepted: list[Evidence] = []
        rejected: list[RejectedEvidence] = []
        usage = LlmUsage()
        failed_windows = 0

        windows = build_windows(lines, candidates, self.config.windows)
        for position, window in enumerate(windows, start=1):
            logger.info("LLM: ventana %d de %d", position, len(windows))
            try:
                reply = self.ask(window, topic)
            except LlmError as error:
                logger.error("Ventana %d sin clasificar: %s", position, error)
                failed_windows += 1
                continue
            usage = usage.plus(reply.usage)

            parsed = parse_reply(reply.content, window)
            validated = validate_evidence(parsed.items, lines, self.tolerance_s())
            accepted += validated.accepted
            rejected += parsed.rejected + validated.rejected

        return self.merge(accepted, rejected, usage, failed_windows)

    def ask(self, window: TranscriptWindow, topic: str | None) -> JsonReply:
        prompt = build_classification_prompt(window, self.definitions, topic)
        return complete_json(self.client, prompt, CLASSIFICATION_SCHEMA, self.config.max_attempts)

    def tolerance_s(self) -> float:
        return self.config.validation.timestamp_tolerance_s

    def merge(
        self,
        accepted: list[Evidence],
        rejected: list[RejectedEvidence],
        usage: LlmUsage,
        failed_windows: int,
    ) -> ClassificationResult:
        """Une las ventanas: lo que aparece en dos ventanas solapadas cuenta una sola vez."""
        tolerance_s = self.tolerance_s()
        questions = items_of_kind(accepted, EvidenceKind.QUESTION)
        feedback = deduplicate(items_of_kind(accepted, EvidenceKind.FEEDBACK), tolerance_s)
        regulation = deduplicate(items_of_kind(accepted, EvidenceKind.REGULATION), tolerance_s)
        participation = deduplicate(
            items_of_kind(accepted, EvidenceKind.PARTICIPATION), tolerance_s
        )
        respect = deduplicate(items_of_kind(accepted, EvidenceKind.RESPECT), tolerance_s)
        return ClassificationResult(
            question_labels=first_label_per_question(questions),
            feedback=feedback,
            regulation=regulation,
            participation=participation,
            respect=respect,
            rejected=rejected,
            usage=usage,
            failed_windows=failed_windows,
        )
