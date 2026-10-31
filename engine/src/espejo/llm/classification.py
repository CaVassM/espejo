"""Clasificación por ventanas: eventos y preguntas, ya validados contra la transcripción."""

import logging
from dataclasses import dataclass, field
from typing import Any

from espejo.audio.schemas import QuestionCategory
from espejo.config import LlmConfig
from espejo.llm.clients import LlmClient, LlmError, complete_json
from espejo.llm.prompts import (
    ANSWERED_QUESTIONS_SCHEMA,
    EVENTS_SCHEMA,
    QUESTIONS_SCHEMA,
    build_answered_questions_prompt,
    build_events_prompt,
    build_questions_prompt,
)
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
ANSWERED_RHETORICAL_REASON = "un alumno respondió: no puede ser retórica"
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
        addressee=raw.get("addressee"),
    )


def question_from_raw(raw: dict[str, Any], candidate: CandidateQuestion) -> Evidence:
    """La cita y el minuto son los de la pregunta candidata: el LLM solo da la clase."""
    return Evidence(
        kind=EvidenceKind.QUESTION,
        label=raw["label"],
        timestamp_s=candidate.start_s,
        quote=candidate.text,
        reason=raw["reason"],
        confidence=raw["confidence"],
        question_id=candidate.question_id,
    )


def unknown_question(raw: dict[str, Any]) -> Evidence:
    return Evidence(
        EvidenceKind.QUESTION,
        raw["label"],
        0.0,
        "",
        raw["reason"],
        raw["confidence"],
        question_id=raw["question_id"],
    )


def parse_reply(content: dict[str, Any], window: TranscriptWindow) -> ParsedItems:
    """Convierte el JSON en evidencias; descarta minutos ilegibles y preguntas desconocidas."""
    candidates_by_id = {}
    for candidate in window.candidates:
        candidates_by_id[candidate.question_id] = candidate

    items = []
    rejected = []
    for raw in content.get(REPLY_KEYS[EvidenceKind.QUESTION], []):
        candidate = candidates_by_id.get(raw["question_id"])
        if candidate is None:
            rejected.append(RejectedEvidence(unknown_question(raw), UNKNOWN_QUESTION_REASON))
        elif candidate.student_answered and raw["label"] == QuestionCategory.RHETORICAL:
            evidence = question_from_raw(raw, candidate)
            rejected.append(RejectedEvidence(evidence, ANSWERED_RHETORICAL_REASON))
        else:
            items.append(question_from_raw(raw, candidate))

    for kind, key in REPLY_KEYS.items():
        if kind == EvidenceKind.QUESTION:
            continue
        for raw in content.get(key, []):
            timestamp_s = parse_mmss(raw["timestamp"])
            evidence = evidence_from_raw(raw, kind, timestamp_s or 0.0)
            if timestamp_s is None:
                rejected.append(RejectedEvidence(evidence, INVALID_TIME_REASON))
            else:
                items.append(evidence)
    return ParsedItems(items, rejected)


def answered_marked_rhetorical(
    raw_questions: list[dict[str, Any]], candidates: list[CandidateQuestion]
) -> list[CandidateQuestion]:
    rhetorical_ids = set()
    for raw in raw_questions:
        if raw["label"] == QuestionCategory.RHETORICAL:
            rhetorical_ids.add(raw["question_id"])
    selected = []
    for candidate in candidates:
        if candidate.student_answered and candidate.question_id in rhetorical_ids:
            selected.append(candidate)
    return selected


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

        windows = build_windows(lines, candidates, self.config.active_windows())
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
            validated = validate_evidence(
                parsed.items, lines, self.tolerance_s(), self.config.validation.whole_class_words
            )
            accepted += validated.accepted
            rejected += parsed.rejected + validated.rejected

        return self.merge(accepted, rejected, usage, failed_windows)

    def ask(self, window: TranscriptWindow, topic: str | None) -> JsonReply:
        """Dos llamados por ventana (eventos y preguntas), juntados en un solo contenido."""
        attempts = self.config.max_attempts
        events_prompt = build_events_prompt(window, self.definitions, topic)
        events = complete_json(self.client, events_prompt, EVENTS_SCHEMA, attempts)
        if not window.candidates:
            return events

        questions_prompt = build_questions_prompt(window, self.definitions, topic)
        questions = complete_json(self.client, questions_prompt, QUESTIONS_SCHEMA, attempts)
        questions = self.reclassify_answered(window, topic, questions)
        content = dict(events.content)
        content.update(questions.content)
        return JsonReply(content, events.usage.plus(questions.usage))

    def reclassify_answered(
        self, window: TranscriptWindow, topic: str | None, reply: JsonReply
    ) -> JsonReply:
        """Si marcó retórica una pregunta que un alumno respondió, la vuelve a pedir solo
        entre abierta y cerrada. Si ese llamado falla, parse_reply la rechaza igual."""
        raw_questions = reply.content.get(REPLY_KEYS[EvidenceKind.QUESTION], [])
        answered = answered_marked_rhetorical(raw_questions, window.candidates)
        if not answered:
            return reply

        prompt = build_answered_questions_prompt(window, self.definitions, topic, answered)
        try:
            second = complete_json(
                self.client, prompt, ANSWERED_QUESTIONS_SCHEMA, self.config.max_attempts
            )
        except LlmError as error:
            logger.warning("No se pudo reclasificar preguntas respondidas: %s", error)
            return reply

        answered_ids = set()
        for candidate in answered:
            answered_ids.add(candidate.question_id)
        kept = []
        for raw in raw_questions:
            if raw["question_id"] not in answered_ids:
                kept.append(raw)
        for raw in second.content["questions"]:
            if raw["question_id"] in answered_ids:
                kept.append(raw)
        return JsonReply({"questions": kept}, reply.usage.plus(second.usage))

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
