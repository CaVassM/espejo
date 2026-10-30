import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from espejo.audio.metrics import AudioMetricsCalculator
from espejo.audio.schemas import (
    Question,
    QuestionCategory,
    ResponseCertainty,
    SpeakerRole,
    StudentResponse,
    TranscriptSegment,
)
from espejo.config import load_llm_config
from espejo.llm.analyzer import LlmAnalyzer, apply_question_labels
from espejo.llm.clients import ClientChoice, LlmError, complete_json
from espejo.llm.prompts import CLASSIFICATION_SCHEMA
from espejo.llm.schemas import (
    CandidateQuestion,
    Evidence,
    EvidenceKind,
    LlmUsage,
    RawReply,
    TranscriptLine,
)
from espejo.llm.transcript_view import build_windows
from espejo.llm.validation import deduplicate, parse_mmss, quote_exists, validate_evidence

TOLERANCE_S = 2.0
LINES = [
    TranscriptLine(10.0, "DOCENTE", "¿Cuántos hijos habrá tenido el labrador?"),
    TranscriptLine(14.0, "ALUMNO", "Más de uno"),
    TranscriptLine(16.0, "DOCENTE", "Muy bien, felicitaciones hija"),
]


class FakeClient:
    """Cliente simulado: devuelve las respuestas preparadas, en orden. Nunca llama a un LLM."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def send(self, prompt: str, schema: dict[str, Any]) -> RawReply:
        self.prompts.append(prompt)
        return RawReply(self.replies.pop(0), LlmUsage(input_tokens=10, output_tokens=5, calls=1))


def evidence(kind: EvidenceKind, label: str, timestamp_s: float, quote: str) -> Evidence:
    return Evidence(kind, label, timestamp_s, quote, "razón", "alta")


def classification_reply(
    questions: list[dict[str, Any]] | None = None, feedback: list[dict[str, Any]] | None = None
) -> str:
    return json.dumps({"questions": questions or [], "feedback": feedback or [], "regulation": []})


# --- Validación anti-invento -----------------------------------------------------------------


def test_parse_mmss() -> None:
    assert parse_mmss("03:12") == 192.0
    assert parse_mmss("[1:05]") == 65.0
    assert parse_mmss("minuto tres") is None


def test_quote_that_exists_at_its_minute_is_accepted() -> None:
    assert quote_exists("felicitaciones hija", 16.0, LINES, TOLERANCE_S)


def test_quote_ignores_case_accents_and_punctuation() -> None:
    assert quote_exists("cuantos hijos habra tenido", 11.0, LINES, TOLERANCE_S)


def test_invented_quote_is_rejected() -> None:
    assert not quote_exists("excelente razonamiento", 16.0, LINES, TOLERANCE_S)


def test_real_quote_at_wrong_minute_is_rejected() -> None:
    # La cita existe a los 16 s, pero el LLM dice que fue a los 40 s
    assert not quote_exists("felicitaciones hija", 40.0, LINES, TOLERANCE_S)


def test_invalid_class_is_rejected() -> None:
    items = [
        evidence(EvidenceKind.FEEDBACK, "feedback_genial", 16.0, "felicitaciones hija"),
        evidence(EvidenceKind.FEEDBACK, "feedback_elemental", 16.0, "felicitaciones hija"),
    ]

    result = validate_evidence(items, LINES, TOLERANCE_S)

    assert len(result.accepted) == 1
    assert result.rejected[0].evidence.label == "feedback_genial"


# --- Ventanas -----------------------------------------------------------------------------


def test_windows_overlap_and_skip_empty_ranges() -> None:
    config = replace(load_llm_config().windows, window_s=180, overlap_s=20)
    lines = [TranscriptLine(0.0, "DOCENTE", "a"), TranscriptLine(170.0, "DOCENTE", "b")]
    lines.append(TranscriptLine(400.0, "DOCENTE", "c"))

    windows = build_windows(lines, [], config)

    starts = [window.start_s for window in windows]
    assert starts == [0.0, 160.0, 320.0]
    # La línea de los 170 s cae en dos ventanas: por eso hay que unir sin duplicados
    assert len(windows[0].lines) == 2 and len(windows[1].lines) == 1


def test_overlapping_windows_do_not_duplicate_evidence() -> None:
    same = evidence(EvidenceKind.FEEDBACK, "feedback_elemental", 16.0, "felicitaciones hija")
    repeated = replace(same, timestamp_s=17.0)

    assert deduplicate([same, repeated], TOLERANCE_S) == [same]


# --- Cliente: JSON válido y reintento --------------------------------------------------------


def test_invalid_json_is_retried_once() -> None:
    client = FakeClient(["esto no es json", classification_reply()])

    reply = complete_json(client, "prompt", CLASSIFICATION_SCHEMA, max_attempts=2)

    assert reply.content["questions"] == []
    assert reply.usage.calls == 2


def test_two_invalid_replies_raise() -> None:
    client = FakeClient(["no", '{"questions": "mal"}'])
    with pytest.raises(LlmError):
        complete_json(client, "prompt", CLASSIFICATION_SCHEMA, max_attempts=2)


# --- Analizador completo con el cliente simulado ---------------------------------------------


def make_segment(start_s: float, text: str, role: SpeakerRole) -> TranscriptSegment:
    return TranscriptSegment(start_s, start_s + 2.0, text, [], -0.3, 0.1, role=role)


def make_question(start_s: float, text: str, is_tag: bool = False) -> Question:
    response = StudentResponse(start_s + 3, start_s + 4, "SPEAKER_01", ResponseCertainty.HIGH)
    return Question(
        text=text,
        start_s=start_s,
        question_end_s=start_s + 2,
        speaker="SPEAKER_00",
        role=SpeakerRole.TEACHER,
        reliable=True,
        speaker_known=True,
        is_tag=is_tag,
        wait_time_s=1.0,
        wait_time_ok=False,
        response=response,
    )


def analyzer_with(client: FakeClient | None, tmp_path: Path) -> LlmAnalyzer:
    definitions = tmp_path / "definitions.md"
    definitions.write_text("Definiciones de prueba", encoding="utf-8")
    config = replace(load_llm_config(), definitions_path=definitions)
    return LlmAnalyzer(config, ClientChoice(client, "cliente de prueba"))


def test_without_llm_the_pipeline_uses_the_rules(tmp_path: Path) -> None:
    analyzer = analyzer_with(None, tmp_path)

    analysis = analyzer.classify([], [], topic=None)

    assert analysis.used is False
    assert analysis.note.startswith("sin análisis LLM")


def test_llm_classes_replace_the_rule_in_metrics(tmp_path: Path) -> None:
    segments = [
        make_segment(10.0, "¿Cuántos hijos habrá tenido el labrador?", SpeakerRole.TEACHER),
        make_segment(14.0, "Más de uno", SpeakerRole.STUDENT),
        make_segment(16.0, "Muy bien, ¿ya?", SpeakerRole.TEACHER),
    ]
    questions = [make_question(10.0, "¿Cuántos hijos habrá tenido el labrador?")]
    questions.append(make_question(16.0, "¿ya?", is_tag=True))
    reply = classification_reply(
        questions=[
            {
                "question_id": 0,
                "label": "question_open",
                "quote": "Cuántos hijos habrá tenido",
                "timestamp": "00:10",
                "reason": "pide inferir",
                "confidence": "alta",
            },
            {
                "question_id": 1,
                "label": "question_rhetorical",
                "quote": "¿ya?",
                "timestamp": "00:16",
                "reason": "muletilla",
                "confidence": "alta",
            },
        ],
        feedback=[
            {
                "label": "feedback_elemental",
                "quote": "Muy bien",
                "timestamp": "00:16",
                "reason": "solo dice que está bien",
                "confidence": "media",
            },
            {
                "label": "feedback_descriptive",
                "quote": "Excelente, explicaste por qué son dos",
                "timestamp": "00:16",
                "reason": "inventada",
                "confidence": "alta",
            },
        ],
    )
    analyzer = analyzer_with(FakeClient([reply]), tmp_path)

    analysis = analyzer.classify(segments, questions, topic="Cuento")
    labeled = apply_question_labels(questions, analysis.question_labels)
    metrics = AudioMetricsCalculator().calculate(segments, labeled, [], 60.0, True)

    assert analysis.used is True
    assert labeled[0].llm_category == QuestionCategory.OPEN
    assert len(analysis.feedback) == 1
    assert len(analysis.rejected) == 1
    assert metrics.teacher_question_count == 1
    assert metrics.open_question_ratio == pytest.approx(1.0)


def test_unknown_question_id_is_rejected(tmp_path: Path) -> None:
    segments = [make_segment(10.0, "¿Qué pasará?", SpeakerRole.TEACHER)]
    questions = [make_question(10.0, "¿Qué pasará?")]
    reply = classification_reply(
        questions=[
            {
                "question_id": 7,
                "label": "question_open",
                "quote": "Qué pasará",
                "timestamp": "00:10",
                "reason": "x",
                "confidence": "alta",
            }
        ]
    )
    analyzer = analyzer_with(FakeClient([reply]), tmp_path)

    analysis = analyzer.classify(segments, questions, topic=None)

    assert analysis.question_labels == {}
    assert len(analysis.rejected) == 1


def test_candidate_questions_reach_the_prompt(tmp_path: Path) -> None:
    client = FakeClient([classification_reply()])
    analyzer = analyzer_with(client, tmp_path)
    segments = [make_segment(10.0, "¿Qué pasará?", SpeakerRole.TEACHER)]

    analyzer.classify(segments, [make_question(10.0, "¿Qué pasará?")], topic="Cuento")

    assert "0 | [00:10] | ¿Qué pasará?" in client.prompts[0]
    assert "Definiciones de prueba" in client.prompts[0]


def test_candidate_dataclass_keeps_rule_flag() -> None:
    candidate = CandidateQuestion(0, 1.0, "¿ya?", rule_is_tag=True)
    assert candidate.rule_is_tag
