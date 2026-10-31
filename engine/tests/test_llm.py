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
from espejo.llm.prompts import ANSWERED_QUESTIONS_SCHEMA, EVENTS_SCHEMA
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
    TranscriptLine(10.0, 13.0, "DOCENTE", "¿Cuántos hijos habrá tenido el labrador?"),
    TranscriptLine(14.0, 15.0, "ALUMNO", "Más de uno"),
    TranscriptLine(16.0, 18.0, "DOCENTE", "Muy bien, felicitaciones hija"),
]


class FakeClient:
    """Cliente simulado: nunca llama a un LLM.

    Devuelve las respuestas preparadas en orden; la última se repite. Si la respuesta es un
    JSON con todas las listas, entrega solo las que pide el esquema de cada llamado.
    """

    provider = "fake"
    model = "fake-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def send(self, prompt: str, schema: dict[str, Any]) -> RawReply:
        self.prompts.append(prompt)
        text = self.replies[0]
        if len(self.replies) > 1:
            text = self.replies.pop(0)
        return RawReply(keep_schema_keys(text, schema), LlmUsage(10, 5, calls=1))

    def release(self) -> None:
        self.released = True


def keep_schema_keys(text: str, schema: dict[str, Any]) -> str:
    try:
        content = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(content, dict):
        return text
    kept = {}
    for key, value in content.items():
        if key in schema["properties"]:
            kept[key] = value
    return json.dumps(kept)


def evidence(kind: EvidenceKind, label: str, timestamp_s: float, quote: str) -> Evidence:
    return Evidence(kind, label, timestamp_s, quote, "razón", "alta")


def classification_reply(
    questions: list[dict[str, Any]] | None = None, feedback: list[dict[str, Any]] | None = None
) -> str:
    content = {
        "questions": questions or [],
        "feedback": feedback or [],
        "regulation": [],
        "participation": [],
        "respect": [],
    }
    return json.dumps(content)


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
    lines = [TranscriptLine(0.0, 1.0, "DOCENTE", "a"), TranscriptLine(170.0, 171.0, "DOCENTE", "b")]
    lines.append(TranscriptLine(400.0, 401.0, "DOCENTE", "c"))

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

    reply = complete_json(client, "prompt", EVENTS_SCHEMA, max_attempts=2)

    assert reply.content["feedback"] == []
    assert reply.usage.calls == 2


def test_two_invalid_replies_raise() -> None:
    client = FakeClient(["no", '{"feedback": "mal"}'])
    with pytest.raises(LlmError):
        complete_json(client, "prompt", EVENTS_SCHEMA, max_attempts=2)


# --- Analizador completo con el cliente simulado ---------------------------------------------


def make_segment(start_s: float, text: str, role: SpeakerRole) -> TranscriptSegment:
    return TranscriptSegment(start_s, start_s + 2.0, text, [], -0.3, 0.1, role=role)


def make_question(
    start_s: float, text: str, is_tag: bool = False, answered: bool = True
) -> Question:
    response = None
    if answered:
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
    questions.append(make_question(16.0, "¿ya?", is_tag=True, answered=False))
    reply = classification_reply(
        questions=[
            {
                "question_id": 0,
                "label": "question_open",
                "reason": "pide inferir",
                "confidence": "alta",
            },
            {
                "question_id": 1,
                "label": "question_rhetorical",
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

    # Dos llamados: primero eventos, luego preguntas (con la lista de candidatas)
    assert len(client.prompts) == 2
    assert "0 | [00:10] | ¿Qué pasará?" in client.prompts[1]
    assert "Definiciones de prueba" in client.prompts[0]


def test_candidate_dataclass_keeps_rule_flag() -> None:
    candidate = CandidateQuestion(0, 1.0, "¿ya?", rule_is_tag=True)
    assert candidate.rule_is_tag


def test_participation_and_respect_are_validated_like_the_rest(tmp_path: Path) -> None:
    teacher_line = "A ver, Juanito, tú no has hablado, ¿qué opinas?"
    segments = [
        make_segment(20.0, teacher_line, SpeakerRole.TEACHER),
        make_segment(25.0, "Oye cabezón, cállate", SpeakerRole.STUDENT),
    ]
    invitation = {
        "label": "participation_invitation",
        "quote": "Juanito, tú no has hablado",
        "timestamp": "00:20",
        "reason": "invita a quien no participó",
        "confidence": "alta",
        "addressee": "Juanito",
    }
    nickname = {
        "label": "respect_issue",
        "quote": "Oye cabezón",
        "timestamp": "00:25",
        "reason": "apodo",
        "confidence": "alta",
    }
    invented = dict(nickname, quote="eres un tonto")
    content = {
        "questions": [],
        "feedback": [],
        "regulation": [],
        "participation": [invitation],
        "respect": [nickname, invented],
    }
    analyzer = analyzer_with(FakeClient([json.dumps(content)]), tmp_path)

    analysis = analyzer.classify(segments, [], topic=None)

    assert len(analysis.participation) == 1
    assert len(analysis.respect) == 1
    assert len(analysis.rejected) == 1


def test_quote_inside_a_long_line_uses_the_whole_line_span() -> None:
    # Una línea de 12 s con varias preguntas: el LLM cita el minuto de la tercera (01:45)
    lines = [
        TranscriptLine(
            100.0, 112.0, "DOCENTE", "¿qué pasará? ¿Dónde habrá sucedido? ¿Dónde habrá pasado?"
        )
    ]
    assert quote_exists("¿Dónde habrá pasado?", 108.0, lines, TOLERANCE_S)
    assert not quote_exists("¿Dónde habrá pasado?", 130.0, lines, TOLERANCE_S)


def invitation(quote: str, addressee: str) -> Evidence:
    item = evidence(EvidenceKind.PARTICIPATION, "participation_invitation", 30.0, quote)
    return replace(item, addressee=addressee)


def test_invitation_must_address_one_student() -> None:
    lines = [
        TranscriptLine(30.0, 33.0, "DOCENTE", "¿Qué me han dicho, hija?"),
        TranscriptLine(34.0, 37.0, "DOCENTE", "Niños, ¿cómo será la historia?"),
    ]
    to_one = invitation("¿Qué me han dicho, hija?", "hija")
    to_class = replace(invitation("Niños, ¿cómo será la historia?", "Niños"), timestamp_s=34.0)
    not_in_quote = invitation("¿Qué me han dicho, hija?", "Juanito")
    whole_class_words = load_llm_config().validation.whole_class_words

    result = validate_evidence(
        [to_one, to_class, not_in_quote], lines, TOLERANCE_S, whole_class_words
    )

    assert result.accepted == [to_one]
    assert len(result.rejected) == 2


RHETORICAL_ANSWERED = {
    "question_id": 0,
    "label": "question_rhetorical",
    "reason": "no espera respuesta",
    "confidence": "media",
}


def answered_case() -> tuple[list[TranscriptSegment], list[Question]]:
    segments = [
        make_segment(10.0, "¿qué sucederá en el cuento?", SpeakerRole.TEACHER),
        make_segment(13.0, "Que los hijos estarían ayudando", SpeakerRole.STUDENT),
    ]
    return segments, [make_question(10.0, "¿qué sucederá en el cuento?")]


def test_answered_question_marked_rhetorical_is_asked_again(tmp_path: Path) -> None:
    segments, questions = answered_case()
    open_label = dict(RHETORICAL_ANSWERED, label="question_open", reason="pide predecir")
    replies = [
        classification_reply(),
        classification_reply(questions=[RHETORICAL_ANSWERED]),
        classification_reply(questions=[open_label]),
    ]
    client = FakeClient(replies)
    analyzer = analyzer_with(client, tmp_path)

    analysis = analyzer.classify(segments, questions, topic=None)

    assert analysis.question_labels[0].label == "question_open"
    assert len(client.prompts) == 3
    assert "PREGUNTAS QUE UN ALUMNO RESPONDIÓ" in client.prompts[2]


def test_answered_question_never_stays_rhetorical(tmp_path: Path) -> None:
    # El segundo llamado insiste en "retórica" (inválido en su esquema): se rechaza igual
    segments, questions = answered_case()
    client = FakeClient([classification_reply(questions=[RHETORICAL_ANSWERED])])
    analyzer = analyzer_with(client, tmp_path)

    analysis = analyzer.classify(segments, questions, topic=None)

    assert analysis.question_labels == {}
    assert analysis.rejected[0].reason == "un alumno respondió: no puede ser retórica"


def test_answered_questions_schema_only_allows_open_or_closed() -> None:
    item_schema = ANSWERED_QUESTIONS_SCHEMA["properties"]["questions"]["items"]
    assert item_schema["properties"]["label"]["enum"] == ["question_open", "question_closed"]
