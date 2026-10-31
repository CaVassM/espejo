import json
from typing import Any

from espejo.analysis.focus import FocusThresholds, choose_focus
from espejo.analysis.schemas import RubricMetric
from espejo.audio.schemas import Question, QuestionCategory, SpeakerRole
from espejo.config import load_llm_config
from espejo.llm.schemas import (
    Evidence,
    EvidenceKind,
    LlmUsage,
    RawReply,
    SuggestionFocus,
    TranscriptLine,
    Weakness,
)
from espejo.llm.suggestion import SuggestionWriter

THRESHOLDS = FocusThresholds(wait_ok_s=3.0, monitoring=0.25, min_open_question_ratio=0.5)


def metric(name: str, value: float) -> RubricMetric:
    return RubricMetric(name, value, ["R2"], "prueba")


def question(start_s: float, text: str, wait_s: float, category: QuestionCategory) -> Question:
    return Question(
        text=text,
        start_s=start_s,
        question_end_s=start_s + 1.0,
        speaker="SPEAKER_00",
        role=SpeakerRole.TEACHER,
        reliable=True,
        llm_category=category,
        wait_time_s=wait_s,
        wait_time_ok=wait_s >= THRESHOLDS.wait_ok_s,
    )


def feedback(timestamp_s: float, label: str, quote: str) -> Evidence:
    return Evidence(EvidenceKind.FEEDBACK, label, timestamp_s, quote, "razón", "alta")


QUESTIONS = [
    question(60.0, "¿ya?", 0.5, QuestionCategory.RHETORICAL),
    question(65.0, "¿Cuántos hijos tenía?", 0.8, QuestionCategory.CLOSED),
    question(82.0, "¿De qué tratará?", 1.1, QuestionCategory.OPEN),
]
ELEMENTAL = [
    feedback(197.0, "feedback_elemental", "Muy bien hija"),
    feedback(131.0, "feedback_elemental", "Muy bien, felicitaciones hija"),
]


def test_short_wait_comes_first_and_cites_an_open_question() -> None:
    metrics = [metric("mean_wait_time_s", 1.27), metric("monitoring_ratio", 0.2)]

    focus = choose_focus(metrics, QUESTIONS, ELEMENTAL, THRESHOLDS)

    assert focus is not None
    assert focus.weakness == Weakness.SHORT_WAIT
    assert focus.quote == "¿De qué tratará?"
    assert "01:22" in focus.what_we_saw
    assert "1,1 s" in focus.what_we_saw
    assert "1,3 s" in focus.what_we_saw


def test_only_elemental_feedback_is_next_and_cites_the_first_one() -> None:
    metrics = [metric("mean_wait_time_s", 3.5), metric("monitoring_ratio", 0.2)]

    focus = choose_focus(metrics, QUESTIONS, ELEMENTAL, THRESHOLDS)

    assert focus is not None
    assert focus.weakness == Weakness.ELEMENTAL_FEEDBACK
    assert focus.timestamp_s == 131.0
    assert "Las 2 retroalimentaciones" in focus.what_we_saw


def test_one_descriptive_feedback_skips_to_monitoring() -> None:
    metrics = [metric("mean_wait_time_s", 3.5), metric("monitoring_ratio", 0.2)]
    mixed = [*ELEMENTAL, feedback(250.0, "feedback_descriptive", "Bien, explicaste por qué")]

    focus = choose_focus(metrics, QUESTIONS, mixed, THRESHOLDS)

    assert focus is not None
    assert focus.weakness == Weakness.LOW_MONITORING
    assert focus.timestamp_s is None
    assert "20 %" in focus.what_we_saw


def test_mostly_closed_questions_is_last() -> None:
    metrics = [
        metric("mean_wait_time_s", 3.5),
        metric("monitoring_ratio", 0.4),
        metric("open_question_ratio", 0.3),
    ]

    focus = choose_focus(metrics, QUESTIONS, [], THRESHOLDS)

    assert focus is not None
    assert focus.weakness == Weakness.CLOSED_QUESTIONS
    assert focus.quote == "¿Cuántos hijos tenía?"


def test_no_weakness_means_no_focus() -> None:
    metrics = [
        metric("mean_wait_time_s", 3.5),
        metric("monitoring_ratio", 0.4),
        metric("open_question_ratio", 0.8),
    ]
    assert choose_focus(metrics, QUESTIONS, [], THRESHOLDS) is None


class SuggestionClient:
    """Cliente simulado que devuelve siempre la misma respuesta y guarda los prompts."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, content: dict[str, Any]) -> None:
        self.text = json.dumps(content)
        self.prompts: list[str] = []

    def send(self, prompt: str, schema: dict[str, Any]) -> RawReply:
        self.prompts.append(prompt)
        return RawReply(self.text, LlmUsage(10, 5, calls=1))

    def release(self) -> None:
        """Nada que liberar."""


def test_suggestion_has_three_parts_and_keeps_the_real_quote() -> None:
    focus = SuggestionFocus(
        Weakness.SHORT_WAIT,
        'En el minuto 01:22 preguntaste "¿De qué tratará?" y la pausa fue de 1,1 s.',
        82.0,
        "¿De qué tratará?",
    )
    lines = [
        TranscriptLine(82.0, 84.0, "DOCENTE", "¿De qué tratará?"),
        TranscriptLine(300.0, 302.0, "DOCENTE", "Lejos del momento citado"),
    ]
    client = SuggestionClient(
        {
            "why_it_matters": "Pensar toma tiempo.",
            "try_this": "Cuenta hasta tres en silencio.",
            "example_phrase": '"Piénsalo un momento, luego me cuentas"',
        }
    )
    config = load_llm_config()

    outcome = SuggestionWriter(client, config).write(focus, lines, topic="Cuento")

    assert outcome.suggestion is not None
    text = outcome.suggestion.as_text()
    assert text.startswith("Lo que vimos: En el minuto 01:22")
    assert "Por qué importa: Pensar toma tiempo." in text
    assert 'Por ejemplo: "Piénsalo un momento, luego me cuentas"' in text
    assert "¿De qué tratará?" in client.prompts[0]
    assert "Lejos del momento citado" not in client.prompts[0]
