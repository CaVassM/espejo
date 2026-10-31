"""Cliente de Cohere con un SDK simulado: no llama a la API ni necesita la key."""

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

cohere_errors = pytest.importorskip("cohere.errors")

from espejo.config import load_llm_config  # noqa: E402
from espejo.llm.clients import CohereClient, LlmError, complete_json, schema_for_cohere  # noqa: E402
from espejo.llm.prompts import SUGGESTION_SCHEMA  # noqa: E402

LLM_CONFIG = load_llm_config()
COHERE = LLM_CONFIG.cohere
REPLY = {
    "why_it_matters": "Pensar toma tiempo.",
    "try_this": "Espera.",
    "example_phrase": "Piensa.",
}


def response(content: dict[str, Any]) -> SimpleNamespace:
    text_item = SimpleNamespace(type="text", text=json.dumps(content))
    usage = SimpleNamespace(
        billed_units=SimpleNamespace(input_tokens=120.0, output_tokens=30.0), tokens=None
    )
    return SimpleNamespace(
        finish_reason="COMPLETE", message=SimpleNamespace(content=[text_item]), usage=usage
    )


class FakeSdk:
    """Devuelve las respuestas (o lanza los errores) en orden y guarda cada llamado."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeTime:
    def __init__(self) -> None:
        self.now_s = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now_s += seconds


def client_with(outcomes: list[Any]) -> tuple[CohereClient, FakeSdk, FakeTime]:
    sdk = FakeSdk(outcomes)
    fake_time = FakeTime()
    client = CohereClient(COHERE, "key", sdk, sleep=fake_time.sleep, clock=fake_time.clock)
    return client, sdk, fake_time


def test_schema_drops_what_cohere_does_not_support() -> None:
    cleaned = schema_for_cohere(SUGGESTION_SCHEMA)
    assert "minLength" not in json.dumps(cleaned)
    assert cleaned["required"] == SUGGESTION_SCHEMA["required"]
    assert set(cleaned["properties"]) == set(SUGGESTION_SCHEMA["properties"])


def test_send_asks_for_json_with_the_schema_and_counts_tokens() -> None:
    client, sdk, _ = client_with([response(REPLY)])

    reply = complete_json(client, "Escribe el consejo.", SUGGESTION_SCHEMA, max_attempts=2)

    call = sdk.calls[0]
    assert call["model"] == "command-a-03-2025"
    assert call["response_format"]["type"] == "json_object"
    assert "minLength" not in json.dumps(call["response_format"]["schema"])
    assert "JSON" in call["messages"][0]["content"]
    assert call["temperature"] == 0
    assert reply.content == REPLY
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (120, 30)


def test_calls_are_spaced_to_respect_the_rate_limit() -> None:
    client, _, fake_time = client_with([response(REPLY), response(REPLY)])

    client.send("uno", SUGGESTION_SCHEMA)
    client.send("dos", SUGGESTION_SCHEMA)

    # 20 llamados por minuto: al menos 3 s entre uno y otro
    assert fake_time.slept == [pytest.approx(3.0)]


def test_too_many_requests_waits_and_retries() -> None:
    busy = cohere_errors.TooManyRequestsError(body=None)
    client, sdk, fake_time = client_with([busy, response(REPLY)])

    reply = client.send("hola", SUGGESTION_SCHEMA)

    assert json.loads(reply.text) == REPLY
    assert len(sdk.calls) == 2
    assert COHERE.rate_limit_wait_s in fake_time.slept


def test_api_errors_become_llm_errors() -> None:
    client, _, _ = client_with([cohere_errors.UnauthorizedError(body=None)])
    with pytest.raises(LlmError):
        client.send("hola", SUGGESTION_SCHEMA)


def test_cohere_uses_longer_windows() -> None:
    assert replace(LLM_CONFIG, provider="cohere").active_windows().window_s == 600
    assert replace(LLM_CONFIG, provider="ollama").active_windows().window_s == 180
