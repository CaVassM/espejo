"""Un cliente pequeño por proveedor, con la misma interfaz: prompt + esquema JSON -> respuesta."""

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

import jsonschema

from espejo.common.env import read_secret
from espejo.common.gpu import read_gpu_memory
from espejo.config import AnthropicConfig, LlmConfig, OllamaConfig
from espejo.llm.schemas import JsonReply, LlmUsage, RawReply

logger = logging.getLogger(__name__)

PROVIDER_OLLAMA = "ollama"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_NONE = "none"
AUTO_MODEL = "auto"
OLLAMA_PROBE_TIMEOUT_S = 3
NANOSECONDS_PER_SECOND = 1_000_000_000
TOKENS_PER_MILLION = 1_000_000
REFUSAL_STOP_REASON = "refusal"
SERVER_FALLBACK_MODE = "default"


class LlmError(RuntimeError):
    """El proveedor no devolvió un JSON válido después de los reintentos."""


class LlmClient(Protocol):
    provider: str
    model: str

    def send(self, prompt: str, schema: dict[str, Any]) -> RawReply: ...


@dataclass(frozen=True)
class ClientChoice:
    """El cliente elegido, o None con el motivo por el que no hay LLM."""

    client: LlmClient | None
    note: str


def complete_json(
    client: LlmClient, prompt: str, schema: dict[str, Any], max_attempts: int
) -> JsonReply:
    """Pide la respuesta y la valida contra el esquema; si no es válida, reintenta."""
    usage = LlmUsage()
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        reply = client.send(prompt, schema)
        usage = usage.plus(reply.usage)
        try:
            content = json.loads(reply.text)
            jsonschema.validate(content, schema)
            return JsonReply(content, usage)
        except (json.JSONDecodeError, jsonschema.ValidationError) as error:
            last_error = str(error).splitlines()[0]
            logger.warning("Respuesta inválida del LLM (intento %d): %s", attempt, last_error)
    raise LlmError(f"El LLM no devolvió un JSON válido: {last_error}")


class OllamaClient:
    """Ollama local, con salida estructurada (parámetro "format") y temperature 0."""

    provider = PROVIDER_OLLAMA

    def __init__(self, config: OllamaConfig, model: str) -> None:
        self.config = config
        self.model = model

    def send(self, prompt: str, schema: dict[str, Any]) -> RawReply:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.context_tokens,
            },
        }
        started = time.perf_counter()
        response = post_json(f"{self.config.host}/api/chat", body, self.config.request_timeout_s)
        usage = LlmUsage(
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            seconds=time.perf_counter() - started,
            calls=1,
        )
        return RawReply(response["message"]["content"], usage)


class AnthropicClient:
    """Claude por el SDK oficial, con salida estructurada (output_config.format)."""

    provider = PROVIDER_ANTHROPIC

    def __init__(self, config: AnthropicConfig, api_key: str) -> None:
        # Import local: el SDK solo hace falta si se elige este proveedor
        import anthropic

        self.config = config
        self.model = config.model
        self.sdk_client = anthropic.Anthropic(api_key=api_key)

    def send(self, prompt: str, schema: dict[str, Any]) -> RawReply:
        started = time.perf_counter()
        response = self.sdk_client.beta.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            betas=[self.config.fallback_beta],
            fallbacks=SERVER_FALLBACK_MODE,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == REFUSAL_STOP_REASON:
            raise LlmError("Claude rechazó la solicitud (refusal).")

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return RawReply(text, self.usage_from(response.usage, time.perf_counter() - started))

    def usage_from(self, sdk_usage: Any, seconds: float) -> LlmUsage:
        cost_usd = (
            sdk_usage.input_tokens * self.config.input_usd_per_million_tokens
            + sdk_usage.output_tokens * self.config.output_usd_per_million_tokens
        ) / TOKENS_PER_MILLION
        return LlmUsage(
            input_tokens=sdk_usage.input_tokens,
            output_tokens=sdk_usage.output_tokens,
            seconds=seconds,
            cost_usd=cost_usd,
            calls=1,
        )


def post_json(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout_s: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError):
        return None


def choose_ollama_model(config: OllamaConfig) -> str:
    """Con "auto", el modelo grande solo si la VRAM real alcanza."""
    if config.model != AUTO_MODEL:
        return config.model
    memory = read_gpu_memory()
    if memory is not None and memory.total_gb >= config.min_vram_gb_for_large:
        return config.large_model
    return config.small_model


def build_ollama_client(config: OllamaConfig) -> ClientChoice:
    if get_json(f"{config.host}/api/version", OLLAMA_PROBE_TIMEOUT_S) is None:
        return ClientChoice(None, f"Ollama no responde en {config.host}")

    model = choose_ollama_model(config)
    tags = get_json(f"{config.host}/api/tags", OLLAMA_PROBE_TIMEOUT_S) or {}
    installed = []
    for entry in tags.get("models", []):
        installed.append(entry["name"])
    if model not in installed:
        return ClientChoice(None, f"falta el modelo {model} en Ollama (ollama pull {model})")
    return ClientChoice(OllamaClient(config, model), f"Ollama con {model}")


def build_anthropic_client(config: AnthropicConfig) -> ClientChoice:
    api_key = read_secret(config.api_key_env_var)
    if api_key is None:
        return ClientChoice(None, f"falta {config.api_key_env_var} en .env")
    return ClientChoice(AnthropicClient(config, api_key), f"Anthropic con {config.model}")


def build_llm_client(config: LlmConfig) -> ClientChoice:
    """Nunca lanza excepciones: si no hay LLM, devuelve None y el motivo."""
    if config.provider == PROVIDER_OLLAMA:
        return build_ollama_client(config.ollama)
    if config.provider == PROVIDER_ANTHROPIC:
        return build_anthropic_client(config.anthropic)
    return ClientChoice(None, f"proveedor LLM '{config.provider}'")
