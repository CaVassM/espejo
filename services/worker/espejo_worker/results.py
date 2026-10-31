"""El resultado de un trabajo (formato en docs/worker_contract.md) y el callback al backend."""

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Los mismos nombres que usa engine/ (detections_io, capture_io y runner)
DETECTIONS_FILENAME = "detections.json"
AUDIO_RAW_FILENAME = "audio_raw.json"
HEATMAP_FILENAME = "heatmap.png"
RESULT_FILENAMES = [DETECTIONS_FILENAME, AUDIO_RAW_FILENAME, HEATMAP_FILENAME]
CALLBACK_PATH = "/api/analyses/{analysis_id}/raw"
HTTP_OK_MIN = 200
HTTP_OK_MAX = 299

ReadFile = Callable[[str], bytes | None]


def build_result(job_id: str, analysis_id: str | None, read_file: ReadFile) -> dict[str, Any]:
    """read_file recibe la ruta dentro del Volume ("<job_id>/detections.json")."""
    detections = read_file(f"{job_id}/{DETECTIONS_FILENAME}")
    audio_raw = read_file(f"{job_id}/{AUDIO_RAW_FILENAME}")
    heatmap = read_file(f"{job_id}/{HEATMAP_FILENAME}")
    return {
        "job_id": job_id,
        "analysis_id": analysis_id,
        "detections": parse_json(detections),
        "audio_raw": parse_json(audio_raw),
        "heatmap_png_base64": encode_png(heatmap),
    }


def parse_json(content: bytes | None) -> Any:
    if content is None:
        return None
    return json.loads(content.decode("utf-8"))


def encode_png(content: bytes | None) -> str | None:
    if content is None:
        return None
    return base64.b64encode(content).decode("ascii")


def callback_url(api_base_url: str, analysis_id: str) -> str:
    return api_base_url + CALLBACK_PATH.format(analysis_id=analysis_id)


def send_callback(
    url: str,
    worker_token: str,
    payload: dict[str, Any],
    attempts: int,
    timeout_s: float,
    backoff_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """POST con reintentos. Nunca lanza: si el backend no responde, el resultado sigue en GET."""
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {worker_token}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if HTTP_OK_MIN <= response.status <= HTTP_OK_MAX:
                    return True
        except (urllib.error.URLError, OSError) as error:
            logger.warning("Callback falló (intento %d): %s", attempt, type(error).__name__)
        if attempt < attempts:
            sleep(backoff_s * attempt)
    return False
