"""Revisa el worker desplegado: /health responde y el Volume tiene los pesos. Imprime la URL.

Uso:  python scripts/check_modal_worker.py
"""

import json
import urllib.request

import modal

from espejo.common.config_io import CONFIGS_DIR, read_yaml
from espejo.config import load_vision_config

WEB_FUNCTION = "web"
WEIGHTS_DIR = "weights"
HEALTH_TIMEOUT_S = 120  # el contenedor web puede estar apagado y tardar en arrancar


def main() -> None:
    settings = read_yaml(CONFIGS_DIR / "worker.yaml")
    url = modal.Function.from_name(settings["app_name"], WEB_FUNCTION).get_web_url()
    print(f"URL del worker: {url}")

    with urllib.request.urlopen(f"{url}/health", timeout=HEALTH_TIMEOUT_S) as response:
        health = json.loads(response.read().decode("utf-8"))
    if health != {"status": "ok"}:
        raise SystemExit(f"/health respondió {health}")
    print("/health: ok")

    volume = modal.Volume.from_name(settings["models_volume"])
    present = set()
    for entry in volume.listdir(WEIGHTS_DIR):
        present.add(entry.path.split("/")[-1])
    models = load_vision_config().models
    expected = {models.teacher_weights.name, models.behavior_weights.name}
    missing = expected - present
    if missing:
        raise SystemExit(f"Faltan pesos en el Volume: {', '.join(sorted(missing))}")
    print(f"Pesos en el Volume: {', '.join(sorted(expected))}")


if __name__ == "__main__":
    main()
