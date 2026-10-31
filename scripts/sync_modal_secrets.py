"""Crea o actualiza el Secret de Modal del worker desde .env (o desde el entorno, en CI).

Nunca muestra los valores, solo los nombres de las claves.
Uso:  python scripts/sync_modal_secrets.py
"""

import modal

from espejo.common.config_io import CONFIGS_DIR, read_yaml
from espejo.common.env import read_secret

REQUIRED_KEYS = ["HF_TOKEN", "WORKER_TOKEN", "UPLOAD_ACCESS_CODE"]
OPTIONAL_KEYS = ["API_BASE_URL", "FRONTEND_ORIGINS"]


def collect_values() -> dict[str, str]:
    values = {}
    missing = []
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        value = read_secret(key)
        if value is not None:
            values[key] = value
        elif key in REQUIRED_KEYS:
            missing.append(key)
    if missing:
        raise SystemExit(f"Faltan en .env o en el entorno: {', '.join(missing)}")
    return values


def main() -> None:
    secret_name = read_yaml(CONFIGS_DIR / "worker.yaml")["secret_name"]
    values = collect_values()
    modal.Secret.objects.create(secret_name, values, allow_existing=True)
    modal.Secret.from_name(secret_name).update(values)
    print(f"Secret {secret_name} actualizado con: {', '.join(sorted(values))} (valores ocultos)")


if __name__ == "__main__":
    main()
