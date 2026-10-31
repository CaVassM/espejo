"""Variables de entorno del proyecto (p. ej. HF_TOKEN), leídas desde .env."""

import os

from dotenv import load_dotenv

from espejo.common.config_io import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"


def read_secret(variable_name: str) -> str | None:
    """Lee un secreto desde el entorno o desde .env. Nunca lo registres en logs."""
    load_dotenv(ENV_FILE, override=False)
    value = os.environ.get(variable_name)
    if not value:
        return None
    return value
