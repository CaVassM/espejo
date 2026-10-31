"""Rutas del proyecto y lectura de archivos YAML."""

from pathlib import Path
from typing import Any

import yaml

# engine/src/espejo/common/config_io.py: la raíz del repo está cuatro carpetas arriba
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración {path}.")
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_project_path(relative_path: str) -> Path:
    """Las rutas del YAML son relativas a la raíz del repo, no al directorio actual."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
