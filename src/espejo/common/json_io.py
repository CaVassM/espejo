"""Escritura de reportes JSON."""

import json
from pathlib import Path
from typing import Any

JSON_INDENT = 2


def write_json(content: dict[str, Any], output_path: Path) -> None:
    """UTF-8 sin escapar tildes, para que el JSON se lea bien en español."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, indent=JSON_INDENT, ensure_ascii=False)
    output_path.write_text(text, encoding="utf-8")
