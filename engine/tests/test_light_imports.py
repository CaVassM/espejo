"""La parte de INTERPRETAR se instala sin extras (la API): no puede cargar los modelos pesados."""

import json
import os
import subprocess
import sys
from pathlib import Path

ENGINE_SRC = Path(__file__).resolve().parents[1] / "src"
INTERPRET_MODULES = [
    "espejo.config",
    "espejo.analysis.runner",
    "espejo.analysis.report_io",
    "espejo.llm.analyzer",
]
HEAVY_MODULES = ["torch", "ultralytics", "faster_whisper", "ctranslate2", "pyannote.audio", "cv2"]

# Se corre en un proceso aparte: otros tests ya pudieron cargar torch en este.
# Se comparan los módulos de antes y de después del import, porque algunos paquetes instalados
# registran su espacio de nombres al arrancar Python (p. ej. "pyannote").
PROBE = """
import importlib, json, sys
before = set(sys.modules)
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def is_heavy(module_name: str) -> bool:
    for heavy in HEAVY_MODULES:
        if module_name == heavy or module_name.startswith(heavy + "."):
            return True
    return False


def test_interpreting_does_not_import_heavy_models() -> None:
    env = dict(os.environ, PYTHONPATH=str(ENGINE_SRC))
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(INTERPRET_MODULES)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    loaded = json.loads(completed.stdout)

    heavy_loaded = []
    for module_name in loaded:
        if is_heavy(module_name):
            heavy_loaded.append(module_name)
    assert heavy_loaded == []
