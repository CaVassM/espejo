"""INTERPRETAR con engine/ (sin torch): métricas, ficha por rúbrica, LLM (Cohere) y consejo."""

import json
import tempfile
from pathlib import Path
from typing import Any

from espejo.analysis.runner import REPORT_FILENAME, interpret_run, load_all_configs
from espejo.audio.capture_io import AUDIO_RAW_FILENAME
from espejo.vision.detections_io import DETECTIONS_FILENAME


def interpret_raw(detections: Any, audio_raw: Any) -> dict[str, Any]:
    """Escribe los JSON del worker en una carpeta temporal y devuelve class_report.json.

    En Render el disco es temporal: lo que importa se guarda en la base de datos.
    """
    with tempfile.TemporaryDirectory(prefix="espejo-") as folder:
        run_dir = Path(folder)
        (run_dir / DETECTIONS_FILENAME).write_text(json.dumps(detections), encoding="utf-8")
        if audio_raw is not None:
            audio_text = json.dumps(audio_raw, ensure_ascii=False)
            (run_dir / AUDIO_RAW_FILENAME).write_text(audio_text, encoding="utf-8")
        interpret_run(run_dir, load_all_configs())
        return json.loads((run_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
