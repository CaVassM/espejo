"""Sube teacher.pt y behavior.pt al Volume de Modal del worker (carpeta weights/).

Uso:  python scripts/upload_weights.py
"""

import modal

from espejo.common.config_io import CONFIGS_DIR, read_yaml
from espejo.config import load_vision_config

BYTES_PER_MB = 1024 * 1024
REMOTE_WEIGHTS_DIR = "/weights"


def main() -> None:
    volume_name = read_yaml(CONFIGS_DIR / "worker.yaml")["models_volume"]
    models = load_vision_config().models
    weights = [models.teacher_weights, models.behavior_weights]
    for path in weights:
        if not path.exists():
            raise SystemExit(f"No existe {path}. Entrena primero con scripts/train.py.")

    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for path in weights:
            batch.put_file(str(path), f"{REMOTE_WEIGHTS_DIR}/{path.name}")

    for path in weights:
        size_mb = path.stat().st_size / BYTES_PER_MB
        print(f"Subido {path.name} ({size_mb:.1f} MB) a {volume_name}{REMOTE_WEIGHTS_DIR}/")


if __name__ == "__main__":
    main()
