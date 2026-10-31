"""Anonimiza un audio_raw.json capturado antes de la privacidad (lo reescribe en su lugar).

El texto de los alumnos pasa a "[ALUMNO responde]" y los nombres del docente a "[NOMBRE]".
El texto original se pierde: si quieres conservarlo, copia antes la carpeta.
"""

import argparse
import logging
from pathlib import Path

from espejo.audio.capture_io import AUDIO_RAW_FILENAME, load_audio_capture, save_audio_capture
from espejo.audio.privacy import Anonymizer
from espejo.config import load_audio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="carpeta con audio_raw.json")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_audio_config()
    capture_path = args.run / AUDIO_RAW_FILENAME

    capture = load_audio_capture(capture_path)
    anonymizer = Anonymizer(config.privacy, config.hallucinations)
    save_audio_capture(anonymizer.anonymize(capture), capture_path)
    print(f"Anonimizado: {capture_path}")


if __name__ == "__main__":
    main()
