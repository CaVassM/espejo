"""Analiza solo el audio de una clase: preguntas, tiempo de espera y respuestas."""

import argparse
import logging
from pathlib import Path

from espejo.analysis.capture import capture_audio
from espejo.analysis.report_io import format_summary
from espejo.analysis.runner import interpret_run, load_all_configs, run_dir_for
from espejo.common.config_io import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="ruta del video de la clase")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="conserva audio_raw.wav y audio_vad.wav en la carpeta de salida",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="modelo de Whisper a usar (p. ej. small); por defecto lo decide la config",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help='tema de la clase para ayudar a Whisper (p. ej. "Cuento: Los hijos del labrador")',
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    configs = load_all_configs()
    run_dir = run_dir_for(args.video, args.output)

    capture_audio(args.video, run_dir, configs.audio, args.keep_audio, args.model, args.topic)
    result = interpret_run(run_dir, configs, include_video=False)

    print("\n=== Resumen de audio ===")
    print(format_summary(result.report))
    print(f"\nResultados en: {run_dir}")


if __name__ == "__main__":
    main()
