"""Transcribe la clase, detecta las preguntas del docente y mide el tiempo de espera."""

import argparse
import logging
from pathlib import Path

from espejo.audio.pipeline import AudioPipeline
from espejo.audio.report_writer import format_audio_summary
from espejo.common.config_io import OUTPUTS_DIR
from espejo.config import DEFAULT_AUDIO_CONFIG, load_audio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="ruta del video de la clase")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_AUDIO_CONFIG)
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config = load_audio_config(args.config)
    pipeline = AudioPipeline(config, model_override=args.model)
    result = pipeline.run(args.video, args.output, args.keep_audio)

    print("\n=== Resumen de audio ===")
    print(format_audio_summary(result.report))
    print(f"\nReporte: {result.report_path}")
    print(f"Transcripción: {result.transcript_path}")


if __name__ == "__main__":
    main()
