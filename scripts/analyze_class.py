"""Analiza una clase completa: VER (video y audio, una sola vez) e INTERPRETAR."""

import argparse
import logging
from pathlib import Path

from espejo.analysis.report_io import format_summary
from espejo.analysis.runner import (
    capture_video,
    interpret_run,
    load_all_configs,
    run_dir_for,
    try_capture_audio,
    write_debug_video,
)
from espejo.common.config_io import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="ruta del video de la clase")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help="genera un video con la caja y el estado del docente (no va a git)",
    )
    parser.add_argument("--no-audio", action="store_true", help="analiza solo el video")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    configs = load_all_configs()
    run_dir = run_dir_for(args.video, args.output)

    capture_video(args.video, run_dir, configs.vision)
    audio_captured = False
    if not args.no_audio:
        audio_captured = try_capture_audio(args.video, run_dir, configs.audio)

    result = interpret_run(run_dir, configs, include_audio=audio_captured)
    if args.debug_video:
        write_debug_video(args.video, result, configs.vision)

    print("\n=== Resumen de la clase ===")
    print(format_summary(result.report))
    print(f"\nResultados en: {run_dir}")


if __name__ == "__main__":
    main()
