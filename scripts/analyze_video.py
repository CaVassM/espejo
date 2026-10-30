"""Analiza solo el video de una clase (sin audio)."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from espejo.analysis.report_io import format_summary
from espejo.analysis.runner import (
    capture_video,
    interpret_run,
    load_all_configs,
    run_dir_for,
    write_debug_video,
)
from espejo.common.config_io import OUTPUTS_DIR
from espejo.config import with_model_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="ruta del video de la clase")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help="genera un video con la caja y el estado del docente (no va a git)",
    )
    parser.add_argument(
        "--teacher-weights",
        default=None,
        help="reemplaza los pesos del docente (p. ej. yolov8s.pt para una prueba sin entrenar)",
    )
    parser.add_argument(
        "--teacher-class",
        default=None,
        help="clase que se toma como docente (p. ej. person con el modelo genérico)",
    )
    parser.add_argument(
        "--behavior-weights",
        default=None,
        help="reemplaza los pesos de conductas (solo para probar el flujo)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    configs = load_all_configs()
    vision_config = with_model_overrides(
        configs.vision, args.teacher_weights, args.teacher_class, args.behavior_weights
    )
    configs = replace(configs, vision=vision_config)
    run_dir = run_dir_for(args.video, args.output)

    capture_video(args.video, run_dir, configs.vision)
    result = interpret_run(run_dir, configs, include_audio=False)
    if args.debug_video:
        write_debug_video(args.video, result, configs.vision)

    print("\n=== Resumen de video ===")
    print(format_summary(result.report))
    print(f"\nResultados en: {run_dir}")


if __name__ == "__main__":
    main()
