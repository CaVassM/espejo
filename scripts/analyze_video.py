"""Analiza el video de una clase y genera vision_report.json y heatmap.png."""

import argparse
import logging
from pathlib import Path

from espejo.config import (
    DEFAULT_VISION_CONFIG,
    PROJECT_ROOT,
    load_vision_config,
    with_model_overrides,
)
from espejo.vision.pipeline import VisionPipeline
from espejo.vision.report_writer import format_metrics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="ruta del video de la clase")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help="genera un video con la caja, la conducta y la orientación dibujadas",
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
        help="reemplaza los pesos de conductas (solo para probar el flujo sin entrenar)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config = load_vision_config(args.config)
    config = with_model_overrides(
        config, args.teacher_weights, args.teacher_class, args.behavior_weights
    )

    pipeline = VisionPipeline(config)
    result = pipeline.run(args.video, args.output, args.debug_video)

    print("\n=== Resumen de visión ===")
    print(format_metrics_summary(result.report))
    print(f"\nReporte: {result.report_path}")
    print(f"Heatmap: {result.heatmap_path}")
    if result.debug_video_path is not None:
        print(f"Video de depuración: {result.debug_video_path}")


if __name__ == "__main__":
    main()
