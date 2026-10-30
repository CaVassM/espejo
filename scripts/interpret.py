"""Re-interpreta una clase ya analizada, sin volver a correr los modelos."""

import argparse
import logging
from pathlib import Path

from espejo.analysis.report_io import format_summary
from espejo.analysis.runner import interpret_run, load_all_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="carpeta con detections.json y/o audio_raw.json (p. ej. outputs/clase/)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    result = interpret_run(args.run, load_all_configs())

    print("\n=== Resumen de la clase ===")
    print(format_summary(result.report))
    print(f"\nResultados en: {args.run}")


if __name__ == "__main__":
    main()
