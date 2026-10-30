"""Entrena el detector teacher o behavior y reporta el mAP@0.5 por clase."""

import argparse
import logging

from espejo.training import (
    TrainingParams,
    evaluate_per_class,
    export_weights,
    save_scores,
    train_detector,
)

AUTO_BATCH = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["teacher", "behavior"], required=True)
    parser.add_argument("--model", default="yolov8s.pt", help="pesos preentrenados de partida")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=AUTO_BATCH, help="-1 = automático")
    parser.add_argument("--fraction", type=float, default=1.0, help="fracción del train a usar")
    parser.add_argument("--name", default=None, help="nombre de la corrida en runs/")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="no copiar best.pt a weights/ (útil en pruebas de humo)",
    )
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> TrainingParams:
    return TrainingParams(
        dataset=args.dataset,
        base_model=args.model,
        epochs=args.epochs,
        image_size=args.imgsz,
        patience=args.patience,
        seed=args.seed,
        batch=args.batch,
        fraction=args.fraction,
        run_name=args.name or args.dataset,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    params = build_params(args)

    best_weights = train_detector(params)
    scores = evaluate_per_class(best_weights, params)
    scores_path = save_scores(scores, params.run_dir())

    if not args.no_export:
        export_weights(best_weights, params.dataset)

    print(f"\nmAP@0.5 por clase ({params.dataset}), guardado en {scores_path}:")
    for score in scores:
        print(f"  {score.class_name:<20}{score.map50:.3f}")


if __name__ == "__main__":
    main()
