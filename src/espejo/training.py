"""Entrenamiento y evaluación de los detectores YOLO de Espejo."""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from ultralytics import YOLO

from espejo.config import CONFIGS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SCORES_FILENAME = "per_class_map50.json"


@dataclass(frozen=True)
class TrainingParams:
    dataset: str
    base_model: str
    epochs: int
    image_size: int
    patience: int
    seed: int
    batch: int
    fraction: float
    run_name: str

    def data_yaml(self) -> Path:
        return CONFIGS_DIR / f"data_{self.dataset}.yaml"

    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name


@dataclass(frozen=True)
class ClassScore:
    class_name: str
    map50: float


def train_detector(params: TrainingParams) -> Path:
    """Entrena y devuelve la ruta del best.pt."""
    data_yaml = params.data_yaml()
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"No se encontró {data_yaml}. Prepara el dataset con scripts/prepare_dataset.py"
        )

    logger.info("Entrenando %s sobre %s", params.base_model, params.data_yaml())
    model = YOLO(params.base_model)
    model.train(
        data=str(params.data_yaml()),
        epochs=params.epochs,
        imgsz=params.image_size,
        patience=params.patience,
        seed=params.seed,
        batch=params.batch,
        fraction=params.fraction,
        project=str(RUNS_DIR),
        name=params.run_name,
        exist_ok=True,
    )
    return params.run_dir() / "weights" / "best.pt"


def evaluate_per_class(weights_path: Path, params: TrainingParams) -> list[ClassScore]:
    """Valida el best.pt en el split val y devuelve el mAP@0.5 de cada clase."""
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(params.data_yaml()),
        imgsz=params.image_size,
        project=str(RUNS_DIR),
        name=f"{params.run_name}_val",
        exist_ok=True,
    )

    scores = []
    for position, class_id in enumerate(metrics.box.ap_class_index):
        class_name = metrics.names[int(class_id)]
        map50 = float(metrics.box.ap50[position])
        scores.append(ClassScore(class_name, map50))
    return scores


def save_scores(scores: list[ClassScore], run_dir: Path) -> Path:
    serializable = []
    for score in scores:
        serializable.append(asdict(score))

    output_path = run_dir / SCORES_FILENAME
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def export_weights(best_weights: Path, dataset: str) -> Path:
    """Copia el best.pt a weights/<dataset>.pt, que es donde lo busca el pipeline."""
    WEIGHTS_DIR.mkdir(exist_ok=True)
    destination = WEIGHTS_DIR / f"{dataset}.pt"
    shutil.copy2(best_weights, destination)
    logger.info("Pesos copiados a %s", destination)
    return destination
