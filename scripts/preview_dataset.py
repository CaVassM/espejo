"""Guarda imágenes con las cajas dibujadas en outputs/dataset_preview/."""

import logging

from espejo.common.config_io import PROJECT_ROOT
from espejo.config import load_dataset_config
from espejo.dataset.preview import save_previews

PREVIEW_DIR = PROJECT_ROOT / "outputs" / "dataset_preview"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_dataset_config()

    for dataset_name, target_classes in config.derived_datasets.items():
        dataset_root = config.processed_dataset_dir(dataset_name)
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"No se encontró {dataset_root}. Prepáralo primero con scripts/prepare_dataset.py"
            )

        saved_paths = save_previews(
            dataset_root,
            list(target_classes),
            PREVIEW_DIR,
            config.preview_image_count,
            config.preview_seed,
        )
        print(f"{dataset_name}: {len(saved_paths)} imágenes en {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
