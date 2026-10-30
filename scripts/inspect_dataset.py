"""Lista el repo de SCB-Dataset y, opcionalmente, descarga e inspecciona el subconjunto."""

import argparse
import logging

from espejo.config import load_dataset_config
from espejo.dataset.archive import extract_archive, find_archive
from espejo.dataset.hub import download_subset, summarize_repo_folders
from espejo.dataset.inspection import (
    SPLITS,
    format_split_summary,
    read_source_class_names,
    summarize_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="descarga y extrae el subset")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_dataset_config()

    print(f"Repositorio {config.repo_id}")
    for folder in summarize_repo_folders(config.repo_id):
        print(
            f"  {folder.archive_size_mb:>9.1f} MB  {folder.file_count:>4} archivos  {folder.name}"
        )

    if args.download:
        download_subset(config.repo_id, config.subset_folder, config.raw_dir)

    extracted_root = extract_archive(find_archive(config.subset_dir()), config.subset_dir())
    class_names = read_source_class_names(config.subset_dir() / f"{config.subset_folder}.yaml")
    print(f"\nDataset en {extracted_root}\nClases del autor: {class_names}")

    for split in SPLITS:
        summary = summarize_split(extracted_root, split, class_names, config.reference_class)
        print(format_split_summary(summary, config.reference_class))


if __name__ == "__main__":
    main()
