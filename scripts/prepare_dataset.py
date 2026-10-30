"""Genera data/processed/teacher y data/processed/behavior en formato YOLO."""

import argparse
import logging

from espejo.config import load_dataset_config
from espejo.dataset.archive import extract_archive, find_archive
from espejo.dataset.inspection import read_source_class_names
from espejo.dataset.preparation import (
    format_prepared_summary,
    prepare_derived_dataset,
    write_data_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["teacher", "behavior", "all"],
        default="all",
        help="qué dataset preparar",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_dataset_config()

    source_root = extract_archive(find_archive(config.subset_dir()), config.subset_dir())
    source_names = read_source_class_names(config.source_yaml())

    for dataset_name, target_classes in config.derived_datasets.items():
        if args.dataset not in ("all", dataset_name):
            continue

        output_root = config.processed_dataset_dir(dataset_name)
        prepared = prepare_derived_dataset(source_root, output_root, source_names, target_classes)
        write_data_yaml(config.data_yaml_path(dataset_name), output_root, list(target_classes))
        print(format_prepared_summary(dataset_name, prepared))


if __name__ == "__main__":
    main()
