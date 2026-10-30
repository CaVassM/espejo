"""Deriva datasets YOLO de una sola tarea a partir del subconjunto fusionado del autor."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from espejo.config import PROJECT_ROOT
from espejo.dataset.inspection import SPLITS
from espejo.dataset.yolo_labels import (
    YoloBox,
    label_path_for,
    list_images,
    read_label_file,
    write_label_file,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    image_count: int
    box_counts: dict[str, int]


def build_class_mapping(source_names: list[str], target_classes: dict[str, str]) -> dict[int, int]:
    """Traduce id de clase del autor -> id de clase nuevo, según el orden de target_classes."""
    mapping = {}
    for target_id, source_name in enumerate(target_classes.values()):
        if source_name not in source_names:
            raise ValueError(f"La clase '{source_name}' no existe en el dataset del autor.")
        mapping[source_names.index(source_name)] = target_id
    return mapping


def remap_boxes(boxes: list[YoloBox], mapping: dict[int, int]) -> list[YoloBox]:
    """Conserva solo las cajas de clases mapeadas, con su nuevo id."""
    remapped = []
    for box in boxes:
        if box.class_id not in mapping:
            continue
        new_box = YoloBox(mapping[box.class_id], box.x_center, box.y_center, box.width, box.height)
        remapped.append(new_box)
    return remapped


def link_or_copy(source: Path, destination: Path) -> None:
    """Un enlace duro evita duplicar 1.6 GB de imágenes; si el disco no lo permite, copiamos."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prepare_split(
    source_root: Path,
    output_root: Path,
    split: str,
    mapping: dict[int, int],
    target_names: list[str],
) -> PreparedSplit:
    output_images_dir = output_root / "images" / split
    output_labels_dir = output_root / "labels" / split
    output_images_dir.mkdir(parents=True)
    output_labels_dir.mkdir(parents=True)

    source_labels_dir = source_root / "labels" / split
    box_counts = dict.fromkeys(target_names, 0)
    kept_images = 0

    for image_path in list_images(source_root / "images" / split):
        source_boxes = read_label_file(label_path_for(image_path, source_labels_dir))
        boxes = remap_boxes(source_boxes, mapping)
        if not boxes:
            continue

        link_or_copy(image_path, output_images_dir / image_path.name)
        write_label_file(label_path_for(image_path, output_labels_dir), boxes)
        kept_images += 1
        for box in boxes:
            box_counts[target_names[box.class_id]] += 1

    return PreparedSplit(split, kept_images, box_counts)


def prepare_derived_dataset(
    source_root: Path,
    output_root: Path,
    source_names: list[str],
    target_classes: dict[str, str],
) -> list[PreparedSplit]:
    """Regenera el dataset desde cero para que no queden archivos de una corrida anterior."""
    if output_root.exists():
        logger.info("Borrando la versión anterior de %s", output_root)
        shutil.rmtree(output_root)

    mapping = build_class_mapping(source_names, target_classes)
    target_names = list(target_classes.keys())

    prepared_splits = []
    for split in SPLITS:
        logger.info("Preparando %s/%s", output_root.name, split)
        prepared = prepare_split(source_root, output_root, split, mapping, target_names)
        prepared_splits.append(prepared)
    return prepared_splits


def write_data_yaml(yaml_path: Path, output_root: Path, target_names: list[str]) -> None:
    """Rutas relativas a configs/ para que el YAML sirva en cualquier máquina.

    Sin la clave "path", ultralytics resuelve train y val desde la carpeta del YAML.
    """
    relative_root = Path("..") / output_root.relative_to(PROJECT_ROOT)
    names = {}
    for class_id, class_name in enumerate(target_names):
        names[class_id] = class_name

    content = {
        "train": (relative_root / "images" / "train").as_posix(),
        "val": (relative_root / "images" / "val").as_posix(),
        "names": names,
    }
    header = "# Generado por scripts/prepare_dataset.py. No editar a mano.\n"
    yaml_text = yaml.safe_dump(content, sort_keys=False, allow_unicode=True)
    yaml_path.write_text(header + yaml_text, encoding="utf-8")


def format_prepared_summary(dataset_name: str, prepared_splits: list[PreparedSplit]) -> str:
    lines = [f"Dataset {dataset_name}"]
    for prepared in prepared_splits:
        lines.append(f"  [{prepared.split}] imágenes: {prepared.image_count}")
        for class_name, box_count in prepared.box_counts.items():
            lines.append(f"      {class_name:<20}{box_count:>7} cajas")
    return "\n".join(lines)
