"""Estadísticas de un dataset YOLO: imágenes, cajas por clase y co-ocurrencias."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from espejo.dataset.yolo_labels import label_path_for, list_images, read_label_file

SPLITS = ("train", "val")


@dataclass(frozen=True)
class SplitSummary:
    split: str
    image_count: int
    empty_image_count: int
    box_counts: dict[str, int]
    image_counts: dict[str, int]
    image_counts_with_reference: dict[str, int]


def read_source_class_names(dataset_yaml: Path) -> list[str]:
    """Lee los nombres de clase del YAML que publica el autor del dataset.

    Solo parseamos la línea "names:" porque el YAML original trae la ruta de train
    partida en dos líneas y no es YAML válido.
    """
    for line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names_entry = yaml.safe_load(line)
            return list(names_entry["names"])
    raise ValueError(f"No se encontró la lista 'names' en {dataset_yaml}.")


def summarize_split(
    dataset_root: Path,
    split: str,
    class_names: list[str],
    reference_class: str,
) -> SplitSummary:
    """reference_class permite ver cuántas imágenes de cada clase también traen esa clase.

    Lo usamos para saber si en las imágenes de conducta el docente está etiquetado como "teacher".
    """
    images = list_images(dataset_root / "images" / split)
    labels_dir = dataset_root / "labels" / split
    reference_id = class_names.index(reference_class)

    box_counts = dict.fromkeys(class_names, 0)
    image_counts = dict.fromkeys(class_names, 0)
    with_reference_counts = dict.fromkeys(class_names, 0)
    empty_image_count = 0

    for image_path in images:
        boxes = read_label_file(label_path_for(image_path, labels_dir))
        if not boxes:
            empty_image_count += 1

        class_ids_in_image = set()
        for box in boxes:
            box_counts[class_names[box.class_id]] += 1
            class_ids_in_image.add(box.class_id)

        has_reference = reference_id in class_ids_in_image
        for class_id in class_ids_in_image:
            image_counts[class_names[class_id]] += 1
            if has_reference:
                with_reference_counts[class_names[class_id]] += 1

    return SplitSummary(
        split=split,
        image_count=len(images),
        empty_image_count=empty_image_count,
        box_counts=box_counts,
        image_counts=image_counts,
        image_counts_with_reference=with_reference_counts,
    )


def format_split_summary(summary: SplitSummary, reference_class: str) -> str:
    header = (
        f"[{summary.split}] imágenes: {summary.image_count} "
        f"(sin cajas: {summary.empty_image_count})"
    )
    column_titles = f"  {'clase':<24}{'cajas':>8}{'imágenes':>10}{'con ' + reference_class:>14}"
    lines = [header, column_titles]

    for class_name, box_count in summary.box_counts.items():
        image_count = summary.image_counts[class_name]
        with_reference = summary.image_counts_with_reference[class_name]
        lines.append(f"  {class_name:<24}{box_count:>8}{image_count:>10}{with_reference:>14}")
    return "\n".join(lines)
