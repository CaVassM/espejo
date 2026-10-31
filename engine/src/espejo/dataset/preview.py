"""Dibuja las cajas de un dataset YOLO para revisarlas a ojo."""

import random
from pathlib import Path

import cv2
import numpy as np

from espejo.dataset.yolo_labels import YoloBox, label_path_for, list_images, read_label_file

PREVIEW_SPLIT = "train"
# Colores BGR, uno por id de clase
CLASS_COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (200, 0, 200)]
BOX_THICKNESS = 3
FONT_SCALE = 0.9
FONT_THICKNESS = 2
TEXT_MARGIN_PX = 6


def draw_yolo_box(image: np.ndarray, box: YoloBox, class_name: str) -> None:
    """Convierte la caja normalizada a píxeles y la dibuja con su clase."""
    image_height, image_width = image.shape[:2]
    half_width = box.width * image_width / 2
    half_height = box.height * image_height / 2
    center_x = box.x_center * image_width
    center_y = box.y_center * image_height

    top_left = (int(center_x - half_width), int(center_y - half_height))
    bottom_right = (int(center_x + half_width), int(center_y + half_height))
    color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
    text_origin = (top_left[0], max(top_left[1] - TEXT_MARGIN_PX, 0))

    cv2.rectangle(image, top_left, bottom_right, color, BOX_THICKNESS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, class_name, text_origin, font, FONT_SCALE, color, FONT_THICKNESS)


def save_previews(
    dataset_root: Path,
    class_names: list[str],
    output_dir: Path,
    image_count: int,
    seed: int,
) -> list[Path]:
    images = list_images(dataset_root / "images" / PREVIEW_SPLIT)
    labels_dir = dataset_root / "labels" / PREVIEW_SPLIT
    chosen_images = random.Random(seed).sample(images, min(image_count, len(images)))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for position, image_path in enumerate(chosen_images):
        image = cv2.imread(str(image_path))
        for box in read_label_file(label_path_for(image_path, labels_dir)):
            draw_yolo_box(image, box, class_names[box.class_id])

        output_path = output_dir / f"{dataset_root.name}_{position}.jpg"
        cv2.imwrite(str(output_path), image)
        saved_paths.append(output_path)
    return saved_paths
