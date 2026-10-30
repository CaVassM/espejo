"""Lectura y escritura de etiquetas en formato YOLO (un .txt por imagen)."""

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
FIELDS_PER_LINE = 5


@dataclass(frozen=True)
class YoloBox:
    """Caja con coordenadas normalizadas entre 0 y 1, centrada."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        coordinates = f"{self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
        return f"{self.class_id} {coordinates}"


def parse_label_line(line: str) -> YoloBox | None:
    """Devuelve None en líneas vacías o mal formadas para no romper la lectura."""
    fields = line.split()
    if len(fields) != FIELDS_PER_LINE:
        return None

    return YoloBox(
        class_id=int(fields[0]),
        x_center=float(fields[1]),
        y_center=float(fields[2]),
        width=float(fields[3]),
        height=float(fields[4]),
    )


def read_label_file(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []

    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        box = parse_label_line(line)
        if box is not None:
            boxes.append(box)
    return boxes


def write_label_file(path: Path, boxes: list[YoloBox]) -> None:
    lines = []
    for box in boxes:
        lines.append(box.to_line())
    path.write_text("\n".join(lines), encoding="utf-8")


def list_images(images_dir: Path) -> list[Path]:
    images = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return images


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"
