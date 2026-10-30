"""Geometría básica de cajas delimitadoras."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Caja en píxeles con formato (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def foot_point(self) -> tuple[float, float]:
        """Punto medio del borde inferior: aproxima dónde está parada la persona."""
        center_x = (self.x1 + self.x2) / 2
        return center_x, self.y2


@dataclass(frozen=True)
class GridCell:
    column: int
    row: int


@dataclass(frozen=True)
class FloorGrid:
    """Divide el frame en celdas para medir por qué zonas del aula pasó el docente."""

    columns: int
    rows: int
    frame_width: int
    frame_height: int

    def cell_count(self) -> int:
        return self.columns * self.rows

    def cell_at(self, x: float, y: float) -> GridCell:
        """Los puntos en el borde derecho o inferior caen en la última celda."""
        column = int(x / self.frame_width * self.columns)
        row = int(y / self.frame_height * self.rows)
        column = min(max(column, 0), self.columns - 1)
        row = min(max(row, 0), self.rows - 1)
        return GridCell(column, row)


def intersection_area(first: BoundingBox, second: BoundingBox) -> float:
    overlap = BoundingBox(
        x1=max(first.x1, second.x1),
        y1=max(first.y1, second.y1),
        x2=min(first.x2, second.x2),
        y2=min(first.y2, second.y2),
    )
    return overlap.area()


def iou(first: BoundingBox, second: BoundingBox) -> float:
    """Intersección sobre unión; 0 si las cajas no se tocan o son degeneradas."""
    shared_area = intersection_area(first, second)
    union_area = first.area() + second.area() - shared_area
    if union_area <= 0:
        return 0.0
    return shared_area / union_area
