"""Mapa de calor del recorrido del docente sobre un fondo desenfocado."""

from pathlib import Path

import cv2
import numpy as np

from espejo.config import HeatmapConfig
from espejo.vision.geometry import BoundingBox, FloorGrid

MAX_PIXEL_VALUE = 255


class HeatmapRenderer:
    """Dibuja por dónde se paró el docente.

    El fondo se desenfoca con fuerza para que no se pueda reconocer a nadie.
    """

    def __init__(self, config: HeatmapConfig) -> None:
        self.config = config

    def render(self, background: np.ndarray, teacher_boxes: list[BoundingBox]) -> np.ndarray:
        frame_height, frame_width = background.shape[:2]
        grid = FloorGrid(self.config.grid_columns, self.config.grid_rows, frame_width, frame_height)

        visit_counts = accumulate_visits(teacher_boxes, grid)
        heat_layer = self.colored_heat_layer(visit_counts, frame_width, frame_height)
        private_background = self.blur_background(background)

        alpha = self.config.overlay_alpha
        return cv2.addWeighted(private_background, 1 - alpha, heat_layer, alpha, 0)

    def blur_background(self, background: np.ndarray) -> np.ndarray:
        kernel = odd_kernel_size(self.config.background_blur_kernel)
        return cv2.GaussianBlur(background, (kernel, kernel), 0)

    def colored_heat_layer(
        self, visit_counts: np.ndarray, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """Escala la grilla al tamaño del frame y la suaviza para que se lea como zonas."""
        normalized = visit_counts.astype(np.float32)
        if normalized.max() > 0:
            normalized = normalized / normalized.max()

        frame_size = (frame_width, frame_height)
        upscaled = cv2.resize(normalized, frame_size, interpolation=cv2.INTER_NEAREST)
        kernel = odd_kernel_size(self.config.heat_blur_kernel)
        smoothed = cv2.GaussianBlur(upscaled, (kernel, kernel), 0)

        intensity = np.clip(smoothed * MAX_PIXEL_VALUE, 0, MAX_PIXEL_VALUE).astype(np.uint8)
        return cv2.applyColorMap(intensity, cv2.COLORMAP_JET)


def accumulate_visits(teacher_boxes: list[BoundingBox], grid: FloorGrid) -> np.ndarray:
    """Cuántos frames pasó el docente en cada celda (filas x columnas)."""
    visit_counts = np.zeros((grid.rows, grid.columns), dtype=np.int64)
    for box in teacher_boxes:
        foot_x, foot_y = box.foot_point()
        cell = grid.cell_at(foot_x, foot_y)
        visit_counts[cell.row, cell.column] += 1
    return visit_counts


def odd_kernel_size(size: int) -> int:
    """OpenCV exige kernels impares para el desenfoque gaussiano."""
    if size % 2 == 0:
        return size + 1
    return size


def save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
