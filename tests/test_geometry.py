import pytest

from espejo.vision.geometry import BoundingBox, FloorGrid, GridCell, iou


def test_iou_of_identical_boxes_is_one() -> None:
    box = BoundingBox(10, 10, 50, 50)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(20, 20, 30, 30)
    assert iou(first, second) == 0.0


def test_iou_of_partial_overlap() -> None:
    # Intersección 5x10 = 50; unión 100 + 100 - 50 = 150
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 15, 10)
    assert iou(first, second) == pytest.approx(50 / 150)


def test_iou_with_degenerate_box_is_zero() -> None:
    empty = BoundingBox(5, 5, 5, 5)
    assert iou(empty, empty) == 0.0


def test_foot_point_is_bottom_center() -> None:
    box = BoundingBox(100, 50, 200, 300)
    assert box.foot_point() == (150, 300)


def test_grid_cell_for_point_inside_frame() -> None:
    grid = FloorGrid(columns=8, rows=6, frame_width=800, frame_height=600)
    assert grid.cell_at(150, 250) == GridCell(column=1, row=2)


def test_grid_cell_clamps_point_on_border() -> None:
    grid = FloorGrid(columns=8, rows=6, frame_width=800, frame_height=600)
    assert grid.cell_at(800, 600) == GridCell(column=7, row=5)
