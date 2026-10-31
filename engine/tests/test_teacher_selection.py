from espejo.config import BehaviorConfig, TeacherConfig
from espejo.vision.geometry import BoundingBox
from espejo.vision.schemas import Behavior, Detection, FrameDetections
from espejo.vision.teacher_state import assign_teacher_state, has_stage_interaction
from espejo.vision.teacher_tracker import TeacherTracker

TEACHER_BOX = BoundingBox(100, 100, 200, 400)
OVERLAPPING_BOX = BoundingBox(90, 100, 260, 420)
STUDENT_BOX = BoundingBox(600, 300, 700, 500)
SAMPLE_INTERVAL_S = 0.5

TEACHER_CONFIG = TeacherConfig(
    detection_conf=0.25,
    teacher_min_conf=0.7,
    similar_conf_margin=0.1,
    gap_fill_s=3.0,
    class_name="teacher",
)
BEHAVIOR_CONFIG = BehaviorConfig(
    detection_conf=0.25,
    behavior_iou_threshold=0.3,
    guide_class="guide",
    guide_min_conf=0.5,
    board_writing_min_conf=0.25,
    ignored_classes=[],
    min_guide_s=2.0,
    board_writing_class="board_writing",
    stage_interaction_class="stage_interaction",
)


def teacher(box: BoundingBox, confidence: float) -> Detection:
    return Detection(box, "teacher", confidence)


def make_frames(detections_per_frame: list[list[Detection]]) -> list[FrameDetections]:
    """Frames muestreados a 2 fps, sin conductas."""
    frames = []
    for index, detections in enumerate(detections_per_frame):
        frames.append(FrameDetections(index, index * SAMPLE_INTERVAL_S, detections, []))
    return frames


def selected_boxes(frames: list[FrameDetections]) -> list[BoundingBox | None]:
    boxes = []
    for selection in TeacherTracker(TEACHER_CONFIG).select(frames):
        boxes.append(selection.box)
    return boxes


def test_most_confident_teacher_wins() -> None:
    frames = make_frames([[teacher(STUDENT_BOX, 0.75), teacher(TEACHER_BOX, 0.95)]])
    assert selected_boxes(frames) == [TEACHER_BOX]


def test_low_confidence_box_is_not_the_teacher() -> None:
    # Un alumno quieto detectado con 0.6 ya no puede ser el docente
    frames = make_frames([[teacher(STUDENT_BOX, 0.6)]])
    assert selected_boxes(frames) == [None]


def test_similar_confidence_prefers_last_position() -> None:
    near_box = BoundingBox(110, 100, 210, 400)
    frames = make_frames(
        [
            [teacher(TEACHER_BOX, 0.9)],
            [teacher(STUDENT_BOX, 0.88), teacher(near_box, 0.82)],
        ]
    )
    assert selected_boxes(frames) == [TEACHER_BOX, near_box]


def test_short_gap_is_filled_and_marked_interpolated() -> None:
    # Se pierde 1 s (2 frames) y vuelve: se rellena con la última posición
    frames = make_frames([[teacher(TEACHER_BOX, 0.9)], [], [], [teacher(TEACHER_BOX, 0.9)]])

    selections = TeacherTracker(TEACHER_CONFIG).select(frames)

    assert selections[1].box == TEACHER_BOX
    assert selections[1].interpolated is True
    assert selections[0].interpolated is False


def test_long_gap_is_not_filled() -> None:
    # 4 s sin docente: más que gap_fill_s, no se inventa su posición
    empty_frames: list[list[Detection]] = [[] for _ in range(7)]
    frames = make_frames([[teacher(TEACHER_BOX, 0.9)], *empty_frames, [teacher(TEACHER_BOX, 0.9)]])

    boxes = selected_boxes(frames)

    assert boxes[1:8] == [None] * 7


def test_gap_at_the_end_is_not_filled() -> None:
    frames = make_frames([[teacher(TEACHER_BOX, 0.9)], [], []])
    assert selected_boxes(frames) == [TEACHER_BOX, None, None]


def test_board_writing_wins_over_guide() -> None:
    # guide tiene más confianza, pero la prioridad es board_writing > guide
    detections = [
        Detection(OVERLAPPING_BOX, "guide", 0.95),
        Detection(TEACHER_BOX, "board_writing", 0.4),
    ]
    assert assign_teacher_state(detections, TEACHER_BOX, BEHAVIOR_CONFIG) == Behavior.BOARD_WRITING


def test_guide_wins_over_other() -> None:
    detections = [Detection(OVERLAPPING_BOX, "guide", 0.6)]
    assert assign_teacher_state(detections, TEACHER_BOX, BEHAVIOR_CONFIG) == Behavior.GUIDE


def test_low_confidence_guide_is_other() -> None:
    detections = [Detection(OVERLAPPING_BOX, "guide", 0.4)]
    assert assign_teacher_state(detections, TEACHER_BOX, BEHAVIOR_CONFIG) == Behavior.OTHER


def test_behavior_far_from_teacher_is_other() -> None:
    detections = [Detection(STUDENT_BOX, "board_writing", 0.9)]
    assert assign_teacher_state(detections, TEACHER_BOX, BEHAVIOR_CONFIG) == Behavior.OTHER


def test_student_classes_are_never_assigned_to_teacher() -> None:
    # Aunque se superpongan exactamente con el docente, answer y stage_interaction son de alumnos
    detections = [
        Detection(TEACHER_BOX, "answer", 0.99),
        Detection(TEACHER_BOX, "stage_interaction", 0.99),
    ]
    assert assign_teacher_state(detections, TEACHER_BOX, BEHAVIOR_CONFIG) == Behavior.OTHER


def test_no_teacher_means_not_visible() -> None:
    assert assign_teacher_state([], None, BEHAVIOR_CONFIG) == Behavior.NOT_VISIBLE


def test_stage_interaction_is_detected_anywhere_in_the_room() -> None:
    detections = [Detection(STUDENT_BOX, "stage_interaction", 0.7)]
    assert has_stage_interaction(detections, BEHAVIOR_CONFIG)


def test_answer_is_not_stage_interaction() -> None:
    detections = [Detection(STUDENT_BOX, "answer", 0.9)]
    assert not has_stage_interaction(detections, BEHAVIOR_CONFIG)
