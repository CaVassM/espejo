from espejo.config import BehaviorConfig
from espejo.vision.geometry import BoundingBox
from espejo.vision.schemas import Behavior, Detection
from espejo.vision.teacher_state import assign_teacher_state, has_stage_interaction
from espejo.vision.teacher_tracker import TeacherTracker, find_dominant_track_id

TEACHER_BOX = BoundingBox(100, 100, 200, 400)
OVERLAPPING_BOX = BoundingBox(90, 100, 260, 420)
STUDENT_BOX = BoundingBox(600, 300, 700, 500)

BEHAVIOR_CONFIG = BehaviorConfig(
    detection_conf=0.25,
    behavior_iou_threshold=0.3,
    guide_class="guide",
    board_writing_class="board_writing",
    stage_interaction_class="stage_interaction",
)


def teacher(track_id: int | None, confidence: float = 0.9) -> Detection:
    return Detection(TEACHER_BOX, "teacher", confidence, track_id)


def test_dominant_track_is_most_frequent() -> None:
    detections_per_frame = [[teacher(1)], [teacher(1), teacher(7)], [teacher(1)], [teacher(7)]]
    assert find_dominant_track_id(detections_per_frame) == 1


def test_no_tracks_gives_none() -> None:
    assert find_dominant_track_id([[], [teacher(None)]]) is None


def test_fallback_requires_high_confidence() -> None:
    tracker = TeacherTracker(teacher_fallback_conf=0.5)
    detections_per_frame = [
        [teacher(1)],
        [teacher(1)],
        [teacher(1)],
        [teacher(9, confidence=0.8)],
        [teacher(9, confidence=0.3)],
        [],
    ]

    boxes = tracker.select_teacher_boxes(detections_per_frame)

    assert boxes == [TEACHER_BOX, TEACHER_BOX, TEACHER_BOX, TEACHER_BOX, None, None]


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
