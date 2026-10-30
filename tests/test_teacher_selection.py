from espejo.config import BehaviorConfig
from espejo.vision.detectors import assign_behavior
from espejo.vision.geometry import BoundingBox
from espejo.vision.schemas import Behavior, Detection
from espejo.vision.teacher_tracker import TeacherTracker, find_dominant_track_id

TEACHER_BOX = BoundingBox(100, 100, 200, 400)
STUDENT_BOX = BoundingBox(600, 300, 700, 500)

BEHAVIOR_CONFIG = BehaviorConfig(
    detection_conf=0.25,
    behavior_iou_threshold=0.3,
    guide_class="guide",
    board_writing_class="board_writing",
    participation_classes=["answer", "stage_interaction"],
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


def test_overlapping_guide_is_assigned_to_teacher() -> None:
    detections = [
        Detection(BoundingBox(90, 100, 260, 420), "guide", 0.6),
        Detection(BoundingBox(100, 100, 200, 400), "board_writing", 0.8),
    ]
    result = assign_behavior(detections, TEACHER_BOX, BEHAVIOR_CONFIG)
    assert result.behavior == Behavior.BOARD_WRITING


def test_behavior_far_from_teacher_is_other() -> None:
    detections = [Detection(STUDENT_BOX, "guide", 0.9)]
    result = assign_behavior(detections, TEACHER_BOX, BEHAVIOR_CONFIG)
    assert result.behavior == Behavior.OTHER


def test_student_behaviors_count_as_participation_only() -> None:
    detections = [
        Detection(TEACHER_BOX, "answer", 0.9),
        Detection(STUDENT_BOX, "stage_interaction", 0.7),
    ]
    result = assign_behavior(detections, TEACHER_BOX, BEHAVIOR_CONFIG)
    assert result.behavior == Behavior.OTHER
    assert result.participation_events == 2


def test_no_teacher_means_not_visible() -> None:
    result = assign_behavior([], None, BEHAVIOR_CONFIG)
    assert result.behavior == Behavior.NOT_VISIBLE
