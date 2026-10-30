from espejo.config import PoseConfig
from espejo.vision.geometry import BoundingBox
from espejo.vision.orientation import classify_orientation, estimate_teacher_orientation
from espejo.vision.schemas import Keypoint, Orientation, PersonPose, PoseKeypoints

FACE_CONF = 0.5
SHOULDER_CONF = 0.5
VISIBLE = 0.9
HIDDEN = 0.1

POSE_CONFIG = PoseConfig(
    detection_conf=0.25,
    match_iou_threshold=0.3,
    face_kp_conf=FACE_CONF,
    shoulder_kp_conf=SHOULDER_CONF,
)


def make_keypoints(face_conf: float, shoulder_conf: float) -> PoseKeypoints:
    face_point = Keypoint(0, 0, face_conf)
    shoulder_point = Keypoint(0, 0, shoulder_conf)
    return PoseKeypoints(
        nose=face_point,
        left_eye=face_point,
        right_eye=face_point,
        left_shoulder=shoulder_point,
        right_shoulder=shoulder_point,
    )


def test_visible_face_means_facing() -> None:
    keypoints = make_keypoints(face_conf=VISIBLE, shoulder_conf=VISIBLE)
    assert classify_orientation(keypoints, FACE_CONF, SHOULDER_CONF) == Orientation.FACING


def test_shoulders_without_face_means_back() -> None:
    keypoints = make_keypoints(face_conf=HIDDEN, shoulder_conf=VISIBLE)
    assert classify_orientation(keypoints, FACE_CONF, SHOULDER_CONF) == Orientation.BACK


def test_nothing_visible_is_unknown() -> None:
    keypoints = make_keypoints(face_conf=HIDDEN, shoulder_conf=HIDDEN)
    assert classify_orientation(keypoints, FACE_CONF, SHOULDER_CONF) == Orientation.UNKNOWN


def test_only_nose_is_enough_for_facing() -> None:
    hidden = Keypoint(0, 0, HIDDEN)
    keypoints = PoseKeypoints(
        nose=Keypoint(0, 0, VISIBLE),
        left_eye=hidden,
        right_eye=hidden,
        left_shoulder=hidden,
        right_shoulder=hidden,
    )
    assert classify_orientation(keypoints, FACE_CONF, SHOULDER_CONF) == Orientation.FACING


def test_missing_teacher_is_not_visible() -> None:
    assert estimate_teacher_orientation(None, [], POSE_CONFIG) == Orientation.NOT_VISIBLE


def test_pose_far_from_teacher_is_ignored() -> None:
    teacher_box = BoundingBox(0, 0, 100, 200)
    other_person = PersonPose(BoundingBox(500, 0, 600, 200), make_keypoints(VISIBLE, VISIBLE))
    orientation = estimate_teacher_orientation(teacher_box, [other_person], POSE_CONFIG)
    assert orientation == Orientation.UNKNOWN


def test_pose_matching_teacher_is_used() -> None:
    teacher_box = BoundingBox(0, 0, 100, 200)
    teacher_pose = PersonPose(BoundingBox(5, 5, 100, 200), make_keypoints(HIDDEN, VISIBLE))
    orientation = estimate_teacher_orientation(teacher_box, [teacher_pose], POSE_CONFIG)
    assert orientation == Orientation.BACK
