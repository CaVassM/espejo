"""Orientación del docente (de frente o de espaldas) a partir de keypoints de pose."""

from espejo.config import PoseConfig
from espejo.vision.geometry import BoundingBox, iou
from espejo.vision.schemas import Behavior, Orientation, PersonPose, PoseKeypoints


def apply_behavior_prior(pose_orientation: Orientation, behavior: Behavior) -> Orientation:
    """Escribir en la pizarra implica estar de espaldas a la clase, diga lo que diga la pose.

    En una muestra al azar de 16 cajas board_writing del split val, el docente estaba de
    espaldas (o de tres cuartos de espaldas) en las 16. El modelo de pose, en cambio, con el
    docente pequeño y de espaldas suele "ver" la nariz con confianza alta y marcarlo "facing".
    """
    if behavior == Behavior.BOARD_WRITING:
        return Orientation.BACK
    return pose_orientation


def estimate_teacher_orientation(
    teacher_box: BoundingBox | None,
    poses: list[PersonPose],
    config: PoseConfig,
) -> Orientation:
    if teacher_box is None:
        return Orientation.NOT_VISIBLE

    teacher_pose = match_pose(poses, teacher_box, config.match_iou_threshold)
    if teacher_pose is None:
        return Orientation.UNKNOWN

    return classify_orientation(
        teacher_pose.keypoints,
        config.face_kp_conf,
        config.shoulder_kp_conf,
    )


def match_pose(
    poses: list[PersonPose],
    teacher_box: BoundingBox,
    min_iou: float,
) -> PersonPose | None:
    """La pose cuya caja más se solapa con la del docente, si supera el umbral."""
    best_pose = None
    best_iou = min_iou
    for pose in poses:
        overlap = iou(pose.box, teacher_box)
        if overlap >= best_iou:
            best_pose = pose
            best_iou = overlap
    return best_pose


def classify_orientation(
    keypoints: PoseKeypoints,
    face_kp_conf: float,
    shoulder_kp_conf: float,
) -> Orientation:
    """Si se ve la cara, está de frente; si solo se ven los hombros, está de espaldas.

    El modelo de pose casi nunca detecta nariz u ojos en una persona de espaldas,
    así que la ausencia de cara con hombros visibles es una buena señal de "back".
    """
    if is_face_visible(keypoints, face_kp_conf):
        return Orientation.FACING

    if are_shoulders_visible(keypoints, shoulder_kp_conf):
        return Orientation.BACK

    return Orientation.UNKNOWN


def is_face_visible(keypoints: PoseKeypoints, min_confidence: float) -> bool:
    face_points = [keypoints.nose, keypoints.left_eye, keypoints.right_eye]
    for point in face_points:
        if point.confidence >= min_confidence:
            return True
    return False


def are_shoulders_visible(keypoints: PoseKeypoints, min_confidence: float) -> bool:
    """Basta con un hombro: de espaldas y de perfil a veces se oculta el otro."""
    shoulder_points = [keypoints.left_shoulder, keypoints.right_shoulder]
    for point in shoulder_points:
        if point.confidence >= min_confidence:
            return True
    return False
