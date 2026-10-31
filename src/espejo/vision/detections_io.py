"""Lectura y escritura de detections.json, la salida de VER para el video."""

from pathlib import Path
from typing import Any

from espejo.common.json_io import read_json, write_json
from espejo.vision.geometry import BoundingBox
from espejo.vision.schemas import (
    Detection,
    DetectionRun,
    FrameDetections,
    Keypoint,
    PersonPose,
    PoseKeypoints,
    VideoInfo,
)

DETECTIONS_FILENAME = "detections.json"
BACKGROUND_FILENAME = "background_blurred.png"
KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder")
MISSING_HINT = "Corre primero scripts/analyze_class.py o scripts/analyze_video.py"


def box_to_list(box: BoundingBox) -> list[float]:
    return [box.x1, box.y1, box.x2, box.y2]


def box_from_list(values: list[float]) -> BoundingBox:
    x1, y1, x2, y2 = values
    return BoundingBox(x1, y1, x2, y2)


def detection_to_dict(detection: Detection) -> dict[str, Any]:
    return {
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "box": box_to_list(detection.box),
        "track_id": detection.track_id,
    }


def detection_from_dict(raw: dict[str, Any]) -> Detection:
    return Detection(
        box=box_from_list(raw["box"]),
        class_name=raw["class_name"],
        confidence=raw["confidence"],
        track_id=raw["track_id"],
    )


def pose_to_dict(pose: PersonPose) -> dict[str, Any]:
    keypoints = {}
    for name in KEYPOINT_NAMES:
        point = getattr(pose.keypoints, name)
        keypoints[name] = [point.x, point.y, point.confidence]
    return {"box": box_to_list(pose.box), "keypoints": keypoints}


def pose_from_dict(raw: dict[str, Any]) -> PersonPose:
    points = {}
    for name in KEYPOINT_NAMES:
        x, y, confidence = raw["keypoints"][name]
        points[name] = Keypoint(x, y, confidence)
    return PersonPose(box_from_list(raw["box"]), PoseKeypoints(**points))


def detections_to_list(detections: list[Detection]) -> list[dict[str, Any]]:
    serialized = []
    for detection in detections:
        serialized.append(detection_to_dict(detection))
    return serialized


def detections_from_list(raw_list: list[dict[str, Any]]) -> list[Detection]:
    detections = []
    for raw in raw_list:
        detections.append(detection_from_dict(raw))
    return detections


def frame_to_dict(frame: FrameDetections) -> dict[str, Any]:
    poses = []
    for pose in frame.poses:
        poses.append(pose_to_dict(pose))
    return {
        "frame_index": frame.frame_index,
        "timestamp_s": frame.timestamp_s,
        "teacher": detections_to_list(frame.teacher_detections),
        "behaviors": detections_to_list(frame.behavior_detections),
        "poses": poses,
    }


def frame_from_dict(raw: dict[str, Any]) -> FrameDetections:
    poses = []
    for raw_pose in raw.get("poses", []):
        poses.append(pose_from_dict(raw_pose))
    return FrameDetections(
        frame_index=raw["frame_index"],
        timestamp_s=raw["timestamp_s"],
        teacher_detections=detections_from_list(raw["teacher"]),
        behavior_detections=detections_from_list(raw["behaviors"]),
        poses=poses,
    )


def save_detection_run(run: DetectionRun, output_path: Path) -> None:
    frames = []
    for frame in run.frames:
        frames.append(frame_to_dict(frame))

    content = {
        "video": {
            "path": run.video.path,
            "duration_s": run.video.duration_s,
            "native_fps": run.video.native_fps,
            "width": run.video.width,
            "height": run.video.height,
        },
        "sample_fps": run.sample_fps,
        "models": run.models,
        "frames": frames,
    }
    write_json(content, output_path)


def load_detection_run(input_path: Path) -> DetectionRun:
    raw = read_json(input_path, MISSING_HINT)
    frames = []
    for raw_frame in raw["frames"]:
        frames.append(frame_from_dict(raw_frame))

    return DetectionRun(
        video=VideoInfo(**raw["video"]),
        sample_fps=raw["sample_fps"],
        models=raw["models"],
        frames=frames,
    )
