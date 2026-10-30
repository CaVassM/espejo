"""Serialización del reporte de visión a JSON y resumen legible para consola."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from espejo.common.json_io import write_json
from espejo.common.time_format import format_mmss, round_seconds
from espejo.vision.schemas import Segment, VisionReport

RATIO_DECIMALS = 3


def segment_to_dict(segment: Segment) -> dict[str, Any]:
    return {
        "label": segment.label,
        "start_s": round_seconds(segment.start_s),
        "end_s": round_seconds(segment.end_s),
        "start": format_mmss(segment.start_s),
        "end": format_mmss(segment.end_s),
        "duration_s": round_seconds(segment.duration_s()),
    }


def timeline_to_list(timeline: list[Segment]) -> list[dict[str, Any]]:
    serialized = []
    for segment in timeline:
        serialized.append(segment_to_dict(segment))
    return serialized


def metrics_to_dict(report: VisionReport) -> dict[str, Any]:
    if report.metrics is None:
        return {}

    rounded = {}
    for name, value in asdict(report.metrics).items():
        if isinstance(value, float):
            rounded[name] = round(value, RATIO_DECIMALS)
        else:
            rounded[name] = value
    rounded["analyzed_duration"] = format_mmss(report.metrics.analyzed_duration_s)
    return rounded


def report_to_dict(report: VisionReport) -> dict[str, Any]:
    video = report.video
    metadata = {
        "video": Path(video.path).name,
        "duration_s": round_seconds(video.duration_s),
        "duration": format_mmss(video.duration_s),
        "native_fps": round_seconds(video.native_fps),
        "resolution": {"width": video.width, "height": video.height},
        "sample_fps": report.sample_fps,
        "models": report.models,
    }
    return {
        "metadata": metadata,
        "behavior_timeline": timeline_to_list(report.behavior_timeline),
        "orientation_timeline": timeline_to_list(report.orientation_timeline),
        "metrics": metrics_to_dict(report),
    }


def save_report_json(report: VisionReport, output_path: Path) -> None:
    write_json(report_to_dict(report), output_path)


def format_metrics_summary(report: VisionReport) -> str:
    """Resumen para mostrar al final del CLI."""
    metrics = report.metrics
    if metrics is None:
        return "Sin métricas."

    threshold_text = "no cumple"
    if metrics.meets_monitoring_threshold:
        threshold_text = "cumple"

    lines = [
        f"Video: {Path(report.video.path).name} ({format_mmss(report.video.duration_s)})",
        f"Duración analizada:        {format_mmss(metrics.analyzed_duration_s)}",
        f"Docente visible:           {metrics.teacher_visible_ratio:.1%}",
        f"Monitoreo (guide):         {metrics.monitoring_ratio:.1%}  ({threshold_text} el 25 %)",
        f"Escribiendo en pizarra:    {metrics.board_writing_ratio:.1%}",
        f"De espaldas a la clase:    {metrics.back_to_class_ratio:.1%}",
        f"Recorrido (anchos frame):  {metrics.path_length_normalized:.2f}",
        f"Cobertura del aula:        {metrics.room_coverage_ratio:.1%}",
        f"Eventos de participación:  {metrics.student_participation_events}",
        f"Segmentos de conducta:     {len(report.behavior_timeline)}",
        f"Segmentos de orientación:  {len(report.orientation_timeline)}",
    ]
    return "\n".join(lines)
