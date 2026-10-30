"""Serialización del reporte de visión a JSON y resumen legible para consola."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from espejo.vision.schemas import Segment, VisionReport

SECONDS_PER_MINUTE = 60
RATIO_DECIMALS = 3
SECONDS_DECIMALS = 2


def format_mmss(seconds: float) -> str:
    """Formato mm:ss; los minutos pueden pasar de 59 en clases largas."""
    total_seconds = int(round(seconds))
    minutes = total_seconds // SECONDS_PER_MINUTE
    remaining_seconds = total_seconds % SECONDS_PER_MINUTE
    return f"{minutes:02d}:{remaining_seconds:02d}"


def segment_to_dict(segment: Segment) -> dict[str, Any]:
    return {
        "label": segment.label,
        "start_s": round(segment.start_s, SECONDS_DECIMALS),
        "end_s": round(segment.end_s, SECONDS_DECIMALS),
        "start": format_mmss(segment.start_s),
        "end": format_mmss(segment.end_s),
        "duration_s": round(segment.duration_s(), SECONDS_DECIMALS),
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
        "duration_s": round(video.duration_s, SECONDS_DECIMALS),
        "duration": format_mmss(video.duration_s),
        "native_fps": round(video.native_fps, SECONDS_DECIMALS),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report_to_dict(report), indent=2, ensure_ascii=False)
    output_path.write_text(content, encoding="utf-8")


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
