"""transcript.txt: una línea por segmento, pensado para el LLM de la siguiente fase."""

from pathlib import Path

from espejo.audio.schemas import SpeakerRole, TranscriptSegment
from espejo.common.time_format import format_mmss

LOW_CONFIDENCE_TAG = "[AUDIO DUDOSO] "
# Sin diarización no sabemos quién habla: todos los segmentos son "HABLANTE"
ROLE_LABELS = {
    SpeakerRole.TEACHER: "DOCENTE",
    SpeakerRole.STUDENT: "ALUMNO",
    SpeakerRole.UNKNOWN: "HABLANTE",
}


def transcript_line(segment: TranscriptSegment) -> str:
    """Ejemplo: "[03:12] DOCENTE: ¿Qué creen que pasará con el agua?"."""
    tag = ""
    if segment.low_confidence:
        tag = LOW_CONFIDENCE_TAG
    label = ROLE_LABELS[segment.role]
    return f"[{format_mmss(segment.start_s)}] {label}: {tag}{segment.text}"


def save_transcript(segments: list[TranscriptSegment], output_path: Path) -> None:
    lines = []
    for segment in segments:
        lines.append(transcript_line(segment))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
