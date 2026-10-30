"""Validación anti-invento: toda cita debe existir en la transcripción, en su minuto."""

import re
from dataclasses import dataclass

from espejo.common.text import normalize_text
from espejo.llm.schemas import LABELS_BY_KIND, Evidence, RejectedEvidence, TranscriptLine

MMSS_PATTERN = re.compile(r"(\d+):(\d{1,2})")
SECONDS_PER_MINUTE = 60
INVALID_CLASS_REASON = "clase fuera de la lista permitida"
QUOTE_NOT_FOUND_REASON = "la cita no aparece en la transcripción en ese minuto"


@dataclass(frozen=True)
class ValidationResult:
    accepted: list[Evidence]
    rejected: list[RejectedEvidence]


def parse_mmss(text: str) -> float | None:
    """ "03:12", "[3:12]" -> 192.0. None si no tiene formato de minuto."""
    match = MMSS_PATTERN.search(text)
    if match is None:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    return float(minutes * SECONDS_PER_MINUTE + seconds)


def nearby_texts(lines: list[TranscriptLine], timestamp_s: float, tolerance_s: float) -> list[str]:
    """Textos de las líneas a ±tolerancia, solos y unidos con la siguiente.

    Una cita puede cruzar dos líneas cuando Whisper partió la oración en dos segmentos.
    """
    texts = []
    for index, line in enumerate(lines):
        if abs(line.start_s - timestamp_s) > tolerance_s:
            continue
        texts.append(line.text)
        if index + 1 < len(lines):
            texts.append(line.text + " " + lines[index + 1].text)
    return texts


def quote_exists(
    quote: str, timestamp_s: float, lines: list[TranscriptLine], tolerance_s: float
) -> bool:
    """La cita debe estar textual (sin importar mayúsculas, tildes ni signos) cerca de su minuto."""
    target = normalize_text(quote)
    if not target:
        return False
    for text in nearby_texts(lines, timestamp_s, tolerance_s):
        if target in normalize_text(text):
            return True
    return False


def validate_evidence(
    items: list[Evidence], lines: list[TranscriptLine], tolerance_s: float
) -> ValidationResult:
    accepted = []
    rejected = []
    for item in items:
        if item.label not in LABELS_BY_KIND[item.kind]:
            rejected.append(RejectedEvidence(item, INVALID_CLASS_REASON))
        elif not quote_exists(item.quote, item.timestamp_s, lines, tolerance_s):
            rejected.append(RejectedEvidence(item, QUOTE_NOT_FOUND_REASON))
        else:
            accepted.append(item)
    return ValidationResult(accepted, rejected)


def is_duplicate(item: Evidence, kept: list[Evidence], tolerance_s: float) -> bool:
    """Misma clase y misma cita en el mismo momento: viene de dos ventanas solapadas."""
    for other in kept:
        same_kind = other.kind == item.kind and other.label == item.label
        same_quote = normalize_text(other.quote) == normalize_text(item.quote)
        close_in_time = abs(other.timestamp_s - item.timestamp_s) <= tolerance_s
        if same_kind and same_quote and close_in_time:
            return True
    return False


def deduplicate(items: list[Evidence], tolerance_s: float) -> list[Evidence]:
    kept: list[Evidence] = []
    for item in items:
        if not is_duplicate(item, kept, tolerance_s):
            kept.append(item)
    return kept


def first_label_per_question(items: list[Evidence]) -> dict[int, Evidence]:
    """Si una pregunta cae en dos ventanas solapadas, vale la primera clasificación."""
    labels: dict[int, Evidence] = {}
    for item in items:
        if item.question_id is not None and item.question_id not in labels:
            labels[item.question_id] = item
    return labels
