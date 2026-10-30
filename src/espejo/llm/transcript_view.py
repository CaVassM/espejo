"""Lo que ve el LLM: líneas "[mm:ss] DOCENTE: texto" agrupadas en ventanas con solapamiento."""

from dataclasses import dataclass

from espejo.analysis.transcript import ROLE_LABELS
from espejo.audio.schemas import Question, TranscriptSegment
from espejo.common.time_format import format_mmss
from espejo.config import WindowConfig
from espejo.llm.schemas import CandidateQuestion, TranscriptLine


@dataclass(frozen=True)
class TranscriptWindow:
    start_s: float
    end_s: float
    lines: list[TranscriptLine]
    candidates: list[CandidateQuestion]


def lines_from_segments(segments: list[TranscriptSegment]) -> list[TranscriptLine]:
    lines = []
    for segment in segments:
        label = ROLE_LABELS[segment.role]
        lines.append(TranscriptLine(segment.start_s, segment.end_s, label, segment.text))
    return lines


def candidates_from_questions(questions: list[Question]) -> list[CandidateQuestion]:
    """El id es la posición en la lista de preguntas del audio, para aplicar luego la clase."""
    candidates = []
    for question_id, question in enumerate(questions):
        candidates.append(
            CandidateQuestion(question_id, question.start_s, question.text, question.is_tag)
        )
    return candidates


def format_line(line: TranscriptLine) -> str:
    return f"[{format_mmss(line.start_s)}] {line.speaker_label}: {line.text}"


def format_candidate(candidate: CandidateQuestion) -> str:
    return f"{candidate.question_id} | [{format_mmss(candidate.start_s)}] | {candidate.text}"


def build_windows(
    lines: list[TranscriptLine], candidates: list[CandidateQuestion], config: WindowConfig
) -> list[TranscriptWindow]:
    """Ventanas de window_s que avanzan window_s - overlap_s; las vacías se omiten."""
    if not lines:
        return []

    step_s = config.window_s - config.overlap_s
    last_start_s = lines[-1].start_s
    windows = []
    window_start_s = 0.0
    while window_start_s <= last_start_s:
        window_end_s = window_start_s + config.window_s
        window_lines = lines_between(lines, window_start_s, window_end_s)
        if window_lines:
            window_candidates = candidates_between(candidates, window_start_s, window_end_s)
            windows.append(
                TranscriptWindow(window_start_s, window_end_s, window_lines, window_candidates)
            )
        window_start_s += step_s
    return windows


def lines_between(
    lines: list[TranscriptLine], start_s: float, end_s: float
) -> list[TranscriptLine]:
    selected = []
    for line in lines:
        if start_s <= line.start_s < end_s:
            selected.append(line)
    return selected


def candidates_between(
    candidates: list[CandidateQuestion], start_s: float, end_s: float
) -> list[CandidateQuestion]:
    selected = []
    for candidate in candidates:
        if start_s <= candidate.start_s < end_s:
            selected.append(candidate)
    return selected
