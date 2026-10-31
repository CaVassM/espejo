"""Lectura y escritura de audio_raw.json, la salida de VER para el audio."""

from pathlib import Path
from typing import Any

from espejo.audio.schemas import (
    AudioCapture,
    DiarizationOutcome,
    SpeakerTurn,
    SpeechInterval,
    TranscriberSetup,
    TranscriptSegment,
    Word,
)
from espejo.common.json_io import read_json, write_json

AUDIO_RAW_FILENAME = "audio_raw.json"
MISSING_HINT = "Corre primero scripts/analyze_class.py o scripts/analyze_audio.py"


def word_to_dict(word: Word) -> dict[str, Any]:
    return {
        "text": word.text,
        "start_s": word.start_s,
        "end_s": word.end_s,
        "probability": word.probability,
    }


def segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
    words = []
    for word in segment.words:
        words.append(word_to_dict(word))
    return {
        "start_s": segment.start_s,
        "end_s": segment.end_s,
        "text": segment.text,
        "avg_logprob": segment.avg_logprob,
        "no_speech_prob": segment.no_speech_prob,
        "words": words,
    }


def segment_from_dict(raw: dict[str, Any]) -> TranscriptSegment:
    words = []
    for raw_word in raw["words"]:
        words.append(Word(**raw_word))
    return TranscriptSegment(
        start_s=raw["start_s"],
        end_s=raw["end_s"],
        text=raw["text"],
        words=words,
        avg_logprob=raw["avg_logprob"],
        no_speech_prob=raw["no_speech_prob"],
    )


def diarization_to_dict(outcome: DiarizationOutcome) -> dict[str, Any]:
    turns = None
    if outcome.turns is not None:
        turns = []
        for turn in outcome.turns:
            turns.append({"speaker": turn.speaker, "start_s": turn.start_s, "end_s": turn.end_s})
    return {"used": outcome.succeeded(), "note": outcome.note, "turns": turns}


def diarization_from_dict(raw: dict[str, Any]) -> DiarizationOutcome:
    if raw["turns"] is None:
        return DiarizationOutcome(None, raw["note"])

    turns = []
    for raw_turn in raw["turns"]:
        turns.append(SpeakerTurn(**raw_turn))
    return DiarizationOutcome(turns, raw["note"])


def save_audio_capture(capture: AudioCapture, output_path: Path) -> None:
    segments = []
    for segment in capture.segments:
        segments.append(segment_to_dict(segment))
    intervals = []
    for interval in capture.speech_intervals:
        intervals.append({"start_s": interval.start_s, "end_s": interval.end_s})

    content = {
        "video_path": capture.video_path,
        "duration_s": capture.duration_s,
        "language": capture.language,
        "topic": capture.topic,
        "transcriber": {
            "model_name": capture.transcriber.model_name,
            "device": capture.transcriber.device,
            "compute_type": capture.transcriber.compute_type,
        },
        "segments": segments,
        "speech_intervals": intervals,
        "diarization": diarization_to_dict(capture.diarization),
    }
    write_json(content, output_path)


def load_audio_capture(input_path: Path) -> AudioCapture:
    raw = read_json(input_path, MISSING_HINT)

    segments = []
    for raw_segment in raw["segments"]:
        segments.append(segment_from_dict(raw_segment))
    intervals = []
    for raw_interval in raw["speech_intervals"]:
        intervals.append(SpeechInterval(**raw_interval))

    return AudioCapture(
        video_path=raw["video_path"],
        duration_s=raw["duration_s"],
        language=raw["language"],
        transcriber=TranscriberSetup(**raw["transcriber"]),
        segments=segments,
        speech_intervals=intervals,
        diarization=diarization_from_dict(raw["diarization"]),
        topic=raw.get("topic"),
    )
