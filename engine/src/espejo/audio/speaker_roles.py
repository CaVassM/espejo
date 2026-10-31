"""Decide qué hablante es el docente."""

from dataclasses import replace

from espejo.audio.schemas import UNKNOWN_SPEAKER, SpeakerRole, TranscriptSegment


def talk_time_by_speaker(segments: list[TranscriptSegment]) -> dict[str, float]:
    """Segundos de habla de cada hablante identificado; ignora los "unknown"."""
    talk_time: dict[str, float] = {}
    for segment in segments:
        if segment.speaker == UNKNOWN_SPEAKER:
            continue
        talk_time[segment.speaker] = talk_time.get(segment.speaker, 0.0) + segment.duration_s()
    return talk_time


def find_teacher_speaker(segments: list[TranscriptSegment]) -> str | None:
    talk_time = talk_time_by_speaker(segments)
    teacher_speaker = None
    for speaker, seconds in talk_time.items():
        if teacher_speaker is None or seconds > talk_time[teacher_speaker]:
            teacher_speaker = speaker
    return teacher_speaker


def role_for_speaker(speaker: str, teacher_speaker: str | None) -> SpeakerRole:
    if speaker == UNKNOWN_SPEAKER or teacher_speaker is None:
        return SpeakerRole.UNKNOWN
    if speaker == teacher_speaker:
        return SpeakerRole.TEACHER
    return SpeakerRole.STUDENT


def assign_roles(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Heurística: el hablante con más tiempo total de voz es el docente; el resto, alumnos.

    En una clase expositiva el docente habla la mayor parte del tiempo, así que la heurística
    suele acertar. Puede fallar en clases centradas en el trabajo de los alumnos (debates,
    exposiciones), donde un alumno podría acumular más tiempo. Sin diarización todos los
    hablantes son "unknown" y ningún segmento recibe un rol.
    """
    teacher_speaker = find_teacher_speaker(segments)

    with_roles = []
    for segment in segments:
        role = role_for_speaker(segment.speaker, teacher_speaker)
        with_roles.append(replace(segment, role=role))
    return with_roles
