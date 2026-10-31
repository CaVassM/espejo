from espejo.audio.diarization import assign_speakers, speaker_for_segment
from espejo.audio.schemas import (
    UNKNOWN_SPEAKER,
    DiarizationOutcome,
    SpeakerRole,
    SpeakerTurn,
    TranscriptSegment,
)
from espejo.audio.speaker_roles import assign_roles


def make_segment(start_s: float, end_s: float, speaker: str = UNKNOWN_SPEAKER) -> TranscriptSegment:
    return TranscriptSegment(
        start_s=start_s,
        end_s=end_s,
        text="texto",
        words=[],
        avg_logprob=-0.3,
        no_speech_prob=0.1,
        speaker=speaker,
    )


def roles_of(segments: list[TranscriptSegment]) -> list[SpeakerRole]:
    roles = []
    for segment in segments:
        roles.append(segment.role)
    return roles


def test_speaker_with_most_talk_time_is_teacher() -> None:
    # SPEAKER_01 habla menos veces pero acumula más tiempo (40 s contra 9 s)
    segments = [
        make_segment(0, 3, "SPEAKER_00"),
        make_segment(3, 43, "SPEAKER_01"),
        make_segment(43, 46, "SPEAKER_00"),
        make_segment(46, 49, "SPEAKER_00"),
        make_segment(49, 50, UNKNOWN_SPEAKER),
    ]

    with_roles = assign_roles(segments)

    assert roles_of(with_roles) == [
        SpeakerRole.STUDENT,
        SpeakerRole.TEACHER,
        SpeakerRole.STUDENT,
        SpeakerRole.STUDENT,
        SpeakerRole.UNKNOWN,
    ]


def test_without_diarization_everything_is_unknown() -> None:
    segments = [make_segment(0, 5), make_segment(5, 9)]

    with_roles = assign_roles(segments)

    assert roles_of(with_roles) == [SpeakerRole.UNKNOWN, SpeakerRole.UNKNOWN]


def test_segment_takes_speaker_with_most_overlap() -> None:
    turns = [SpeakerTurn("SPEAKER_00", 0.0, 4.0), SpeakerTurn("SPEAKER_01", 4.0, 10.0)]
    assert speaker_for_segment(make_segment(3.0, 8.0), turns) == "SPEAKER_01"


def test_failed_diarization_keeps_unknown_speakers() -> None:
    segments = [make_segment(0, 5)]
    outcome = DiarizationOutcome(turns=None, note="diarización desactivada")

    labeled = assign_speakers(segments, outcome)

    assert labeled[0].speaker == UNKNOWN_SPEAKER
