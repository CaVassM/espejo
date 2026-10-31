import pytest

from espejo.audio.diarization import DiarizationOutcome
from espejo.audio.hallucinations import HallucinationFilter
from espejo.audio.privacy import (
    Anonymizer,
    Entity,
    ListNameFinder,
    NameSpan,
    default_name_finder,
    is_name,
    load_fallback_names,
    topic_words,
)
from espejo.audio.schemas import (
    AudioCapture,
    SpeakerTurn,
    SpeechInterval,
    TranscriberSetup,
    TranscriptSegment,
    Word,
)
from espejo.config import load_audio_config

AUDIO_CONFIG = load_audio_config()
PRIVACY = AUDIO_CONFIG.privacy


class FakeNameFinder:
    """Marca como nombre cada palabra de la lista, sin spaCy."""

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def find(self, text: str, keep_words: set[str]) -> list[NameSpan]:
        spans = []
        for name in self.names:
            start = text.find(name)
            if start >= 0 and name.lower() not in keep_words:
                spans.append(NameSpan(start, start + len(name), name))
        return spans


def segment(start_s: float, end_s: float, text: str) -> TranscriptSegment:
    words = []
    for index, piece in enumerate(text.split()):
        words.append(Word(" " + piece, start_s + index * 0.2, start_s + index * 0.2 + 0.2, 0.9))
    return TranscriptSegment(start_s, end_s, text, words, -0.2, 0.05)


def capture_with(segments: list[TranscriptSegment], topic: str | None = None) -> AudioCapture:
    turns = [
        SpeakerTurn("SPEAKER_00", 0.0, 10.0),
        SpeakerTurn("SPEAKER_01", 10.0, 14.0),
        SpeakerTurn("SPEAKER_00", 14.0, 30.0),
        SpeakerTurn("SPEAKER_01", 30.0, 31.0),
    ]
    return AudioCapture(
        video_path="clase.mp4",
        duration_s=40.0,
        language="es",
        transcriber=TranscriberSetup("large-v3-turbo", "cuda", "float16"),
        segments=segments,
        speech_intervals=[SpeechInterval(0.0, 14.0), SpeechInterval(14.0, 30.0)],
        diarization=DiarizationOutcome(turns, "diarización de prueba"),
        topic=topic,
    )


def anonymize(segments: list[TranscriptSegment], names: list[str]) -> AudioCapture:
    anonymizer = Anonymizer(PRIVACY, AUDIO_CONFIG.hallucinations, FakeNameFinder(names))
    return anonymizer.anonymize(capture_with(segments, topic="Cuento: Los hijos del labrador"))


def test_student_text_is_replaced_and_keeps_its_time() -> None:
    result = anonymize(
        [
            segment(0.0, 9.0, "A ver, Juanito, ¿qué pasará con los hijos del labrador?"),
            segment(10.5, 13.5, "Que los hijos estarían ayudando a su papá"),
            segment(15.0, 25.0, "Muy bien, felicitaciones hija"),
        ],
        names=["Juanito"],
    )

    student = result.segments[1]
    assert student.text == "[ALUMNO responde]"
    assert student.words == []
    assert (student.start_s, student.end_s) == (10.5, 13.5)
    assert result.segments[0].text == "A ver, [NOMBRE], ¿qué pasará con los hijos del labrador?"
    assert result.segments[2].text == "Muy bien, felicitaciones hija"


def test_names_are_masked_in_the_words_too() -> None:
    result = anonymize([segment(0.0, 9.0, "Gracias María")], names=["María"])

    word_texts = []
    for word in result.segments[0].words:
        word_texts.append(word.text)
    assert word_texts == [" Gracias", "[NOMBRE]"]


def test_invented_student_text_is_dropped_before_losing_it() -> None:
    # "Gracias." sin voz detectada, atribuido a un alumno: es texto inventado por Whisper
    result = anonymize(
        [segment(0.0, 9.0, "Empecemos la clase"), segment(30.0, 31.0, "Gracias.")], names=[]
    )

    assert len(result.segments) == 1


def test_placeholder_is_not_filtered_again_when_interpreting() -> None:
    placeholder = segment(10.5, 13.5, "[ALUMNO responde]")
    assert not HallucinationFilter(AUDIO_CONFIG.hallucinations).is_hallucination(placeholder, [])


def test_name_rule() -> None:
    keep = {"hija", "niños"}
    assert is_name(Entity("Luis", "PER", 0, 4, True), PRIVACY, keep)
    assert not is_name(Entity("¿Qué", "PER", 0, 4, False), PRIVACY, keep)
    assert is_name(Entity("María", "MISC", 0, 5, True), PRIVACY, keep)
    assert is_name(Entity("Pedro", "LOC", 0, 5, True), PRIVACY, keep)
    assert not is_name(Entity("¿qué opinas tú?", "MISC", 0, 14, False), PRIVACY, keep)
    assert not is_name(Entity("Niños", "MISC", 0, 5, True), PRIVACY, keep)


def test_topic_words_are_kept() -> None:
    expected = {"cuento", "los", "hijos", "del", "labrador"}
    assert topic_words("Cuento: Los hijos del labrador") == expected
    assert topic_words(None) == set()


def test_a_multiword_name_from_the_topic_is_kept() -> None:
    keep = topic_words("Canto coral a Túpac Amaru (parte 1)")
    assert not is_name(Entity("Túpac Amaru", "PER", 0, 11, True), PRIVACY, keep)
    # Un nombre que comparte solo una palabra con el tema sí se tapa
    assert is_name(Entity("Juan Amaru", "PER", 0, 10, True), PRIVACY, keep)


def found_names(finder: object, text: str) -> list[str]:
    names = []
    for span in finder.find(text, {"hija", "hijos"}):
        names.append(span.text)
    return names


def test_fallback_list_masks_capitalized_names_only() -> None:
    finder = ListNameFinder(load_fallback_names(PRIVACY))

    assert found_names(finder, "Muy bien, Juanita") == ["Juanita"]
    assert found_names(finder, "A ver, Pedrito") == ["Pedrito"]
    assert found_names(finder, "Quispe, pasa a la pizarra") == ["Quispe"]
    assert found_names(finder, "una rosa roja") == []
    assert found_names(finder, "El labrador tenía dos hijos") == []


def test_invented_phrases_with_spacy() -> None:
    pytest.importorskip("spacy")
    try:
        finder = default_name_finder(PRIVACY)
    except OSError:
        pytest.skip("falta el modelo de spaCy (python -m spacy download es_core_news_md)")

    assert found_names(finder, "Muy bien, Juanita") == ["Juanita"]
    assert found_names(finder, "A ver, Pedrito") == ["Pedrito"]
    assert found_names(finder, "Quispe, pasa a la pizarra") == ["Quispe"]
    assert found_names(finder, "El labrador tenía dos hijos") == []
