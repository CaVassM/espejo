import pytest

from espejo.audio.questions import QuestionDetector, is_question, is_tag_question
from espejo.audio.schemas import SpeakerRole, TranscriptSegment, Word
from espejo.config import QuestionsConfig

INTERROGATIVES = [
    "qué",
    "cómo",
    "por qué",
    "cuál",
    "cuáles",
    "cuándo",
    "dónde",
    "quién",
    "quiénes",
    "cuánto",
]
TAG_WORDS = ["no", "ya", "cierto", "verdad", "sí", "ok", "ah", "no es cierto"]
CONFIG = QuestionsConfig(
    interrogatives=INTERROGATIVES, join_segments_max_gap_s=1.0, tag_words=TAG_WORDS
)
WORD_DURATION_S = 0.3


def make_words(text: str, start_s: float) -> list[Word]:
    """Una palabra cada 0.3 s, con el espacio inicial que pone Whisper."""
    words = []
    for position, token in enumerate(text.split()):
        word_start_s = start_s + position * WORD_DURATION_S
        words.append(Word(" " + token, word_start_s, word_start_s + WORD_DURATION_S, 0.9))
    return words


def make_segment(
    text: str,
    start_s: float,
    role: SpeakerRole = SpeakerRole.UNKNOWN,
    speaker: str = "unknown",
    low_confidence: bool = False,
) -> TranscriptSegment:
    words = make_words(text, start_s)
    return TranscriptSegment(
        start_s=start_s,
        end_s=words[-1].end_s,
        text=text,
        words=words,
        avg_logprob=-0.3,
        no_speech_prob=0.1,
        speaker=speaker,
        role=role,
        low_confidence=low_confidence,
    )


@pytest.mark.parametrize(
    "text",
    [
        "¿Qué creen que pasará con el agua?",
        "Y esto, ¿por qué pasa?",
        "Cómo resolvemos este problema",
        "Por qué el agua hierve a cien grados",
        "Quiénes terminaron la tarea",
    ],
)
def test_detects_questions(text: str) -> None:
    assert is_question(text, INTERROGATIVES)


@pytest.mark.parametrize(
    "text",
    [
        "Hoy vamos a estudiar el ciclo del agua.",
        "No sé qué hacer con este ejercicio.",
        "Les explico cómo se resuelve.",
        "Que bueno que vinieron todos.",
        "Porque el agua se evapora.",
    ],
)
def test_ignores_statements(text: str) -> None:
    assert not is_question(text, INTERROGATIVES)


def test_question_end_is_last_word_of_the_question() -> None:
    segment = make_segment("Muy bien. ¿Quién quiere pasar?", start_s=10.0)

    questions = QuestionDetector(CONFIG).detect([segment], teacher_only=False)

    assert len(questions) == 1
    assert questions[0].text == "¿Quién quiere pasar?"
    assert questions[0].start_s == pytest.approx(10.6)
    assert questions[0].question_end_s == pytest.approx(11.5)


def test_question_split_between_segments_counts_once() -> None:
    first = make_segment("¿Qué creen", start_s=0.0)
    second = make_segment("que pasará?", start_s=0.7)

    questions = QuestionDetector(CONFIG).detect([first, second], teacher_only=False)

    assert len(questions) == 1
    assert questions[0].text == "¿Qué creen que pasará?"


def test_long_pause_separates_sentences() -> None:
    first = make_segment("Cómo se llama esto", start_s=0.0)
    second = make_segment("Muy bien.", start_s=5.0)

    questions = QuestionDetector(CONFIG).detect([first, second], teacher_only=False)

    assert len(questions) == 1
    assert questions[0].text == "Cómo se llama esto"


def test_teacher_only_skips_student_questions() -> None:
    teacher = make_segment("¿Alguna duda?", 0.0, SpeakerRole.TEACHER, "SPEAKER_00")
    student = make_segment("¿Profe, puedo salir?", 5.0, SpeakerRole.STUDENT, "SPEAKER_01")

    questions = QuestionDetector(CONFIG).detect([teacher, student], teacher_only=True)

    assert len(questions) == 1
    assert questions[0].role == SpeakerRole.TEACHER


def test_question_in_low_confidence_audio_is_not_reliable() -> None:
    segment = make_segment("¿Entendieron?", 0.0, low_confidence=True)
    questions = QuestionDetector(CONFIG).detect([segment], teacher_only=False)
    assert questions[0].reliable is False


def test_segment_without_words_uses_segment_times() -> None:
    segment = TranscriptSegment(
        start_s=3.0,
        end_s=6.0,
        text="Bien. ¿Dónde vive el pez?",
        words=[],
        avg_logprob=-0.3,
        no_speech_prob=0.1,
    )

    questions = QuestionDetector(CONFIG).detect([segment], teacher_only=False)

    assert len(questions) == 1
    assert questions[0].text == "¿Dónde vive el pez?"
    assert questions[0].question_end_s == pytest.approx(6.0)


def question_texts(segments: list[TranscriptSegment]) -> list[str]:
    texts = []
    for question in QuestionDetector(CONFIG).detect(segments, teacher_only=False):
        texts.append(question.text)
    return texts


def test_question_mark_followed_by_comma_splits_questions() -> None:
    segment = make_segment("¿qué está pasando?, ¿qué sucede?, están peleándose", start_s=0.0)
    assert question_texts([segment]) == ["¿qué está pasando?,", "¿qué sucede?,"]


def test_opening_mark_starts_a_new_question() -> None:
    segment = make_segment("Dos hijos, ahí están ¿Qué más?", start_s=0.0)

    questions = QuestionDetector(CONFIG).detect([segment], teacher_only=False)

    assert [question.text for question in questions] == ["¿Qué más?"]
    assert questions[0].start_s == pytest.approx(1.2)


@pytest.mark.parametrize(
    "text",
    ["¿Ya?", "¿No?", "¿Sí?", "¿No es cierto?", "Analizar, ¿no?", "muy bien, ya?", "¿verdad?"],
)
def test_tag_questions(text: str) -> None:
    assert is_tag_question(text, TAG_WORDS)


@pytest.mark.parametrize(
    "text",
    ["¿Por qué no?", "¿Qué pasará?", "¿Cuántos hijos habrá tenido?", "¿felices o tristes?"],
)
def test_real_questions_are_not_tags(text: str) -> None:
    assert not is_tag_question(text, TAG_WORDS)


def test_detector_marks_tag_questions() -> None:
    segment = make_segment("Muy bien, felicitaciones hija ¿No? ¿Qué más sucederá?", start_s=0.0)

    questions = QuestionDetector(CONFIG).detect([segment], teacher_only=False)

    assert [question.is_tag for question in questions] == [True, False]
