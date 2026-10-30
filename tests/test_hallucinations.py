from espejo.audio.hallucinations import HallucinationFilter
from espejo.audio.schemas import SpeechInterval, TranscriptSegment
from espejo.config import HallucinationConfig

CONFIG = HallucinationConfig(
    min_vad_overlap=0.3, max_words_for_vad_check=3, phrases=["gracias", "gracias por ver"]
)
VOICE = [SpeechInterval(0.0, 10.0)]


def make_segment(text: str, start_s: float, end_s: float) -> TranscriptSegment:
    return TranscriptSegment(
        start_s=start_s,
        end_s=end_s,
        text=text,
        words=[],
        avg_logprob=-0.3,
        no_speech_prob=0.1,
    )


def test_segment_with_voice_is_kept() -> None:
    segment = make_segment("Vamos a leer un cuento", 2.0, 5.0)
    assert not HallucinationFilter(CONFIG).is_hallucination(segment, VOICE)


def test_short_segment_without_voice_is_discarded() -> None:
    # Solo 1 de 10 s coincide con voz: Whisper lo inventó sobre silencio
    segment = make_segment("Y así termina", 9.0, 19.0)
    assert HallucinationFilter(CONFIG).is_hallucination(segment, VOICE)


def test_long_segment_without_detected_voice_is_kept() -> None:
    # Silero no oye bien voces lejanas: una respuesta larga sin voz detectada se conserva
    segment = make_segment("Que los hijos estarían ayudando a su labrador", 12.0, 16.0)
    assert not HallucinationFilter(CONFIG).is_hallucination(segment, VOICE)


def test_typical_phrase_is_discarded_even_with_voice() -> None:
    segment = make_segment("¡Gracias!", 2.0, 3.0)
    assert HallucinationFilter(CONFIG).is_hallucination(segment, VOICE)


def test_phrase_inside_real_sentence_is_kept() -> None:
    segment = make_segment("Gracias, hija, muy bien respondido", 2.0, 5.0)
    assert not HallucinationFilter(CONFIG).is_hallucination(segment, VOICE)


def test_filter_splits_kept_and_discarded() -> None:
    segments = [make_segment("Vamos a leer", 1.0, 3.0), make_segment("Gracias.", 4.0, 5.0)]

    result = HallucinationFilter(CONFIG).filter(segments, VOICE)

    assert len(result.kept) == 1
    assert len(result.discarded) == 1
