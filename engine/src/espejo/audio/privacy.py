"""Privacidad: lo que dicen los alumnos no sale de VER, y los nombres que dice el docente se tapan.

- Todo segmento que no es del docente pasa a "[ALUMNO responde]", con su minuto y su duración.
  El texto original se descarta: nunca se guarda en audio_raw.json.
- En las líneas del docente, los nombres de personas pasan a "[NOMBRE]" (spaCy). El modelo
  pequeño de spaCy confunde las etiquetas: marca "María" o "Pedro" como MISC o LOC, y "¿Qué",
  "Claro" o "Aquí" como PER. Por eso se tapa una entidad PER, MISC o LOC solo si todas sus
  palabras son nombres propios, salvo una lista de palabras a conservar y las del tema.
- Respaldo: una palabra con mayúscula que está en la lista de nombres comunes en el Perú
  (configs/names_pe.yaml: nombres, diminutivos, nombres quechuas y apellidos) también se tapa.
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from espejo.audio.diarization import assign_speakers
from espejo.audio.hallucinations import HallucinationFilter
from espejo.audio.schemas import (
    NAME_PLACEHOLDER,
    STUDENT_PLACEHOLDER,
    AudioCapture,
    SpeakerRole,
    TranscriptSegment,
    Word,
)
from espejo.audio.speaker_roles import assign_roles
from espejo.common.config_io import read_yaml, resolve_project_path
from espejo.common.text import normalize_text
from espejo.config import HallucinationConfig, PrivacyConfig

logger = logging.getLogger(__name__)

PROPER_NOUN_TAG = "PROPN"
PUNCTUATION_TAG = "PUNCT"
# Palabras con letras (incluye tildes y ñ); los signos y números las separan
WORD_PATTERN = re.compile(r"[^\W\d_]+")


@dataclass(frozen=True)
class NameSpan:
    """Un nombre dentro del texto, por posición de caracteres."""

    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class Entity:
    """Lo que spaCy marcó como entidad, con lo necesario para decidir si se tapa."""

    text: str
    label: str
    start_char: int
    end_char: int
    all_proper_nouns: bool


class NameFinder(Protocol):
    def find(self, text: str, keep_words: set[str]) -> list[NameSpan]: ...


def is_name(entity: Entity, config: PrivacyConfig, keep_words: set[str]) -> bool:
    """PER, MISC o LOC hechos solo de nombres propios. Nunca las palabras a conservar."""
    if normalize_text(entity.text) in keep_words:
        return False
    labels = config.person_labels + config.proper_noun_labels
    return entity.label in labels and entity.all_proper_nouns


def topic_words(topic: str | None) -> set[str]:
    """ "Cuento: Los hijos del labrador" -> {"cuento", "los", "hijos", "del", "labrador"}."""
    if not topic:
        return set()
    return set(normalize_text(topic).split())


class SpacyNameFinder:
    """Carga spaCy una sola vez (al crear el objeto), no en cada segmento."""

    def __init__(self, config: PrivacyConfig) -> None:
        # Import local: spaCy es parte de espejo[audio]; INTERPRETAR no lo necesita
        import spacy

        self.config = config
        self.nlp = spacy.load(config.spacy_model)

    def find(self, text: str, keep_words: set[str]) -> list[NameSpan]:
        spans = []
        for spacy_entity in self.nlp(text).ents:
            entity = Entity(
                spacy_entity.text,
                spacy_entity.label_,
                spacy_entity.start_char,
                spacy_entity.end_char,
                only_proper_nouns(spacy_entity),
            )
            if is_name(entity, self.config, keep_words):
                spans.append(NameSpan(entity.start_char, entity.end_char, entity.text))
        return spans


def only_proper_nouns(tokens: Iterable[Any]) -> bool:
    """Al menos un nombre propio y nada más que nombres propios o signos ("¿" solo no cuenta)."""
    has_proper_noun = False
    for token in tokens:
        if token.pos_ == PROPER_NOUN_TAG:
            has_proper_noun = True
        elif token.pos_ != PUNCTUATION_TAG:
            return False
    return has_proper_noun


def load_fallback_names(config: PrivacyConfig) -> set[str]:
    raw = read_yaml(resolve_project_path(config.names_path))
    names = set()
    for name in raw["names"]:
        names.add(normalize_text(name))
    return names


class ListNameFinder:
    """Respaldo de spaCy: palabras con mayúscula que están en la lista de nombres."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def find(self, text: str, keep_words: set[str]) -> list[NameSpan]:
        spans = []
        for match in WORD_PATTERN.finditer(text):
            word = match.group()
            normalized = normalize_text(word)
            if word[0].isupper() and normalized in self.names and normalized not in keep_words:
                spans.append(NameSpan(match.start(), match.end(), word))
        return spans


class CombinedNameFinder:
    """Une lo que encuentran varios buscadores; si dos se solapan, queda el primero."""

    def __init__(self, finders: list[NameFinder]) -> None:
        self.finders = finders

    def find(self, text: str, keep_words: set[str]) -> list[NameSpan]:
        found = []
        for finder in self.finders:
            found += finder.find(text, keep_words)
        return without_overlaps(found)


def without_overlaps(spans: list[NameSpan]) -> list[NameSpan]:
    kept: list[NameSpan] = []
    for span in sorted(spans, key=span_start):
        if not kept or span.start_char >= kept[-1].end_char:
            kept.append(span)
    return kept


def default_name_finder(config: PrivacyConfig) -> NameFinder:
    fallback = ListNameFinder(load_fallback_names(config))
    return CombinedNameFinder([SpacyNameFinder(config), fallback])


def mask_text(text: str, spans: list[NameSpan]) -> str:
    """Reemplaza de atrás hacia adelante para no mover las posiciones que faltan."""
    masked = text
    for span in sorted(spans, key=span_start, reverse=True):
        masked = masked[: span.start_char] + NAME_PLACEHOLDER + masked[span.end_char :]
    return masked


def span_start(span: NameSpan) -> int:
    return span.start_char


def mask_words(words: list[Word], spans: list[NameSpan]) -> list[Word]:
    """Las palabras con tiempo también llevan texto: se tapan las que forman parte de un nombre."""
    name_tokens = set()
    for span in spans:
        name_tokens.update(normalize_text(span.text).split())

    masked = []
    for word in words:
        if normalize_text(word.text) in name_tokens:
            masked.append(replace(word, text=NAME_PLACEHOLDER))
        else:
            masked.append(word)
    return masked


def student_segment(segment: TranscriptSegment) -> TranscriptSegment:
    """Conserva el minuto y la duración; descarta el texto y las palabras."""
    return replace(segment, text=STUDENT_PLACEHOLDER, words=[])


class Anonymizer:
    def __init__(
        self,
        config: PrivacyConfig,
        hallucinations: HallucinationConfig,
        name_finder: NameFinder | None = None,
    ) -> None:
        """name_finder permite inyectar uno simulado en los tests (sin spaCy)."""
        self.config = config
        self.hallucination_filter = HallucinationFilter(hallucinations)
        self.name_finder = name_finder or default_name_finder(config)
        self.base_keep_words = set()
        for word in config.keep_words:
            self.base_keep_words.add(normalize_text(word))

    def anonymize(self, capture: AudioCapture) -> AudioCapture:
        """Usa la misma regla de roles que INTERPRETAR: el que más habla es el docente.

        Sin diarización nadie es docente y todo pasa a "[ALUMNO responde]": se pierde el
        análisis del texto, pero no sale nada de los alumnos.
        """
        keep_words = self.base_keep_words | topic_words(capture.topic)
        labeled = assign_roles(assign_speakers(capture.segments, capture.diarization))

        anonymized = []
        masked_names = 0
        for original, segment in zip(capture.segments, labeled, strict=True):
            if segment.role == SpeakerRole.TEACHER:
                spans = self.name_finder.find(original.text, keep_words)
                masked_names += len(spans)
                masked = mask_text(original.text, spans)
                anonymized.append(
                    replace(original, text=masked, words=mask_words(original.words, spans))
                )
            elif not self.hallucination_filter.is_hallucination(original, capture.speech_intervals):
                # El filtro de texto inventado se aplica antes de perder el texto del alumno
                anonymized.append(student_segment(original))

        logger.info(
            "Privacidad: %d segmentos de alumnos anonimizados, %d nombres tapados",
            len(anonymized) - teacher_count(labeled),
            masked_names,
        )
        return replace(capture, segments=anonymized)


def teacher_count(segments: list[TranscriptSegment]) -> int:
    count = 0
    for segment in segments:
        if segment.role == SpeakerRole.TEACHER:
            count += 1
    return count
