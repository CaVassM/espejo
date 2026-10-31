"""Normalización de texto en español para comparar frases sin importar tildes ni signos."""

import re
import unicodedata

NON_ALPHANUMERIC = re.compile(r"[^a-z0-9ñ ]+")
MULTIPLE_SPACES = re.compile(r"\s+")
COMBINING_MARK_CATEGORY = "Mn"


def strip_accents(text: str) -> str:
    """Quita tildes pero conserva la ñ (que en NFD es n + virgulilla)."""
    protected = text.replace("ñ", "\0").replace("Ñ", "\1")
    decomposed = unicodedata.normalize("NFD", protected)
    kept = []
    for character in decomposed:
        if unicodedata.category(character) != COMBINING_MARK_CATEGORY:
            kept.append(character)
    return "".join(kept).replace("\0", "ñ").replace("\1", "Ñ")


def normalize_text(text: str) -> str:
    """Minúsculas, sin tildes, sin signos y con espacios simples: "¿Sí?" -> "si"."""
    lowered = strip_accents(text).lower()
    without_punctuation = NON_ALPHANUMERIC.sub(" ", lowered)
    return MULTIPLE_SPACES.sub(" ", without_punctuation).strip()
