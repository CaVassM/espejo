"""Clave del docente (teacher_key): con ella su asistente de IA ve sus análisis, y nada más.

Se guarda solo su hash (SHA-256): si alguien lee la base de datos, no puede usar las claves.
Nunca se registra en los logs.
"""

import hashlib
import re
import secrets

KEY_BYTES = 32  # 43 caracteres en base64 url-safe
# Solo letras, números, "-" y "_", y bien larga: así no se adivina ni se confunde con otra ruta
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


def new_teacher_key() -> str:
    return secrets.token_urlsafe(KEY_BYTES)


def hash_teacher_key(teacher_key: str) -> str:
    return hashlib.sha256(teacher_key.encode("utf-8")).hexdigest()


def looks_like_key(text: str) -> bool:
    return bool(KEY_PATTERN.match(text))
