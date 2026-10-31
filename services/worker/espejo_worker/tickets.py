"""Verifica el permiso de subida que firma la API de Render (mismo WORKER_TOKEN).

Formato: "<analysis_id>.<vence_en_epoch_s>.<firma>", con firma = HMAC-SHA256 en hex de
"<analysis_id>.<vence_en_epoch_s>". La misma regla está en services/api/app/services/tickets.py.
"""

import hashlib
import hmac

SEPARATOR = "."
TICKET_PARTS = 3


def signature(analysis_id: str, expires_at_s: int, secret: str) -> str:
    message = f"{analysis_id}{SEPARATOR}{expires_at_s}".encode()
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_ticket(ticket: str | None, secret: str, now_s: float) -> str | None:
    """Devuelve el analysis_id si el permiso es válido y no venció; si no, None."""
    if not ticket:
        return None
    parts = ticket.split(SEPARATOR)
    if len(parts) != TICKET_PARTS or not parts[1].isdigit():
        return None
    analysis_id, expires_text, given = parts
    expires_at_s = int(expires_text)
    expected = signature(analysis_id, expires_at_s, secret)
    if not hmac.compare_digest(given, expected) or now_s > expires_at_s:
        return None
    return analysis_id
