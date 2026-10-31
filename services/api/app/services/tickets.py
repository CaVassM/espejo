"""Permiso de subida: Render lo firma y el worker de Modal lo verifica (mismo WORKER_TOKEN).

Formato: "<analysis_id>.<vence_en_epoch_s>.<firma>", con firma = HMAC-SHA256 en hex de
"<analysis_id>.<vence_en_epoch_s>". La misma regla está en services/worker/espejo_worker/tickets.py.
"""

import hashlib
import hmac

SEPARATOR = "."


def signature(analysis_id: str, expires_at_s: int, secret: str) -> str:
    message = f"{analysis_id}{SEPARATOR}{expires_at_s}".encode()
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def sign_ticket(analysis_id: str, expires_at_s: int, secret: str) -> str:
    signed = signature(analysis_id, expires_at_s, secret)
    return SEPARATOR.join([analysis_id, str(expires_at_s), signed])
