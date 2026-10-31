"""Formato de tiempos para los reportes."""

SECONDS_PER_MINUTE = 60
SECONDS_DECIMALS = 2


def format_mmss(seconds: float) -> str:
    """Formato mm:ss; los minutos pueden pasar de 59 en clases largas."""
    total_seconds = int(round(seconds))
    minutes = total_seconds // SECONDS_PER_MINUTE
    remaining_seconds = total_seconds % SECONDS_PER_MINUTE
    return f"{minutes:02d}:{remaining_seconds:02d}"


def round_seconds(seconds: float) -> float:
    return round(seconds, SECONDS_DECIMALS)
