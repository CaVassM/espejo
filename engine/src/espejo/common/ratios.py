"""Divisiones seguras para métricas."""


def safe_ratio(numerator: float, denominator: float) -> float:
    """Devuelve 0 cuando no hay nada que medir, en vez de fallar por división entre cero."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator
