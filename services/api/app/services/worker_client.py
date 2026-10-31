"""Llamados de la API al worker de Modal: estado de un trabajo y su resultado."""

from typing import Any, Protocol

import httpx

HTTP_NOT_FOUND = 404


class WorkerGateway(Protocol):
    def job_status(self, job_id: str) -> dict[str, Any] | None: ...

    def job_result(self, job_id: str) -> dict[str, Any]: ...


class WorkerClient:
    def __init__(self, base_url: str, worker_token: str, timeout_s: float) -> None:
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {worker_token}"}
        self.timeout_s = timeout_s

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        """None si el worker aún no tiene ese trabajo (el video todavía no se sube)."""
        response = httpx.get(f"{self.base_url}/jobs/{job_id}", timeout=self.timeout_s)
        if response.status_code == HTTP_NOT_FOUND:
            return None
        response.raise_for_status()
        return response.json()

    def job_result(self, job_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/jobs/{job_id}/result", headers=self.headers, timeout=self.timeout_s
        )
        response.raise_for_status()
        return response.json()
