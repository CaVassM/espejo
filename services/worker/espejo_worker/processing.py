"""Un trabajo de principio a fin: VER, guardar resultados, borrar el video y avisar al backend."""

import logging
import shutil
import tempfile
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from espejo_worker.jobs import Stage, with_stage
from espejo_worker.results import RESULT_FILENAMES, build_result, callback_url, send_callback
from espejo_worker.settings import WorkerSecrets, WorkerSettings

logger = logging.getLogger(__name__)

ERROR_MESSAGE = "No se pudo procesar el video ({error_name})."
CALLBACK_SENT = "sent"
CALLBACK_FAILED = "failed"


class Pipeline(Protocol):
    def run(
        self, video_path: Path, run_dir: Path, topic: str | None, on_stage: Callable[[Stage], None]
    ) -> None: ...


@dataclass
class ProcessingDeps:
    store: MutableMapping[str, Any]
    jobs_dir: Path
    pipeline: Pipeline
    reload: Callable[[], None]
    commit: Callable[[], None]
    settings: WorkerSettings
    secrets: WorkerSecrets
    now: Callable[[], float] = field(default=time.time)


@dataclass(frozen=True)
class ContainerInfo:
    """Arranque en frío: el primer video de un contenedor paga la carga de los modelos."""

    cold_start: bool
    models_load_s: float


def copy_results(run_dir: Path, job_dir: Path) -> None:
    for filename in RESULT_FILENAMES:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, job_dir / filename)


def read_local(jobs_dir: Path) -> Callable[[str], bytes | None]:
    def read_file(relative_path: str) -> bytes | None:
        path = jobs_dir / relative_path
        if not path.exists():
            return None
        return path.read_bytes()

    return read_file


def process_job(job_id: str, deps: ProcessingDeps, container: ContainerInfo) -> None:
    """Nunca lanza: un error queda en el estado del trabajo. El video se borra siempre."""
    job = deps.store.get(job_id)
    if job is None:
        logger.error("No existe el trabajo %s", job_id)
        return

    deps.reload()
    job_dir = deps.jobs_dir / job_id
    video_path = job_dir / job["video_file"]
    run_dir = Path(tempfile.mkdtemp(prefix="espejo-"))

    def on_stage(stage: Stage) -> None:
        deps.store[job_id] = with_stage(deps.store[job_id], stage, deps.now())

    succeeded = False
    try:
        deps.pipeline.run(video_path, run_dir, job["topic"], on_stage)
        copy_results(run_dir, job_dir)
        succeeded = True
    except Exception as error:
        # Cualquier fallo (video dañado, sin memoria, etc.) se informa en el estado
        logger.exception("Falló el trabajo %s", job_id)
        message = ERROR_MESSAGE.format(error_name=type(error).__name__)
        deps.store[job_id] = with_stage(deps.store[job_id], Stage.ERROR, deps.now(), message)
    finally:
        # El video y el audio no se guardan: se borran al terminar o si algo falla
        video_path.unlink(missing_ok=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        deps.commit()

    if succeeded:
        finish(job_id, deps, container)


def finish(job_id: str, deps: ProcessingDeps, container: ContainerInfo) -> None:
    job = with_stage(deps.store[job_id], Stage.DONE, deps.now())
    job["timings"]["cold_start"] = container.cold_start
    job["timings"]["models_load_s"] = round(container.models_load_s, 1)
    deps.store[job_id] = job

    if deps.secrets.api_base_url is None:
        return
    analysis_id = job["analysis_id"] or job_id
    payload = build_result(job_id, job["analysis_id"], read_local(deps.jobs_dir))
    sent = send_callback(
        callback_url(deps.secrets.api_base_url, analysis_id),
        deps.secrets.worker_token,
        payload,
        deps.settings.callback_attempts,
        deps.settings.callback_timeout_s,
        deps.settings.callback_backoff_s,
    )
    job = dict(deps.store[job_id])
    job["callback"] = CALLBACK_SENT if sent else CALLBACK_FAILED
    deps.store[job_id] = job
