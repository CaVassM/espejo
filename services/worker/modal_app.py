"""Worker de Espejo en Modal: corre VER (YOLO, Whisper, Silero, pyannote) sobre una clase.

Despliegue:  modal deploy services/worker/modal_app.py
Contrato:    docs/worker_contract.md
"""

import time
from pathlib import Path

import modal
from modal.exception import NotFoundError

from espejo_worker.settings import load_settings

# --- Rutas: en tu PC salen del repo; dentro de los contenedores, de /root/espejo ---
# Las rutas del contenedor son texto con "/": Path en Windows las escribiría con "\"
REMOTE_ROOT = "/root/espejo"
REMOTE_ENGINE = f"{REMOTE_ROOT}/engine"
REMOTE_CONFIGS = f"{REMOTE_ROOT}/configs"
MODELS_DIR = "/models"
JOBS_DIR = "/jobs"
PYTHON_PACKAGES = "/usr/local/lib/python3.11/site-packages"
CONFIGS_DIR = Path(REMOTE_CONFIGS)
ENGINE_DIR = Path(REMOTE_ENGINE)
if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    CONFIGS_DIR = REPO_ROOT / "configs"
    ENGINE_DIR = REPO_ROOT / "engine"

SETTINGS = load_settings(CONFIGS_DIR / "worker.yaml")
REQUIRED_SECRET_KEYS = ["HF_TOKEN", "WORKER_TOKEN", "UPLOAD_ACCESS_CODE"]
CACHE_IGNORE = ["**/__pycache__", "**/*.egg-info", "**/.pytest_cache"]

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]>=0.115", "python-multipart>=0.0.9", "pyyaml>=6.0")
    .add_local_dir(CONFIGS_DIR, REMOTE_CONFIGS)
    .add_local_python_source("espejo_worker")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    # El motor se copia en la imagen y se instala editable: así configs/ queda en su lugar
    .add_local_dir(ENGINE_DIR, REMOTE_ENGINE, copy=True, ignore=CACHE_IGNORE)
    .run_commands(
        f"pip install -e '{REMOTE_ENGINE}[vision,audio]' 'torch==2.5.1' 'torchaudio==2.5.1'",
        "python -m spacy download es_core_news_md",
        # Los pesos viven en el Volume; el motor los busca en weights/ (configs/vision.yaml)
        f"ln -s {MODELS_DIR}/weights {REMOTE_ROOT}/weights",
    )
    .env(
        {
            # Whisper y pyannote se descargan una vez y quedan en el Volume
            "HF_HOME": f"{MODELS_DIR}/hf",
            # ctranslate2 (Whisper) usa la cuDNN y la cuBLAS que trae torch
            "LD_LIBRARY_PATH": f"{PYTHON_PACKAGES}/nvidia/cudnn/lib:"
            f"{PYTHON_PACKAGES}/nvidia/cublas/lib",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
        }
    )
    .add_local_dir(CONFIGS_DIR, REMOTE_CONFIGS)
    .add_local_python_source("espejo_worker")
)

app = modal.App(SETTINGS.app_name)
models_volume = modal.Volume.from_name(SETTINGS.models_volume, create_if_missing=True)
jobs_volume = modal.Volume.from_name(SETTINGS.jobs_volume, create_if_missing=True)
jobs_store = modal.Dict.from_name(SETTINGS.jobs_dict, create_if_missing=True)
secrets = modal.Secret.from_name(SETTINGS.secret_name, required_keys=REQUIRED_SECRET_KEYS)


@app.cls(
    image=gpu_image,
    gpu=SETTINGS.gpu,
    timeout=SETTINGS.timeout_s,
    scaledown_window=SETTINGS.scaledown_window_s,
    max_containers=SETTINGS.max_containers,
    volumes={MODELS_DIR: models_volume, JOBS_DIR: jobs_volume},
    secrets=[secrets],
)
class Analyzer:
    @modal.enter()
    def load_models(self) -> None:
        """Una vez por contenedor, no en cada video."""
        import logging

        from espejo_worker.pipeline import VerPipeline

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        started = time.perf_counter()
        self.pipeline = VerPipeline()
        self.models_load_s = time.perf_counter() - started
        self.videos_done = 0
        # Guarda en el Volume lo que se descargó la primera vez (Whisper, pyannote)
        models_volume.commit()

    @modal.method()
    def process(self, job_id: str) -> None:
        from espejo_worker.processing import ContainerInfo, ProcessingDeps, process_job
        from espejo_worker.settings import load_secrets

        deps = ProcessingDeps(
            store=jobs_store,
            jobs_dir=Path(JOBS_DIR),
            pipeline=self.pipeline,
            reload=jobs_volume.reload,
            commit=jobs_volume.commit,
            settings=SETTINGS,
            secrets=load_secrets(),
        )
        cold_start = self.videos_done == 0
        container = ContainerInfo(cold_start=cold_start, models_load_s=self.models_load_s)
        process_job(job_id, deps, container)
        self.videos_done += 1


def read_job_file(relative_path: str) -> bytes | None:
    """Lee del Volume por la API (sin montar): ve lo último que guardó el contenedor con GPU."""
    try:
        return b"".join(jobs_volume.read_file(relative_path))
    except (FileNotFoundError, NotFoundError):
        return None


def spawn_job(job_id: str) -> None:
    Analyzer().process.spawn(job_id)


@app.function(
    image=web_image,
    volumes={JOBS_DIR: jobs_volume},
    secrets=[secrets],
    timeout=SETTINGS.web_timeout_s,
    scaledown_window=SETTINGS.web_scaledown_window_s,
)
@modal.concurrent(max_inputs=SETTINGS.web_max_inputs)
@modal.asgi_app()
def web():
    from espejo_worker.settings import load_secrets
    from espejo_worker.web import WebDeps, create_app

    deps = WebDeps(
        store=jobs_store,
        jobs_dir=Path(JOBS_DIR),
        spawn=spawn_job,
        commit=jobs_volume.commit,
        read_file=read_job_file,
        settings=SETTINGS,
        secrets=load_secrets(),
    )
    return create_app(deps)


@app.function(image=web_image, volumes={JOBS_DIR: jobs_volume}, schedule=modal.Period(days=1))
def clean_old_jobs() -> None:
    """Una vez al día: borra los resultados con más de job_ttl_days (el estado expira solo)."""
    from espejo_worker.jobs import remove_expired_jobs

    jobs_volume.reload()
    removed = remove_expired_jobs(Path(JOBS_DIR), SETTINGS.job_ttl_days, time.time())
    jobs_volume.commit()
    print(f"Trabajos vencidos borrados: {len(removed)}")
