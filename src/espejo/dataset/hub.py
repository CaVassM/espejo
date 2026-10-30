"""Acceso al repositorio de SCB-Dataset en Hugging Face."""

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files, snapshot_download

logger = logging.getLogger(__name__)

REPO_TYPE = "dataset"
BYTES_PER_MB = 1_000_000
ARCHIVE_SUFFIX = ".zip"


@dataclass(frozen=True)
class RepoFolder:
    """Carpeta de primer nivel del repositorio remoto."""

    name: str
    file_count: int
    archive_size_mb: float


def summarize_repo_folders(repo_id: str) -> list[RepoFolder]:
    """Agrupa los archivos del repo por carpeta para decidir qué descargar."""
    file_paths = list_repo_files(repo_id, repo_type=REPO_TYPE)
    archive_sizes = _archive_sizes_mb(repo_id, file_paths)

    file_counts: dict[str, int] = {}
    sizes_by_folder: dict[str, float] = {}
    for file_path in file_paths:
        folder_name = file_path.split("/")[0]
        file_counts[folder_name] = file_counts.get(folder_name, 0) + 1
        archive_size = archive_sizes.get(file_path, 0.0)
        sizes_by_folder[folder_name] = sizes_by_folder.get(folder_name, 0.0) + archive_size

    folders = []
    for folder_name in sorted(file_counts):
        folder = RepoFolder(folder_name, file_counts[folder_name], sizes_by_folder[folder_name])
        folders.append(folder)
    return folders


def _archive_sizes_mb(repo_id: str, file_paths: list[str]) -> dict[str, float]:
    """Solo pedimos el tamaño de los .zip: son los únicos archivos pesados."""
    archive_paths = []
    for file_path in file_paths:
        if file_path.endswith(ARCHIVE_SUFFIX):
            archive_paths.append(file_path)

    path_infos = HfApi().get_paths_info(repo_id, archive_paths, repo_type=REPO_TYPE)
    sizes = {}
    for path_info in path_infos:
        sizes[path_info.path] = path_info.size / BYTES_PER_MB
    return sizes


def download_subset(repo_id: str, subset_folder: str, raw_dir: Path) -> Path:
    """Descarga solo el subconjunto pedido; si ya está en disco no lo vuelve a bajar."""
    allow_patterns = [f"{subset_folder}/*{ARCHIVE_SUFFIX}", f"{subset_folder}/*.yaml"]
    logger.info("Descargando %s de %s en %s", subset_folder, repo_id, raw_dir)

    snapshot_download(
        repo_id,
        repo_type=REPO_TYPE,
        allow_patterns=allow_patterns,
        local_dir=raw_dir,
    )
    return raw_dir / subset_folder
