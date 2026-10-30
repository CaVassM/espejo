"""Descompresión de los .zip descargados."""

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def find_archive(subset_dir: Path) -> Path:
    archives = sorted(subset_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            f"No se encontró ningún .zip en {subset_dir}. "
            "Descárgalo primero con scripts/inspect_dataset.py --download"
        )
    return archives[0]


def extract_archive(archive_path: Path, destination_dir: Path) -> Path:
    """Extrae una sola vez; devuelve la carpeta raíz que trae el zip."""
    with zipfile.ZipFile(archive_path) as archive:
        root_name = archive.namelist()[0].split("/")[0]
        extracted_root = destination_dir / root_name

        if extracted_root.exists():
            logger.info("Ya estaba extraído en %s", extracted_root)
            return extracted_root

        logger.info("Extrayendo %s en %s", archive_path.name, destination_dir)
        archive.extractall(destination_dir)
    return extracted_root
