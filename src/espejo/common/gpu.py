"""Convivencia de torch y ctranslate2 (faster-whisper) en la misma GPU, en Windows."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

WINDOWS_PLATFORM = "win32"
CUBLAS_DLL_NAME = "cublas64_12.dll"
# Tamaño mínimo de la convolución de calentamiento: solo sirve para cargar cuDNN
WARMUP_IMAGE_SIDE = 8
WARMUP_KERNEL_SIDE = 3
BYTES_PER_GB = 1024**3
GPU_INDEX = 0


@dataclass(frozen=True)
class GpuMemory:
    free_gb: float
    total_gb: float


def read_gpu_memory() -> GpuMemory | None:
    """Memoria según NVML; no crea un contexto CUDA que compita con otros procesos."""
    # Import local: NVML solo existe en equipos con driver NVIDIA
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None
    return GpuMemory(memory_info.free / BYTES_PER_GB, memory_info.total / BYTES_PER_GB)


def torch_lib_dir() -> Path | None:
    # Import local: torch es pesado y solo se necesita para ubicar sus DLL
    try:
        import torch
    except ImportError:
        return None
    return Path(torch.__file__).parent / "lib"


def register_cuda_libraries() -> bool:
    """En Windows, ctranslate2 no encuentra solo las DLL de CUDA 12; torch (cu12) ya las trae.

    Agregamos la carpeta lib de torch a la búsqueda de DLL. En Linux no hace falta.
    """
    if sys.platform != WINDOWS_PLATFORM:
        return True

    lib_dir = torch_lib_dir()
    if lib_dir is None or not (lib_dir / CUBLAS_DLL_NAME).exists():
        return False

    os.add_dll_directory(str(lib_dir))
    os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
    return True


def load_torch_cudnn_first() -> None:
    """Obliga a torch a cargar su cuDNN antes que ctranslate2.

    En Windows, torch y ctranslate2 traen cada uno su propia cuDNN y gana la que se carga
    primero. Si ctranslate2 se adelanta, YOLO y pyannote fallan con "Could not load symbol
    cudnnGetLibConfig" y el proceso termina sin aviso. Una convolución mínima lo evita.
    """
    if sys.platform != WINDOWS_PLATFORM:
        return

    import torch

    if not torch.cuda.is_available():
        return
    image = torch.zeros(1, 1, WARMUP_IMAGE_SIDE, WARMUP_IMAGE_SIDE, device="cuda")
    kernel = torch.zeros(1, 1, WARMUP_KERNEL_SIDE, WARMUP_KERNEL_SIDE, device="cuda")
    torch.nn.functional.conv2d(image, kernel)
