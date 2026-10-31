"""Importar antes que faster-whisper: en Windows, torch debe cargar su cuDNN primero.

Antes vivía en audio/__init__.py, pero así cualquier import de audio (por ejemplo, los esquemas
que usa INTERPRETAR) cargaba torch. Ahora solo lo importan los módulos que usan faster-whisper.
"""

from espejo.common.gpu import load_torch_cudnn_first

load_torch_cudnn_first()
