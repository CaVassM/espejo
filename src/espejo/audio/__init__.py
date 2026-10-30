"""Módulo de audio: transcribe la clase, detecta preguntas y mide el tiempo de espera."""

from espejo.common.gpu import load_torch_cudnn_first

# Debe ocurrir antes de que cualquier submódulo importe faster-whisper (ctranslate2)
load_torch_cudnn_first()
