# Espejo

Espejo le devuelve al docente una mirada objetiva de su propia clase. El docente graba la sesión
con su celular y recibe un reporte con evidencias (cuánto tiempo monitoreó a sus alumnos, cuánto
escribió en la pizarra, por dónde se movió en el aula) alineadas a las rúbricas de observación de
aula del Minedu (Perú).

Este repositorio contiene el **módulo de visión** y el **módulo de audio**. El módulo de reporte con
LLM (`src/espejo/report/`) se agregará en la siguiente fase.

## Pipeline de visión

```mermaid
flowchart LR
    A[Video de la clase] --> B[FrameSampler<br/>2 fps]
    B --> C[TeacherDetector + ByteTrack<br/>modelo teacher]
    B --> D[BehaviorDetector<br/>modelo behavior]
    B --> E[PoseEstimator<br/>yolov8n-pose]
    C --> F[TeacherTracker<br/>track dominante = docente]
    F --> G[Conducta del docente<br/>IoU con guide / board_writing]
    D --> G
    F --> H[Orientación<br/>facing / back / unknown]
    E --> H
    G --> I[TimelineBuilder<br/>voto mayoritario + unión de segmentos]
    H --> I
    I --> J[MetricsCalculator]
    F --> K[HeatmapRenderer<br/>fondo desenfocado]
    J --> L[vision_report.json]
    K --> M[heatmap.png]
```

Cada etapa es una clase en su propio módulo dentro de `src/espejo/vision/`. Todos los umbrales
viven en [`configs/vision.yaml`](configs/vision.yaml).

| Etapa | Qué hace |
|---|---|
| `FrameSampler` | Lee el video con OpenCV y entrega frames a `sample_fps`. |
| `TeacherTracker` | Sigue a las cajas "teacher" con ByteTrack y elige el track que aparece en más frames. Si en un frame falta, acepta otra caja "teacher" solo con confianza alta. |
| `BehaviorDetector` | Detecta conductas en el frame completo. Asigna `guide` o `board_writing` al docente si el IoU con su caja es ≥ 0.3. `answer` y `stage_interaction` se cuentan como señal de participación. |
| `orientation.py` | Con los keypoints del docente: cara visible → `facing`; solo hombros → `back`; si no, `unknown`. |
| `TimelineBuilder` | Suaviza con voto mayoritario (ventana de 3 s) y une segmentos menores a 1.5 s. |
| `MetricsCalculator` | Ratios de monitoreo, pizarra y espalda; recorrido; cobertura del aula; participación. |
| `HeatmapRenderer` | Mapa de calor del recorrido sobre el primer frame con un desenfoque gaussiano fuerte. |

### Métricas

- `analyzed_duration_s` y `teacher_visible_ratio`.
- `monitoring_ratio`: tiempo en `guide` / duración analizada.
- `meets_monitoring_threshold`: `monitoring_ratio >= 0.25`. La rúbrica "Evalúa el progreso de los
  aprendizajes" pide monitorear activamente al menos el 25 % de la sesión.
- `board_writing_ratio` y `back_to_class_ratio`.
- `path_length_normalized`: recorrido del docente medido en anchos de frame.
- `room_coverage_ratio`: fracción de celdas de una grilla de 8x6 que el docente visitó.
- `student_participation_events`: detecciones de `answer` y `stage_interaction` (sin identificar alumnos).

### Privacidad

No se guardan frames ni recortes de personas. La única imagen que se genera es el heatmap, con el
fondo desenfocado al punto de que no se reconoce a nadie. El video de depuración es opcional y se
guarda en `outputs/`, que no se sube al repositorio. No hay reconocimiento facial ni de emociones.

## Instalación

Requiere Python 3.11 o superior y, de preferencia, una GPU NVIDIA con CUDA.

```bash
conda activate ml_env            # o el entorno que uses
pip install -r requirements.txt
pip install -e .                 # instala el paquete espejo en modo editable
```

## Dataset

```bash
python scripts/inspect_dataset.py --download   # lista el repo, descarga y resume el subconjunto
python scripts/prepare_dataset.py              # genera data/processed/teacher y data/processed/behavior
python scripts/preview_dataset.py              # dibuja cajas en outputs/dataset_preview/
```

En el repositorio publicado de SCB-Dataset, los subconjuntos SCB5-F (docente) y SCB5-D (conductas)
vienen fusionados en `SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406`: 8 clases, formato
YOLO, con splits train/val del autor. A partir de él derivamos dos datasets independientes:

| Dataset | Clases | Imágenes que se conservan |
|---|---|---|
| `teacher` | 0 = teacher | solo las que tienen al menos una caja `teacher` |
| `behavior` | 0 = guide, 1 = board_writing, 2 = answer, 3 = stage_interaction | solo las que tienen al menos una caja de conducta |

Filtrar así evita que una imagen sin anotar le enseñe al modelo que el docente o la conducta son
fondo. Las clases `stand`, `screen` y `blackBoard` se descartan.

## Entrenamiento

```bash
# Prueba de humo: 1 época con el 5 % de los datos
python scripts/train.py --dataset teacher --epochs 1 --fraction 0.05 --name smoke_teacher --no-export

# Entrenamiento completo (yolov8s preentrenado, 30 épocas, batch automático)
python scripts/train.py --dataset teacher
python scripts/train.py --dataset behavior
```

Cada corrida se guarda en `runs/<dataset>/`, deja el mAP@0.5 por clase en
`runs/<dataset>/per_class_map50.json` y copia el mejor modelo a `weights/<dataset>.pt`.
Si la GPU se queda sin memoria, usa `--batch 8`.

## Analizar un video

```bash
python scripts/analyze_video.py --video samples/clase.mp4
python scripts/analyze_video.py --video samples/clase.mp4 --output outputs/ --debug-video
```

Genera `outputs/<nombre_video>/vision_report.json` y `outputs/<nombre_video>/heatmap.png`, e
imprime un resumen en consola. Para probar el flujo sin pesos entrenados:

```bash
python scripts/analyze_video.py --video samples/clase.mp4 \
    --teacher-weights yolov8s.pt --teacher-class person --behavior-weights yolov8s.pt
```

### Ejemplo de salida

Estructura de `vision_report.json` (valores ilustrativos):

```json
{
  "metadata": {
    "video": "clase.mp4",
    "duration_s": 149.8,
    "duration": "02:30",
    "native_fps": 30.0,
    "resolution": {"width": 1920, "height": 1080},
    "sample_fps": 2.0,
    "models": {
      "teacher": "teacher.pt",
      "teacher_class": "teacher",
      "behavior": "behavior.pt",
      "pose": "yolov8n-pose.pt",
      "tracker": "bytetrack.yaml"
    }
  },
  "behavior_timeline": [
    {"label": "board_writing", "start_s": 0.0, "end_s": 41.5, "start": "00:00", "end": "00:42", "duration_s": 41.5},
    {"label": "guide", "start_s": 41.5, "end_s": 88.0, "start": "00:42", "end": "01:28", "duration_s": 46.5}
  ],
  "orientation_timeline": [
    {"label": "back", "start_s": 0.0, "end_s": 39.0, "start": "00:00", "end": "00:39", "duration_s": 39.0},
    {"label": "facing", "start_s": 39.0, "end_s": 149.8, "start": "00:39", "end": "02:30", "duration_s": 110.8}
  ],
  "metrics": {
    "analyzed_duration_s": 149.8,
    "teacher_visible_ratio": 0.953,
    "monitoring_ratio": 0.31,
    "meets_monitoring_threshold": true,
    "board_writing_ratio": 0.277,
    "back_to_class_ratio": 0.26,
    "path_length_normalized": 3.42,
    "room_coverage_ratio": 0.271,
    "student_participation_events": 18,
    "analyzed_duration": "02:30"
  }
}
```

## Módulo de audio

El audio cubre lo que la visión no ve: lo que dice el docente. Funciona con **un solo celular**,
sin micrófonos adicionales.

```mermaid
flowchart LR
    A[Video de la clase] --> B[ffmpeg<br/>wav 16 kHz mono]
    B -->|audio crudo| C[faster-whisper<br/>timestamps por palabra]
    B -->|copia con paso alto 100 Hz| D[Silero VAD<br/>intervalos de voz]
    B -.->|opcional| E[pyannote<br/>diarización]
    C --> F[Calidad<br/>segmentos dudosos]
    F --> G[Rol del docente]
    E -.-> G
    G --> H[Preguntas]
    H --> I[Wait time]
    D --> I
    I --> J[AudioMetrics]
    J --> K[audio_report.json<br/>transcript.txt]
```

| Etapa | Qué hace |
|---|---|
| `extraction.py` | Con ffmpeg genera `audio_raw.wav` (16 kHz, mono, sin filtros) y `audio_vad.wav` (lo mismo con un paso alto suave). Son temporales: se borran al terminar salvo con `--keep-audio`. |
| `transcriber.py` | faster-whisper en español con timestamps por palabra. Con `device: auto` usa `large-v3-turbo` en GPU si está libre y `small` en CPU si está ocupada o no hay GPU. |
| `voice_activity.py` | Silero VAD (la versión ONNX incluida en faster-whisper) sobre la copia filtrada: tramos con voz, sin importar quién habla. |
| `diarization.py` | Opcional. Separa hablantes con pyannote y asigna uno a cada segmento por solapamiento temporal. |
| `speaker_roles.py` | El hablante con más tiempo total de voz es el docente; los demás son alumnos. |
| `questions.py` | Una oración es pregunta si tiene `¿` o `?`, o si empieza con un interrogativo (`qué`, `cómo`, `por qué`, ...). |
| `wait_time.py` | Segundos desde el fin de la pregunta hasta que vuelve a haber voz. |
| `quality.py` | Marca segmentos de baja confianza según `avg_logprob` y `no_speech_prob` de Whisper. |

### Decisiones técnicas

- **Whisper recibe el audio crudo.** El denoising previo empeora la transcripción de Whisper
  (arXiv 2603.04710; en un estudio con audio médico, el audio ruidoso ganó en 40 de 40
  configuraciones). Solo la copia que usa el VAD lleva un filtro paso alto de 100 Hz.
- **El wait time no necesita saber quién habla**, solo cuándo hay voz. Para cada pregunta se toma el
  siguiente intervalo de voz del VAD que empieza después de la pregunta. Los huecos menores a
  `min_gap_s` se ignoran (respiración, cola de la propia pregunta) y el resultado se limita a
  `max_wait_s`. Si la voz que contiene el fin de la pregunta continúa más de `min_gap_s`, el docente
  siguió hablando sin pausa y el wait time es 0. `wait_time_ok` exige al menos 3 s (Rowe, 1974).
- **Rol del docente (heurística).** El hablante con más tiempo de voz es el docente. Suele acertar en
  clases expositivas, pero puede fallar en debates o exposiciones de alumnos.
- **Calidad.** Las preguntas que caen en segmentos de baja confianza se marcan `reliable: false` y
  no entran en las métricas de wait time.

### Métricas

- `analyzed_duration_s` y `speech_ratio` (tiempo con voz / duración).
- `teacher_talk_ratio`: requiere diarización; si no la hay, es `null` y se informa el motivo.
- `question_count` y `questions_per_minute`.
- `mean_wait_time_s`, `median_wait_time_s` y `wait_time_ok_ratio`, solo con preguntas confiables.
- `low_confidence_ratio`: proporción del tiempo con audio dudoso.

### Uso

```bash
python scripts/analyze_audio.py --video samples/clase.mp4
python scripts/analyze_audio.py --video samples/clase.mp4 --output outputs/ --keep-audio --model small
```

Requiere ffmpeg en el PATH (o en el entorno: `conda install -n ml_env -c conda-forge ffmpeg`).
Genera en `outputs/<nombre_video>/`:

- `audio_report.json`: metadata (modelo de Whisper, dispositivo, diarización), segmentos con sus
  palabras, preguntas con su wait time y métricas. Tiempos en segundos y en `mm:ss`.
- `transcript.txt`: una línea por segmento, pensado para el LLM de la siguiente fase:

```
[00:05] HABLANTE: ¿Qué creen que pasará con el agua si la calentamos?
[00:12] HABLANTE: Se evapora.
[00:15] HABLANTE: Muy bien. ¿Cómo se llama ese proceso?
```

Con diarización las etiquetas son `DOCENTE` y `ESTUDIANTE`. Los segmentos de baja confianza llevan
la marca `[AUDIO DUDOSO]`.

### Activar la diarización (opcional)

1. Acepta las condiciones de uso de
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) y
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) en Hugging Face.
2. Instala `pyannote.audio` (versión 3.x) y exporta tu token: `export HF_TOKEN=hf_...`
   (en PowerShell: `$env:HF_TOKEN = "hf_..."`).
3. Pon `diarization.enabled: true` en [`configs/audio.yaml`](configs/audio.yaml).

Si la diarización está desactivada o falla, el pipeline sigue igual: los segmentos quedan con
`speaker: "unknown"`, se cuentan todas las preguntas y `teacher_talk_ratio` sale `null`.

## Tests y calidad

```bash
pytest
ruff format .
ruff check .
```

Los tests cubren la lógica pura (geometría, orientación, timeline, métricas, selección del
docente, preguntas, wait time, roles de hablante y calidad del audio) y no necesitan modelos, GPU
ni audio real.

## Estructura

```
configs/            umbrales del pipeline y YAML de los datasets
scripts/            puntos de entrada delgados (CLI)
src/espejo/
  config.py         configuración en dataclasses tipadas
  common/           utilidades compartidas (rutas, YAML, mm:ss, JSON)
  training.py       entrenamiento y evaluación de los detectores
  dataset/          descarga, inspección, preparación y preview de SCB-Dataset
  vision/           etapas del pipeline de visión
  audio/            etapas del pipeline de audio
tests/              tests de la lógica pura
```

## Dataset y licencia

Los modelos se entrenan con **SCB-Dataset**:

> Fan Yang. *SCB-Dataset: A Dataset for Detecting Student and Teacher Classroom Behavior*.
> arXiv:2304.02488, 2023. https://arxiv.org/abs/2304.02488

SCB-Dataset es **solo para uso académico** y no se redistribuye. Por eso este repositorio no
incluye imágenes, etiquetas ni pesos entrenados: cada persona debe descargarlo desde
[Hugging Face](https://huggingface.co/datasets/wintonYF/SCB-Dataset) con `scripts/inspect_dataset.py`
y aceptar sus condiciones de uso.
