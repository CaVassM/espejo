# Contrato del worker (VER en Modal)

El worker recibe el video de una clase, corre **VER** una sola vez (YOLO, Whisper, Silero,
pyannote y spaCy) y devuelve tres cosas: `detections.json`, `audio_raw.json` (ya anonimizado) y
el heatmap. **No** calcula métricas ni llama al LLM: eso es **INTERPRETAR**, la parte liviana de
`engine/`, y la corre el backend.

- Código: [`services/worker/`](../services/worker/) · Configuración: [`configs/worker.yaml`](../configs/worker.yaml)
- URL base: `https://cavassm--espejo-worker-web.us-east.modal.run` (en este documento, `$WORKER`)

## Autenticación

| Quién | Cómo | Para |
|---|---|---|
| Quien sube el video (web o backend) | Cabecera `X-Access-Code: <UPLOAD_ACCESS_CODE>` | `POST /jobs` |
| El backend | Cabecera `Authorization: Bearer <WORKER_TOKEN>` | `GET /jobs/{id}/result` |
| El worker, al avisar al backend | Cabecera `Authorization: Bearer <WORKER_TOKEN>` | Callback (ver abajo) |

`GET /health` y `GET /jobs/{id}` son abiertos. El `job_id` es aleatorio (32 caracteres) y el
estado no trae datos de la clase. Los dos secretos viven en el Modal Secret `espejo-secrets`;
el backend necesita el mismo `WORKER_TOKEN`.

## Endpoints

### `GET /health`

```bash
curl $WORKER/health
```
```json
{"status": "ok"}
```

### `POST /jobs`: subir un video

Formulario `multipart/form-data`:

| Campo | Obligatorio | Qué es |
|---|---|---|
| `video` | Sí | El archivo. Formatos: `.mp4 .mov .m4v .mkv .webm .avi`. Máximo 1024 MB. |
| `topic` | No | Tema de la clase (ayuda a Whisper con nombres del tema), p. ej. `Cuento: Los hijos del labrador`. |
| `analysis_id` | No | El id del análisis en el backend. Si viene, el callback usa este id; si no, el `job_id`. |

```bash
curl -X POST $WORKER/jobs \
  -H "X-Access-Code: $UPLOAD_ACCESS_CODE" \
  -F "video=@clase.mp4" -F "topic=Cuento: Los hijos del labrador" -F "analysis_id=123"
```

Responde **enseguida** (`202 Accepted`) y procesa en segundo plano:

```json
{
  "job_id": "3f9a1c0e8b6d4e2f9a7c5b3d1e0f2a4b",
  "analysis_id": "123",
  "status": "queued",
  "stage": "En cola",
  "created_at": "2026-10-31T19:40:12+00:00",
  "updated_at": "2026-10-31T19:40:12+00:00",
  "error": null,
  "timings": {},
  "callback": null
}
```

### `GET /jobs/{job_id}`: estado

Misma forma que la respuesta de `POST /jobs`. Consultar cada 10 s, más o menos.

| `status` | `stage` (para mostrar) | Qué pasa |
|---|---|---|
| `queued` | En cola | Esperando una GPU (o arrancando el contenedor) |
| `processing` | Mirando el video | YOLO: docente, conducta y recorrido |
| `processing` | Escuchando el audio | Whisper, Silero, pyannote y la anonimización |
| `done` | Listo | El resultado está disponible |
| `error` | Error | `error` dice qué pasó, p. ej. `"No se pudo procesar el video (ValueError)."` |

Al terminar, `timings` trae los segundos de cada etapa: `queued_s` (incluye el arranque en frío),
`video_s`, `audio_s` y `total_s`; además `cold_start` (si fue el primer video del contenedor) y
`models_load_s`. `callback` es `null`, `"sent"` o `"failed"`.

### `GET /jobs/{job_id}/result`: resultado

```bash
curl $WORKER/jobs/$JOB_ID/result -H "Authorization: Bearer $WORKER_TOKEN"
```
```json
{
  "job_id": "3f9a1c0e8b6d4e2f9a7c5b3d1e0f2a4b",
  "analysis_id": "123",
  "detections": { "...": "detections.json, ver formato abajo" },
  "audio_raw": { "...": "audio_raw.json, ver formato abajo (null si el video no tiene audio)" },
  "heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Errores

| Código | Cuándo |
|---|---|
| `401` | Falta o es incorrecto `X-Access-Code` (subida) o el token (resultado). |
| `404` | No existe ese `job_id` (o ya venció: los resultados duran 7 días). |
| `409` | Se pidió el resultado antes de `done`; el cuerpo trae el estado actual en `detail`. |
| `413` | El video pasa de 1024 MB. |
| `415` | Formato de video no admitido. |
| `422` | Falta el campo `video`. |

## Callback opcional

Si el Secret tiene `API_BASE_URL`, al terminar el worker hace:

```
POST {API_BASE_URL}/api/analyses/{analysis_id o job_id}/raw
Authorization: Bearer <WORKER_TOKEN>
Content-Type: application/json

(el mismo cuerpo que GET /jobs/{id}/result)
```

Tres intentos (a los 5 y 10 s). El backend debe responder `2xx`. Si no responde, el resultado
sigue disponible en `GET /jobs/{id}/result`. Sin `API_BASE_URL` no hay callback.

## Privacidad (lo que nunca sale del worker)

- El video y el audio se **borran** al terminar o si algo falla.
- En `audio_raw.json`, todo lo que no dice el docente es `"[ALUMNO responde]"`, con su minuto y
  su duración y sin palabras. El texto de los alumnos no se guarda en ningún lado.
- Los nombres que dice el docente son `"[NOMBRE]"` (spaCy `es_core_news_md` y la lista
  `configs/names_pe.yaml`).
- No hay frames ni recortes de personas: solo coordenadas. El heatmap usa el primer frame muy
  desenfocado como fondo, y ese frame no se entrega.

## Formato de `detections.json`

| Campo | Tipo | Qué es |
|---|---|---|
| `video.path` | texto | Ruta del video dentro del worker (solo informativa) |
| `video.duration_s` | número | Duración en segundos |
| `video.native_fps` | número | Cuadros por segundo del video original |
| `video.width`, `video.height` | entero | Tamaño en píxeles |
| `sample_fps` | número | Cuadros por segundo que se analizaron (2) |
| `models` | objeto | Nombres de los modelos: `teacher`, `teacher_class`, `behavior`, `tracker` |
| `frames[]` | lista | Un elemento por cuadro analizado |
| `frames[].frame_index` | entero | Número de cuadro en el video original |
| `frames[].timestamp_s` | número | Segundo del video |
| `frames[].teacher[]` | lista | Cajas del modelo del docente: `class_name`, `confidence` (0 a 1), `box` (`[x1, y1, x2, y2]` en píxeles), `track_id` (entero o `null`) |
| `frames[].behaviors[]` | lista | Cajas del modelo de conducta, mismo formato. Clases: `guide`, `board_writing`, `answer`, `stage_interaction` |
| `frames[].poses[]` | lista | Vacía (la pose está desactivada) |

## Formato de `audio_raw.json`

| Campo | Tipo | Qué es |
|---|---|---|
| `video_path` | texto | Ruta del video dentro del worker (solo informativa) |
| `duration_s` | número | Duración del audio en segundos |
| `language` | texto | `"es"` |
| `topic` | texto o `null` | El tema enviado en `POST /jobs` |
| `transcriber` | objeto | `model_name`, `device`, `compute_type` de Whisper |
| `segments[]` | lista | Lo que se dijo, en orden |
| `segments[].start_s`, `end_s` | número | Inicio y fin en segundos |
| `segments[].text` | texto | Lo que dijo el docente (con `[NOMBRE]`) o `"[ALUMNO responde]"` |
| `segments[].avg_logprob`, `no_speech_prob` | número | Confianza de Whisper (sirven para marcar audio dudoso) |
| `segments[].words[]` | lista | Palabras con `text`, `start_s`, `end_s`, `probability`. Vacía en los segmentos de alumnos |
| `speech_intervals[]` | lista | Tramos con voz según Silero: `start_s`, `end_s` |
| `diarization.used` | booleano | Si pyannote funcionó |
| `diarization.note` | texto | Modelo usado o el motivo por el que no hubo diarización |
| `diarization.turns[]` | lista o `null` | Quién habla y cuándo: `speaker` (`SPEAKER_00`...), `start_s`, `end_s` |

Quién es el docente **no** viene en el archivo: lo decide INTERPRETAR (el hablante con más tiempo
de voz), igual que lo hizo el worker para anonimizar.

## Qué hace el backend

1. Crea el análisis y sube el video al worker (`POST /jobs` con su `analysis_id`), o deja que la
   web lo suba directo.
2. Recibe el callback, o consulta `GET /jobs/{id}` hasta `done` y pide `GET /jobs/{id}/result`.
3. Guarda `detections.json`, `audio_raw.json` y `heatmap.png` en una carpeta del análisis.
4. Corre INTERPRETAR con `engine/` **sin extras** (sin torch): métricas, ficha por rúbrica, LLM
   y consejo. El resultado es `class_report.json` (formato en
   [`output_contract.md`](output_contract.md)).

### Ejemplo mínimo de INTERPRETAR en el backend

```python
import base64
import json
from pathlib import Path

from espejo.analysis.runner import interpret_run, load_all_configs


def interpret_worker_result(result: dict, run_dir: Path) -> dict:
    """result: el cuerpo de GET /jobs/{id}/result (o del callback)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detections.json").write_text(json.dumps(result["detections"]), encoding="utf-8")
    if result["audio_raw"] is not None:
        audio = json.dumps(result["audio_raw"], ensure_ascii=False)
        (run_dir / "audio_raw.json").write_text(audio, encoding="utf-8")
    if result["heatmap_png_base64"] is not None:
        (run_dir / "heatmap.png").write_bytes(base64.b64decode(result["heatmap_png_base64"]))

    interpretation = interpret_run(run_dir, load_all_configs())
    # interpret_run también guarda run_dir / "class_report.json" y transcript.txt
    return json.loads((run_dir / "class_report.json").read_text(encoding="utf-8"))
```

Instalación en el backend: `pip install -e ../../engine` desde `services/api` (ya está en su
`requirements.txt`). Tiene que ser editable: el motor lee `configs/` y
`docs/rubric_definitions.md` de la raíz del repo. El LLM se elige en
`configs/llm.yaml`; sin LLM disponible, el reporte sale igual con las reglas de respaldo y dice
"sin análisis LLM". Sin OpenCV, `interpret_run` no vuelve a dibujar el heatmap: se usa el que
envió el worker.
