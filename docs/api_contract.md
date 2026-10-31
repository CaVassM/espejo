# Contrato de la API (para el frontend)

La API vive en Render (`services/api`). El frontend (Vercel) solo habla con ella, salvo para
subir el video, que va **directo al worker de Modal** con un permiso que entrega la API.

- URL base: la de Render, p. ej. `https://espejo-api.onrender.com` (en este documento, `$API`).
- Formato: JSON. CORS abierto para `*.vercel.app` y `localhost`. Otro dominio se agrega en la
  variable `FRONTEND_ORIGINS` de Render y del Secret de Modal.
- Plan gratis de Render: tras 15 min sin peticiones se duerme y la primera respuesta tarda
  ~30 a 60 s. Conviene llamar a `GET $API/api/health` al abrir la página para despertarla.

## Flujo

```
1. POST $API/api/analyses            -> analysis_id + permiso de subida
2. POST {upload.url} (Modal)         -> sube el video con el permiso
3. GET  $API/api/analyses/{id}       -> cada ~5 s hasta status "done" o "error"
4. GET  $API/api/analyses/{id}/report     -> el reporte (class_report.json)
   GET  $API/api/analyses/{id}/heatmap.png -> la imagen del recorrido
```

## 1. Crear el análisis

```http
POST $API/api/analyses
Content-Type: application/json

{"access_code": "<código que escribe el profesor>", "topic": "Cuento: Los hijos del labrador"}
```

`topic` es opcional. El código de acceso **no** va escrito en el frontend: lo escribe el
profesor. Respuesta `201`:

```json
{
  "analysis_id": "9b2f0c4e1d6a4f7e8c3b2a1d0e9f8a7b",
  "status": "waiting_upload",
  "stage": "Esperando el video",
  "error": null,
  "topic": "Cuento: Los hijos del labrador",
  "created_at": "2026-10-31T20:10:00Z",
  "updated_at": "2026-10-31T20:10:00Z",
  "report_ready": false,
  "upload": {
    "url": "https://cavassm--espejo-worker-web.us-east.modal.run/jobs",
    "ticket": "9b2f0c4e1d6a4f7e8c3b2a1d0e9f8a7b.1793560200.5c1e...",
    "expires_at": "2026-10-31T21:10:00Z",
    "fields": {"analysis_id": "9b2f0c4e1d6a4f7e8c3b2a1d0e9f8a7b", "topic": "Cuento: Los hijos del labrador"}
  }
}
```

`401` si el código es incorrecto.

## 2. Subir el video (directo a Modal)

Formulario `multipart/form-data` a `upload.url`, con la cabecera `X-Upload-Ticket` y los campos
de `upload.fields` más el archivo en `video`:

```js
const form = new FormData();
for (const [key, value] of Object.entries(upload.fields)) form.append(key, value);
form.append("video", file);  // .mp4 .mov .m4v .mkv .webm .avi, hasta 1024 MB

const response = await fetch(upload.url, {
  method: "POST",
  headers: { "X-Upload-Ticket": upload.ticket },
  body: form,
});
// 202: listo, Modal ya lo está procesando
```

El permiso vale 1 hora y **una sola vez**. Errores: `401` (permiso inválido o vencido), `409`
(ese análisis ya tiene video), `413` (más de 1024 MB), `415` (formato no admitido). Para mostrar
el avance de la subida, usar `XMLHttpRequest` (`upload.onprogress`), que `fetch` no lo da.

## 3. Consultar el estado

```http
GET $API/api/analyses/{analysis_id}
```

Misma forma que la respuesta de crear, sin `upload`. Consultar cada ~5 s mientras el profesor
espera (eso también mantiene despierta a Render).

| `status` | `stage` (texto para mostrar) |
|---|---|
| `waiting_upload` | Esperando el video |
| `processing` | En cola · Mirando el video · Escuchando el audio (vienen de Modal) |
| `interpreting` | Preparando el reporte |
| `done` | Listo (`report_ready: true`) |
| `error` | Error (`error` trae el motivo en español) |

Tiempos de referencia, para un clip de 5 min: ~1 min en Modal y ~2 a 3 min preparando el
reporte (el LLM). Una clase de 45 min tarda más, sobre todo en "Preparando el reporte".

## 4. El reporte

```http
GET $API/api/analyses/{analysis_id}/report        -> 200 JSON (409 si aún no está listo)
GET $API/api/analyses/{analysis_id}/heatmap.png   -> 200 image/png (404 si no hay)
```

El reporte es `class_report.json`, con el formato de [`output_contract.md`](output_contract.md).
Lo más útil para mostrar:

| Campo | Qué es |
|---|---|
| `one_suggestion` | El consejo: `what_we_saw`, `why_it_matters`, `try_this`, `example_phrase` (y `text` con las tres partes) |
| `rubric_evidence[]` | Ficha por rúbrica R1 a R5: `rubric`, `title`, `status` ("con evidencia" o "evidencia insuficiente"), `metrics[]` y `citations[]` (minuto, cita y motivo) |
| `metrics` | Cada métrica con `value`, `rubric` y `description` (p. ej. `mean_wait_time_s`, `open_question_ratio`, `monitoring_ratio`) |
| `audio_events[]` | Las preguntas del docente, con minuto, espera y si hubo respuesta |
| `llm_analysis` | Lo que marcó el LLM: `feedback`, `participation`, `question_labels`, etc. |

Privacidad: lo que dicen los alumnos aparece como "[ALUMNO responde]" y los nombres como
"[NOMBRE]". No hay video, audio ni imágenes de personas: solo el heatmap con el fondo desenfocado.

## Errores comunes

| Código | Cuándo |
|---|---|
| `401` | Código de acceso incorrecto (crear) o permiso inválido (subir) |
| `404` | No existe ese `analysis_id` |
| `409` | Se pidió el reporte antes de `done`, o se reusó el permiso de subida |
| `422` | Falta un campo en el JSON |
