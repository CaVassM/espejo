# Contrato de salida de Espejo (visión + audio)

Este documento es la fuente de verdad: define las clases, los eventos y las métricas que
produce Espejo y cómo se calcula cada una. El código debe cumplirlo; si algo cambia, se cambia
primero aquí.

## Rúbricas

Se usa la versión vigente del manual de rúbricas de observación de aula del Minedu
(`docs/manual_rubricas_minedu.pdf`). Son 5:

| Código | Rúbrica |
|---|---|
| R1 | Involucra activamente a los estudiantes en el proceso de aprendizaje |
| R2 | Promueve el razonamiento, la creatividad y/o el pensamiento crítico |
| R3 | Evalúa el progreso de los aprendizajes para retroalimentar a los estudiantes y adecuar su enseñanza |
| R4 | Propicia un ambiente de respeto y proximidad |
| R5 | Regula positivamente el comportamiento de los estudiantes |

La versión de 2017 tenía 6 rúbricas ("Maximiza el tiempo dedicado al aprendizaje"); esa ya no se usa.
Ninguna métrica de esta fase apoya R5: depende del lenguaje y la cubrirá el LLM.

## Arquitectura: VER e INTERPRETAR

**VER** pasa el video y el audio **una sola vez** por los modelos y guarda solo datos:

- `detections.json`: por cada frame muestreado, el timestamp, las cajas "teacher" con su
  `track_id` y todas las cajas del modelo behavior con su clase y confianza.
- `background_blurred.png`: el primer frame, ya muy desenfocado (fondo del heatmap).
- `audio_raw.json`: segmentos de Whisper (con palabras, `avg_logprob` y `no_speech_prob`),
  intervalos de voz de Silero y turnos de pyannote (o el motivo por el que no hubo diarización).

**INTERPRETAR** calcula todo lo demás leyendo solo esos archivos, sin abrir el video ni el audio:
`python scripts/interpret.py --run outputs/<video>/`.

No se guardan frames ni recortes de personas: solo coordenadas, el heatmap con el fondo
desenfocado y, si se pide, el video de depuración (`--debug-video`, ignorado por git).

## Modelos

| Modelo | Entrenado por nosotros | Uso |
|---|---|---|
| YOLOv8s teacher | Sí | Encontrar al docente |
| YOLOv8s behavior | Sí | guide, board_writing, answer, stage_interaction |
| ByteTrack | Algoritmo, no modelo | Seguir al docente entre frames |
| Whisper large-v3-turbo | No (preentrenado) | Transcribir: qué se dijo |
| Silero VAD | No (preentrenado) | Cuándo hay voz |
| pyannote speaker-diarization-3.1 | No (preentrenado) | Quién habla |

El estimador de pose existe en el código pero está desactivado (`pose.enabled: false`).

## Clases de video

### Estado del docente (exactamente uno por frame)

Prioridad: **board_writing > guide > other**.

| Estado | Cuándo |
|---|---|
| `board_writing` | Una caja board_writing se superpone con la del docente (IoU ≥ `behavior_iou_threshold`). |
| `guide` | Una caja guide se superpone con la del docente y no hay board_writing. |
| `other` | No se detectó una acción del docente. **No es una clase aprendida**: es el relleno. |
| `not_visible` | No se encontró al docente en el frame. No es una conducta: alimenta `teacher_visible_ratio`. |

- Al docente **nunca** se le asignan `answer` ni `stage_interaction`, aunque se superpongan con él.
- Quién es el docente: el track de ByteTrack que aparece en más frames. Si en un frame no está,
  se acepta otra caja "teacher" solo si su confianza supera `teacher_fallback_conf`.
- La línea de tiempo se suaviza con voto mayoritario en una ventana de `smoothing_window_s` y los
  segmentos menores a `min_segment_s` se unen con su vecino.
- `board_writing` implica estar **de espaldas a la clase**: en 16 recortes al azar de
  board_writing del split val, el docente estaba de espaldas en los 16.

### Eventos de alumnos

| Clase | Uso |
|---|---|
| `stage_interaction` | Un alumno pasa al frente. Cada tramo continuo (tras suavizar) es **un** evento con su timestamp. Apoya R1. |
| `answer` | Se detecta y queda en `detections.json`, pero **no se usa en el reporte**: en el dataset es un alumno de pie (aulas chinas); en el Perú se responde sentado. Las respuestas se miden con el audio. |

## Clases y eventos de audio

- Whisper recibe el audio **crudo** (sin denoising), `language="es"`, `word_timestamps=True`.
- Silero VAD trabaja sobre una copia con filtro paso alto suave (`vad_highpass_hz`).
- pyannote se activa si `diarization.enabled` es true, existe `HF_TOKEN` en `.env` y se aceptaron
  las condiciones de los modelos en Hugging Face. Si algo falla, se sigue sin diarización y el
  motivo queda en el reporte.
- **Docente** = el hablante con más tiempo total de voz (heurística).

| Evento | Definición |
|---|---|
| `teacher_question` | Oración con `¿`, `?` o que empieza con un interrogativo (`questions.interrogatives`). Con diarización, solo las del docente (`speaker_known = true`). Sin diarización, todas (`speaker_known = false`). |
| `wait_time_s` | Desde el fin de la última palabra de la pregunta hasta el inicio del siguiente intervalo de voz de Silero. Se ignoran los huecos menores a `min_gap_s` y se limita a `max_wait_s`. Si la voz que contiene el fin de la pregunta continúa más de `min_gap_s` (el docente siguió hablando sin pausa), es 0. Si no vuelve a haber voz, `null`. |
| `wait_time_ok` | `wait_time_s >= ok_threshold_s` (3 s, umbral de Rowe). |
| `student_response` | Primera voz de un hablante distinto al docente dentro de `response_window_s` tras la pregunta (`certainty = "high"`). Sin diarización, cualquier voz nueva dentro de la ventana (`certainty = "low"`). |
| `low_confidence` | Segmento con `avg_logprob < min_avg_logprob` o `no_speech_prob > max_no_speech_prob`. Las preguntas en esos tramos llevan `reliable = false`. |

## Métricas finales

| Métrica | Rúbrica | Cálculo |
|---|---|---|
| `board_writing_ratio` | R1 | Tiempo en `board_writing` / duración analizada. "Tiempo escribiendo en la pizarra, de espaldas a la clase". |
| `stage_interaction_count` | R1 | Número de eventos stage_interaction. |
| `student_response_count` | R1 | Preguntas con `student_response`. |
| `response_rate` | R1 | `student_response_count / teacher_question_count` (`null` si no hay preguntas). |
| `teacher_talk_ratio` | R1 | Tiempo de habla del docente / duración. `null` sin diarización, con el motivo. |
| `teacher_question_count` | R2, R3 | Número de `teacher_question`. |
| `questions_per_minute` | R2, R3 | `teacher_question_count / (duración del audio en minutos)`. |
| `mean_wait_time_s`, `median_wait_time_s` | R2 | Sobre preguntas `reliable` con `wait_time_s` no nulo. |
| `wait_time_ok_ratio` | R2 | Fracción de esas preguntas con `wait_time_ok`. |
| `monitoring_ratio` | R3 | Unión, sin doble conteo, de los tramos `guide` (video) y los episodios de pregunta (audio), dividida entre la duración analizada. |
| `monitoring_video_ratio` | R3 | Solo los tramos `guide`. |
| `monitoring_audio_ratio` | R3 | Solo los episodios de pregunta: desde el inicio de una pregunta `reliable` hasta el fin de su respuesta; si nadie respondió, hasta el fin de la espera (`question_end + wait_time_s`, o `+ max_wait_s` si no hubo más voz). |
| `meets_monitoring_threshold` | R3 | `monitoring_ratio >= 0.25` (el manual pide monitorear al menos el 25 % de la sesión). Sin audio se calcula solo con video y se avisa. |
| `path_length_normalized` | R4 | Recorrido del punto de los pies en anchos de frame: mediana móvil de `median_window_frames` y descarte de pasos mayores a `max_step_ratio` × ancho (se cuentan). Es evidencia visual junto a `heatmap.png`, no un estado. |

### Calidad de los datos (`data_quality`)

`teacher_visible_ratio`, `discarded_path_jumps`, `low_confidence_audio_ratio`, `speech_ratio`,
`diarization_used`, `diarization_note` y una lista de avisos (`notes`).

## Archivos de salida (`outputs/<video>/`)

| Archivo | Etapa | Contenido |
|---|---|---|
| `detections.json` | VER | Cajas por frame muestreado. |
| `background_blurred.png` | VER | Primer frame muy desenfocado. |
| `audio_raw.json` | VER | Whisper, Silero y pyannote. |
| `class_report.json` | INTERPRETAR | `metadata`, `video_timeline`, `stage_interaction_events`, `audio_events` (preguntas con espera y respuesta), `metrics` (cada una con `value`, `rubric`, `description`, `note`) y `data_quality`. |
| `transcript.txt` | INTERPRETAR | `[mm:ss] DOCENTE: ...`, `ALUMNO` o `HABLANTE` (sin diarización); `[AUDIO DUDOSO]` en segmentos de baja confianza. |
| `heatmap.png` | INTERPRETAR | Recorrido del docente sobre el fondo desenfocado. |
| `debug.mp4` | Opcional | Caja y estado del docente dibujados (`--debug-video`). |
