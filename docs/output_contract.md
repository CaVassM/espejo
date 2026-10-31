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
| `guide` | Una caja guide con confianza ≥ `guide_min_conf` se superpone con la del docente y no hay board_writing. Los tramos de guide más cortos que `min_guide_s` pasan a `other`. |
| `other` | No se detectó una acción del docente. **No es una clase aprendida**: es el relleno. |
| `not_visible` | No se encontró al docente en el frame. No es una conducta: alimenta `teacher_visible_ratio`. |

- Al docente **nunca** se le asignan `answer` ni `stage_interaction`, aunque se superpongan con él.
- Quién es el docente: en cada frame, la caja "teacher" de mayor confianza, si supera
  `teacher_min_conf`. Si dos cajas están a menos de `similar_conf_margin` en confianza, gana la
  más cercana a la última posición conocida. (Antes se usaba el track que más aparecía; en una
  clase real eso eligió a un alumno quieto.)
- Si el docente se pierde menos de `gap_fill_s` entre dos apariciones, se mantienen su último
  estado y su posición, marcados como interpolados (`data_quality.interpolated_ratio`).
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
- pyannote se llama con `min_speakers` (2): en un aula hay al menos docente y alumnos.
- `--topic "texto"` se pasa a Whisper como `initial_prompt` (ayuda con nombres propios del tema).
- **Texto inventado**: se descartan los segmentos de Whisper de `max_words_for_vad_check`
  palabras o menos que coinciden con voz de Silero en menos de `min_vad_overlap` de su duración, y los que son solo una frase típica inventada
  (`hallucinations.phrases`, p. ej. "Gracias."). Se cuentan en
  `data_quality.discarded_hallucinated_segments`.
- **Docente** = el hablante con más tiempo total de voz (heurística).

| Evento | Definición |
|---|---|
| `teacher_question` | Oración con `¿`, `?` o que empieza con un interrogativo (`questions.interrogatives`). Las oraciones se cortan después de cada `?` (aunque le siga una coma) y antes de cada `¿`. Con diarización, solo las del docente (`speaker_known = true`). Sin diarización, todas (`speaker_known = false`). |
| `tag_question` | Muletilla (regla de respaldo): la pregunta es solo una de `questions.tag_words` ("¿no?", "¿ya?"...) o termina en una de ellas después de una afirmación ("muy bien, ¿no?"). Si hay LLM, manda su clasificación (`question_rhetorical`). **No cuenta como pregunta, ni para la espera, ni para el monitoreo.** |
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
| `teacher_question_count` | R2, R3 | Número de `teacher_question` que no son muletillas ni retóricas. |
| `rhetorical_question_count` | R2 | Muletillas o preguntas retóricas (no cuentan como preguntas). |
| `open_question_ratio` | R2 | Abiertas / (abiertas + cerradas), según el LLM. `null` sin LLM. |
| `questions_per_minute` | R2, R3 | `teacher_question_count / (duración del audio en minutos)`. |
| `mean_wait_time_s`, `median_wait_time_s` | R2 | Sobre preguntas `reliable` con `wait_time_s` no nulo. |
| `wait_time_ok_ratio` | R2 | Fracción de esas preguntas con `wait_time_ok`. |
| `monitoring_ratio` | R3 | Unión, sin doble conteo, de los tramos `guide` (video) y los episodios de pregunta (audio), dividida entre la duración analizada. |
| `monitoring_video_ratio` | R3 | Solo los tramos `guide`. |
| `monitoring_audio_ratio` | R3 | Solo los episodios de pregunta: preguntas reales (no muletillas), en audio confiable y **con** `student_response`. Cada episodio va desde el inicio de la pregunta hasta el fin de la respuesta, con un máximo de `max_episode_s`. |
| `meets_monitoring_threshold` | R3 | `monitoring_ratio >= 0.25` (el manual pide monitorear al menos el 25 % de la sesión). Sin audio se calcula solo con video y se avisa. |
| `path_length_normalized` | R4 | Recorrido del punto de los pies en anchos de frame: mediana móvil de `median_window_frames` y descarte de pasos mayores a `max_step_ratio` × ancho (se cuentan). Es evidencia visual junto a `heatmap.png`, no un estado. |

### Calidad de los datos (`data_quality`)

`teacher_visible_ratio`, `interpolated_ratio`, `discarded_path_jumps`,
`low_confidence_audio_ratio`, `discarded_hallucinated_segments`, `speech_ratio`,
`diarization_used`, `diarization_note` y una lista de avisos (`notes`).

## Privacidad (al final de VER)

Antes de guardar `audio_raw.json`:

- Todo segmento que no es del docente pasa a `"[ALUMNO responde]"`, con su minuto y su duración;
  su texto y sus palabras se descartan. Antes se le aplica el filtro de texto inventado.
- En las líneas del docente, los nombres pasan a `"[NOMBRE]"` (spaCy `es_core_news_sm`): una
  entidad PER, MISC o LOC hecha solo de nombres propios, salvo `privacy.keep_words` y las
  palabras del tema.
- Docente y alumnos se distinguen con la misma regla de INTERPRETAR (el que más habla es el
  docente). Sin diarización, todo pasa a `"[ALUMNO responde]"`.

## LLM (solo texto)

El LLM recibe la transcripción, las preguntas candidatas y las métricas; nunca audio ni video.
Proveedor en `configs/llm.yaml` (`ollama` por defecto, `anthropic` opcional). Parte fija del
prompt: `docs/rubric_definitions.md`. Ventanas de `window_s` con `overlap_s` de solapamiento;
por ventana, un llamado de eventos y otro de preguntas; la unión elimina duplicados.

| Clase | Rúbrica | Qué marca |
|---|---|---|
| `question_open` / `question_closed` / `question_rhetorical` | R2 | Clasificación de cada pregunta candidata. Las retóricas no cuentan como preguntas, ni para la espera, ni para el monitoreo. Una pregunta con respuesta de un alumno nunca es retórica: si el LLM la marca así, se le vuelve a pedir solo entre abierta y cerrada; si insiste, se rechaza. |
| `feedback_discovery` / `feedback_descriptive` / `feedback_elemental` | R3 | Retroalimentación del docente después de una respuesta. La incorrecta no se marca: el LLM no ve lo que respondió el alumno. |
| `regulation_formative` / `regulation_external_control` / `regulation_mistreatment` | R5 (maltrato también R4) | Mecanismos de regulación del comportamiento. |
| `participation_invitation` | R1 | Se dirige a UN estudiante para que participe (nombre, "a ver tú", "hija"). Las preguntas a toda la clase no cuentan. |
| `respect_issue` | R4 | Apodos, burlas o insultos. |

Cada elemento lleva `timestamp`, clase, cita textual, razón breve y confianza (alta/media/baja).
Las preguntas solo llevan `question_id`, clase, razón y confianza: la cita y el minuto los pone
el código desde la pregunta candidata. Las invitaciones llevan además `addressee`, la palabra
con que el docente se dirige al estudiante.
**Validación anti-invento:** la cita debe existir textualmente (sin importar mayúsculas, tildes
ni signos) en una línea cuyo tramo contiene el minuto indicado (± `timestamp_tolerance_s`); si
no, se rechaza. Un `question_id` que no es candidato de esa ventana: rechazado. Una invitación
cuyo `addressee` no está en la cita o es de toda la clase (`whole_class_words`): rechazada.
Clases fuera de la lista: rechazadas. JSON inválido: un reintento.

- `open_question_ratio` (R2) = abiertas / (abiertas + cerradas), con la clasificación del LLM.
- `rubric_evidence`: por rúbrica R1 a R5, métricas con valor y citas; si no hay ninguna,
  "evidencia insuficiente".
- `one_suggestion`: un consejo sobre UNA debilidad, elegida por reglas en este orden: espera
  promedio < `wait_time.ok_threshold_s`, retroalimentación solo elemental, monitoreo <
  `monitoring.threshold` y abiertas < `suggestion.min_open_question_ratio`. Campos: `weakness`,
  `timestamp`, `quote`, `what_we_saw` (escrito por el código con el minuto y la cita reales;
  el monitoreo va sin minuto), `why_it_matters`, `try_this`, `example_phrase` (estos tres, del
  LLM) y `text` (las tres partes juntas). Si no hay ninguna debilidad, no hay consejo.
- Sin LLM: reglas de respaldo y "sin análisis LLM" en el reporte.
- `llm_analysis.usage`: proveedor, modelo, tokens, tiempo y costo (0 con Ollama).

## Archivos de salida (`outputs/<video>/`)

| Archivo | Etapa | Contenido |
|---|---|---|
| `detections.json` | VER | Cajas por frame muestreado. |
| `background_blurred.png` | VER | Primer frame muy desenfocado. |
| `audio_raw.json` | VER | Whisper, Silero y pyannote. |
| `class_report.json` | INTERPRETAR | `metadata`, `video_timeline`, `stage_interaction_events`, `audio_events` (preguntas con espera y respuesta), `metrics` (cada una con `value`, `rubric`, `description`, `note`), `data_quality`, `rubric_evidence`, `llm_analysis` y `one_suggestion`. |
| `transcript.txt` | INTERPRETAR | `[mm:ss] DOCENTE: ...`, `ALUMNO` o `HABLANTE` (sin diarización); `[AUDIO DUDOSO]` en segmentos de baja confianza. |
| `heatmap.png` | INTERPRETAR | Recorrido del docente sobre el fondo desenfocado. |
| `debug.mp4` | Opcional | Caja y estado del docente dibujados (`--debug-video`). |
