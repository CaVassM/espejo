"use strict";
/* Configuración de la web. No hay secretos aquí: la URL de la API es pública y
   el código de acceso lo escribe el docente, nunca viaja dentro de la página. */

const API_BASE = "https://espejo-api-o8td.onrender.com";

/* Cada cuánto se pregunta por el avance. El contrato de la API pide ~5 s; además
   mantiene despierto el plan gratuito de Render. */
const POLL_MS = 5000;

/* Límites que impone el worker. Se comprueban antes de subir para no gastar
   minutos de subida en un archivo que va a ser rechazado. */
const MAX_UPLOAD_MB = 1024;
const VIDEO_EXTS = ["mp4", "mov", "m4v", "mkv", "webm", "avi"];
