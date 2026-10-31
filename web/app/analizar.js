"use strict";
/* =========================================================================
   Analizar una clase: crear el análisis, subir el video y esperar el reporte.

   Sigue docs/api_contract.md:
     1. POST $API/api/analyses          -> analysis_id + permiso de subida
     2. POST {upload.url}  (a Modal)    -> el video, con el permiso
     3. GET  $API/api/analyses/{id}     -> cada ~5 s hasta "done" o "error"
     4. se abre la ficha con ?analysis={id}

   El código de acceso lo escribe el docente: no viaja dentro de la página.
   ========================================================================= */

const $ = (sel) => document.querySelector(sel);

const STAGE_FALLBACK = {
  waiting_upload: "Esperando el video",
  processing: "Analizando la clase",
  interpreting: "Preparando el reporte",
  done: "Listo",
  error: "Hubo un problema",
};

/* Mensajes por código de error. Decir "error 413" no ayuda a nadie. */
const UPLOAD_ERRORS = {
  401: "El permiso de subida venció. Vuelve a empezar.",
  409: "Ese análisis ya tiene un video.",
  413: `El video pesa más de ${MAX_UPLOAD_MB} MB.`,
  415: "Ese formato de video no se admite.",
};

let yaEnviado = false;

function mostrar(id) {
  for (const seccion of document.querySelectorAll(".paso")) {
    seccion.classList.toggle("hide", seccion.id !== id);
  }
}

function fallar(mensaje, detalle) {
  $("#errorMsg").textContent = mensaje;
  $("#errorDetalle").textContent = detalle || "";
  $("#errorDetalle").classList.toggle("hide", !detalle);
  mostrar("paso-error");
}

function describirArchivo(file) {
  const mb = file.size / (1024 * 1024);
  return `${file.name} · ${mb.toFixed(1)} MB`;
}

/* Se revisa antes de subir: media hora de subida para recibir un 415 es cruel. */
function problemaConElArchivo(file) {
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (!VIDEO_EXTS.includes(ext)) {
    return `Ese archivo es .${ext}. Se admiten: ${VIDEO_EXTS.join(", ")}.`;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `El video pesa ${(file.size / 1048576).toFixed(0)} MB y el máximo es ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

async function crearAnalisis(codigo, tema, claveDelDocente) {
  const cuerpo = { access_code: codigo, topic: tema || null };
  // Con la clave de una clase anterior, la nueva se suma al mismo asistente de IA
  if (claveDelDocente) cuerpo.teacher_key = claveDelDocente;
  const respuesta = await fetch(`${API_BASE}/api/analyses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cuerpo),
  });
  if (respuesta.status === 401) throw new Error("CODIGO");
  if (!respuesta.ok) throw new Error(`La API respondió ${respuesta.status}`);
  return respuesta.json();
}

/* Se usa XMLHttpRequest y no fetch porque solo XHR informa del avance de subida,
   y con un video de varios cientos de MB una barra quieta parece algo colgado. */
function subirVideo(upload, file, alAvanzar) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const [clave, valor] of Object.entries(upload.fields || {})) form.append(clave, valor);
    form.append("video", file);

    const peticion = new XMLHttpRequest();
    peticion.open("POST", upload.url);
    peticion.setRequestHeader("X-Upload-Ticket", upload.ticket);
    peticion.upload.addEventListener("progress", (evento) => {
      if (evento.lengthComputable) alAvanzar(evento.loaded / evento.total);
    });
    peticion.addEventListener("load", () => {
      if (peticion.status >= 200 && peticion.status < 300) resolve();
      else reject(new Error(UPLOAD_ERRORS[peticion.status] || `La subida falló (${peticion.status}).`));
    });
    peticion.addEventListener("error", () =>
      reject(new Error("Se cortó la conexión durante la subida.")));
    peticion.send(form);
  });
}

async function esperarElReporte(id) {
  for (;;) {
    await new Promise((listo) => setTimeout(listo, POLL_MS));

    let estado;
    try {
      const respuesta = await fetch(`${API_BASE}/api/analyses/${id}`, { cache: "no-store" });
      if (!respuesta.ok) continue;          // un fallo suelto no tumba la espera
      estado = await respuesta.json();
    } catch {
      continue;                             // sin red por un momento: se reintenta
    }

    $("#etapa").textContent = estado.stage || STAGE_FALLBACK[estado.status] || "Procesando";
    if (estado.status === "done") return;
    if (estado.status === "error") throw new Error(estado.error || "El análisis no pudo terminar.");
  }
}

async function analizar(evento) {
  evento.preventDefault();
  if (yaEnviado) return;                    // doble clic: no se crean dos análisis

  const file = $("#video").files[0];
  if (!file) return fallar("Elige el video de tu clase.");

  const problema = problemaConElArchivo(file);
  if (problema) return fallar("Ese archivo no sirve", problema);

  yaEnviado = true;
  $("#etapa").textContent = "Creando el análisis";
  $("#barra").style.width = "0%";
  $("#pct").textContent = "";
  mostrar("paso-progreso");

  let analisis;
  try {
    const clave = $("#sumar").checked ? EspejoMcp.lastTeacherKey() : null;
    analisis = await crearAnalisis($("#codigo").value.trim(), $("#tema").value.trim(), clave);
  } catch (error) {
    yaEnviado = false;
    if (error.message === "CODIGO") {
      return fallar("Código de acceso incorrecto",
        "Pídeselo a quien administra Aula Evidente.");
    }
    return fallar("No se pudo crear el análisis", error.message);
  }

  // El enlace del asistente (MCP) llega UNA sola vez: se guarda en este navegador y se muestra
  EspejoMcp.saveLink({
    id: analisis.analysis_id,
    topic: $("#tema").value.trim() || null,
    createdAt: analisis.created_at,
    teacherKey: analisis.teacher_key,
    assistantUrl: analisis.assistant_url,
  });
  $("#mcpHost").innerHTML = "";
  EspejoMcp.renderMcpCard($("#mcpHost"), analisis.assistant_url);

  $("#etapa").textContent = "Subiendo el video";
  try {
    await subirVideo(analisis.upload, file, (fraccion) => {
      const pct = Math.round(fraccion * 100);
      $("#barra").style.width = pct + "%";
      $("#pct").textContent = pct + " %";
    });
  } catch (error) {
    yaEnviado = false;
    return fallar("No se pudo subir el video", error.message);
  }

  // El video queda en este navegador para que la ficha pueda reproducirlo: la
  // API no lo devuelve, y así la grabación no necesita volver de la red.
  await guardarVideo(analisis.analysis_id, file);

  // Subido: a partir de aquí el avance lo marca el servidor, no la barra.
  $("#barra").style.width = "100%";
  $("#pct").textContent = "";
  $("#etapa").textContent = "En cola";
  $("#progresoNota").textContent =
    "Un clip de 5 minutos tarda unos 3 o 4 minutos. Puedes dejar esta pestaña abierta.";

  try {
    await esperarElReporte(analisis.analysis_id);
  } catch (error) {
    yaEnviado = false;
    return fallar("El análisis no pudo terminar", error.message);
  }

  location.href = `./?analysis=${encodeURIComponent(analisis.analysis_id)}`;
}

/* Deja el archivo en el input y refresca lo que se ve. */
function ponerArchivo(file) {
  if (!file) return;
  const datos = new DataTransfer();
  datos.items.add(file);
  $("#video").files = datos.files;
  $("#video").dispatchEvent(new Event("change", { bubbles: true }));
}

function arrancar() {
  $("#formulario").addEventListener("submit", analizar);

  // El botón abre el diálogo por JavaScript. Un <label for> apuntando a un input
  // oculto no siempre lo consigue, y entonces el botón parece no hacer nada.
  $("#elegir").addEventListener("click", () => $("#video").click());

  $("#video").addEventListener("change", () => {
    const file = $("#video").files[0];
    $("#archivo").textContent = file ? describirArchivo(file) : "Ningún archivo elegido";
  });

  // Soltar el video sobre la zona también sirve.
  const zona = $("#zona");
  for (const evento of ["dragenter", "dragover"]) {
    zona.addEventListener(evento, (e) => { e.preventDefault(); zona.classList.add("encima"); });
  }
  for (const evento of ["dragleave", "drop"]) {
    zona.addEventListener(evento, (e) => { e.preventDefault(); zona.classList.remove("encima"); });
  }
  zona.addEventListener("drop", (e) => ponerArchivo(e.dataTransfer.files[0]));

  // El botón solo se habilita con consentimiento marcado y archivo elegido.
  const revisar = () => {
    $("#enviar").disabled = !($("#consent").checked && $("#video").files.length > 0);
  };
  $("#consent").addEventListener("change", revisar);
  $("#video").addEventListener("change", revisar);
  revisar();

  $("#reintentar").addEventListener("click", () => mostrar("paso-formulario"));

  // Solo si este navegador ya tiene el enlace de una clase anterior
  $("#sumarRow").classList.toggle("hide", !EspejoMcp.lastTeacherKey());

  // El plan gratuito de Render duerme tras 15 min: se despierta al abrir la página
  // para que el primer envío no tarde un minuto en responder.
  fetch(`${API_BASE}/api/health`).catch(() => {});
}

document.addEventListener("DOMContentLoaded", arrancar);
