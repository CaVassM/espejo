"use strict";
/* =========================================================================
   Encuadres: lo que vieron los modelos de visión, dibujado sobre la clase.

   Dos modos, según lo que haya:
   - Con video (en el equipo del docente): las cajas se dibujan encima del
     video y siguen su reproducción.
   - Sin video (reporte publicado): las cajas se animan sobre el fondo del
     aula ya desenfocado. Se ve que el modelo funciona sin mostrar a nadie.

   Las cajas vienen en el espacio del frame analizado (640x288 en esta clase):
   se escalan al tamaño que ocupe el lienzo en pantalla.
   ========================================================================= */

const CAJA_ESTILO = {
  teacher:           { color: "#2A5FA8", etiqueta: "Docente" },
  guide:             { color: "#14B8A6", etiqueta: "Guiando" },
  board_writing:     { color: "#1E4D8F", etiqueta: "En la pizarra" },
  answer:            { color: "#D99A1F", etiqueta: "Alumno" },
  stage_interaction: { color: "#C2410C", etiqueta: "Al frente" },
};

/* Debajo de esta confianza la caja no se dibuja: el modelo no está seguro y
   mostrarla haría parecer que detecta cosas que no están. */
const CONFIANZA_MINIMA = 0.5;

let DETS = null;          // {width, height, duration_s, frames:[{t, d:[{c,p,b}]}]}
let lienzo = null, ctx = null;
let reproduciendo = false, tiempoSinVideo = 0, ultimoLatido = 0;
let etiquetasUsadas = [];

/* El motor escribe detections.json con una forma, y la web viaja con una
   versión ligera de otra. Se acepta cualquiera de las dos y se deja una sola. */
function normalizar(bruto) {
  if (!bruto) return null;
  if (Array.isArray(bruto.frames) && bruto.frames.length && "d" in bruto.frames[0]) {
    return bruto;                                   // ya es la versión ligera
  }
  const video = bruto.video || {};
  const frames = [];
  for (const f of bruto.frames || []) {
    const cajas = [];
    for (const t of f.teacher || []) {
      cajas.push({ c: "teacher", p: t.confidence, b: t.box });
    }
    for (const b of f.behaviors || []) {
      cajas.push({ c: b.class_name, p: b.confidence, b: b.box });
    }
    if (cajas.length) frames.push({ t: f.timestamp_s, d: cajas });
  }
  return {
    width: video.width, height: video.height,
    duration_s: video.duration_s, sample_fps: bruto.sample_fps, frames,
  };
}

async function cargarDetecciones(fuentes) {
  for (const url of fuentes) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (res.ok) {
        const datos = normalizar(await res.json());
        if (datos && datos.width && datos.frames.length) return datos;
      }
    } catch { /* se prueba la siguiente */ }
  }
  return null;
}

function frameEn(segundo) {
  if (!DETS || !DETS.frames.length) return null;
  // Los frames están ordenados; búsqueda binaria por tiempo.
  let bajo = 0, alto = DETS.frames.length - 1, mejor = 0;
  while (bajo <= alto) {
    const medio = (bajo + alto) >> 1;
    if (DETS.frames[medio].t <= segundo) { mejor = medio; bajo = medio + 1; }
    else alto = medio - 1;
  }
  const f = DETS.frames[mejor];
  // Si el frame más cercano queda a más de un muestreo de distancia, no hay dato.
  return Math.abs(f.t - segundo) <= (1 / (DETS.sample_fps || 2)) * 1.5 ? f : null;
}

function dibujarCaja(caja, escalaX, escalaY) {
  const estilo = CAJA_ESTILO[caja.c];
  if (!estilo || caja.p < CONFIANZA_MINIMA) return;

  const [x1, y1, x2, y2] = caja.b;
  const x = x1 * escalaX, y = y1 * escalaY;
  const ancho = (x2 - x1) * escalaX, alto = (y2 - y1) * escalaY;

  ctx.lineWidth = 2;
  ctx.strokeStyle = estilo.color;
  ctx.beginPath();
  ctx.roundRect(x, y, ancho, alto, 4);
  ctx.stroke();

  // Etiqueta encima de la caja, o dentro si no hay sitio arriba.
  const texto = `${estilo.etiqueta} ${Math.round(caja.p * 100)}%`;
  ctx.font = "600 11px system-ui, sans-serif";
  const anchoTexto = ctx.measureText(texto).width + 10;
  const altoEtiqueta = 16;

  // Dos cajas que se superponen dejarían sus etiquetas una encima de otra y no
  // se leería ninguna: se busca la primera franja libre hacia abajo.
  let yEtiqueta = y - altoEtiqueta >= 0 ? y - altoEtiqueta : y;
  const choca = (yy) => etiquetasUsadas.some((u) =>
    yy < u.y + altoEtiqueta && yy + altoEtiqueta > u.y &&
    x < u.x + u.ancho && x + anchoTexto > u.x);
  let intentos = 0;
  while (choca(yEtiqueta) && intentos < 6) { yEtiqueta += altoEtiqueta + 2; intentos++; }
  etiquetasUsadas.push({ x, y: yEtiqueta, ancho: anchoTexto });

  ctx.fillStyle = estilo.color;
  ctx.beginPath();
  ctx.roundRect(x, yEtiqueta, anchoTexto, altoEtiqueta, 3);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(texto, x + 5, yEtiqueta + 11.5);
}

function pintar(segundo) {
  if (!ctx || !DETS) return;
  const rect = lienzo.getBoundingClientRect();
  if (!rect.width) return;

  // El lienzo se ajusta a su tamaño real en pantalla para no verse borroso.
  const dpr = window.devicePixelRatio || 1;
  if (lienzo.width !== Math.round(rect.width * dpr)) {
    lienzo.width = Math.round(rect.width * dpr);
    lienzo.height = Math.round(rect.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  etiquetasUsadas = [];
  const frame = frameEn(segundo);
  if (!frame) return;
  const escalaX = rect.width / DETS.width;
  const escalaY = rect.height / DETS.height;

  // El docente se dibuja al final para que su etiqueta no quede tapada por la
  // de un alumno: es la caja que importa leer.
  const orden = [...frame.d].sort((a, b) => (a.c === "teacher") - (b.c === "teacher"));
  for (const caja of orden) dibujarCaja(caja, escalaX, escalaY);
}

/* Sin video, el tiempo lo lleva esta animación. */
function latido(ahora) {
  if (!reproduciendo) return;
  if (ultimoLatido) {
    tiempoSinVideo += (ahora - ultimoLatido) / 1000;
    if (tiempoSinVideo > (DETS.duration_s || 0)) tiempoSinVideo = 0;
  }
  ultimoLatido = ahora;
  pintar(tiempoSinVideo);
  actualizarTransporte();
  requestAnimationFrame(latido);
}

function actualizarTransporte() {
  const barra = document.querySelector("#encBarra");
  const reloj = document.querySelector("#encReloj");
  if (barra) barra.style.width = (tiempoSinVideo / (DETS.duration_s || 1) * 100) + "%";
  if (reloj) reloj.textContent = mmss(tiempoSinVideo);
}

function leyenda() {
  const box = el("div", "enc-leyenda");
  for (const [clase, estilo] of Object.entries(CAJA_ESTILO)) {
    // stage_interaction casi no aparece en esta clase; se muestra igual para
    // que se entienda qué sabe reconocer el modelo.
    const item = el("span");
    const punto = el("i");
    punto.style.background = estilo.color;
    item.append(punto, document.createTextNode(estilo.etiqueta));
    box.append(item);
  }
  return box;
}

async function renderEncuadres() {
  const host = document.querySelector("#encuadres");
  if (!host) return;

  DETS = await cargarDetecciones(DETECTION_SOURCES);
  if (!DETS) { host.remove(); return; }

  /* Red de seguridad: las cajas deben ser de ESTA clase. Si la duración no
     coincide con la del reporte, vienen de otra sesión y no se dibuja nada.
     Un cuadro sobre la persona equivocada es peor que no mostrar cuadro. */
  const duracionReporte = REPORT?.metadata?.duration_s;
  if (duracionReporte && DETS.duration_s &&
      Math.abs(DETS.duration_s - duracionReporte) > 2) {
    console.warn("Encuadres descartados: son de otra clase",
      { deteccion: DETS.duration_s, reporte: duracionReporte });
    host.remove();
    return;
  }

  const video = document.querySelector("#vid");
  const card = chartCard({
    title: "Lo que vieron los modelos", rubric: "visión", wide: true,
    sub: video
      ? "Las cajas siguen el video: el docente, y los estudiantes cuando participan."
      : "Sobre el aula desenfocada, para no mostrar a nadie. Las cajas son las que detectó el modelo en esa clase.",
  });

  lienzo = document.createElement("canvas");
  lienzo.className = "enc-lienzo";

  if (video) {
    // Con video, las cajas van ENCIMA del video, no en una tarjeta aparte: es
    // lo que se espera ver de un detector. El lienzo necesita un contenedor
    // posicionado, así que se envuelve el reproductor.
    let marcoVideo = video.parentElement;
    if (!marcoVideo.classList.contains("vid-marco")) {
      marcoVideo = el("div", "vid-marco");
      video.parentElement.insertBefore(marcoVideo, video);
      marcoVideo.append(video);
    }
    marcoVideo.append(lienzo);
    card.append(el("p", "enc-nota",
      "Las cajas se dibujan sobre el reproductor, a la izquierda, y siguen la reproducción."));
  } else {
    const marco = el("div", "enc-marco");
    const fondo = document.createElement("img");
    fondo.src = BACKGROUND_URL;
    fondo.alt = "";
    fondo.className = "enc-fondo";
    marco.append(fondo, lienzo);
    card.append(marco);
  }

  if (!video) {
    const transporte = el("div", "enc-transporte");
    const boton = el("button", "enc-play", "Reproducir");
    boton.type = "button";
    boton.addEventListener("click", () => {
      reproduciendo = !reproduciendo;
      boton.textContent = reproduciendo ? "Pausar" : "Reproducir";
      ultimoLatido = 0;
      if (reproduciendo) requestAnimationFrame(latido);
    });
    const pista = el("div", "enc-pista");
    const barra = el("div", "enc-barra");
    barra.id = "encBarra";
    pista.append(barra);
    const reloj = el("span", "enc-reloj", "00:00");
    reloj.id = "encReloj";
    transporte.append(boton, pista, reloj);
    card.append(transporte);
  }

  card.append(leyenda());
  host.append(card);

  if (video) {
    const seguir = () => pintar(video.currentTime);
    video.addEventListener("timeupdate", seguir);
    video.addEventListener("seeked", seguir);
    addEventListener("resize", seguir);
    ctx = lienzo.getContext("2d");
    seguir();
  } else {
    ctx = lienzo.getContext("2d");
    pintar(0);
    addEventListener("resize", () => pintar(tiempoSinVideo));
  }
}
