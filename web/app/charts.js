"use strict";
/* =========================================================================
   Gráficas de Aula Evidente.

   SVG escrito a mano, sin librerías ni CDN: el reporte tiene que abrirse en el
   evento aunque no haya internet, y así entra completo en el PDF.

   Color: los tonos de marca corridos a la banda de luminosidad que pasa la
   separación para daltonismo (protan/deutan/tritan ΔE ≥ 14). Como el contraste
   sobre el lienzo claro queda por debajo de 3:1, TODA marca lleva su etiqueta
   visible: el color nunca es el único portador del dato.

   Depende de index.html: EVENTS, DURATION, REPORT, COLOR, KIND_LABEL, kindOf,
   el(), $() y HEATMAP_URL.
   ========================================================================= */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function svgText(text, attrs, fontSize, extraStyle = "") {
  const node = svgEl("text", attrs);
  node.setAttribute("style", `font-size:${fontSize}px;${extraStyle}`);
  node.textContent = text;
  return node;
}

function tooltip(target, text) {
  const title = svgEl("title");
  title.textContent = text;
  target.append(title);
  return target;
}

function chartCard({ title, rubric, sub, wide }) {
  const card = el("div", "chart" + (wide ? " wide" : ""));
  const head = el("div", "chart-h");
  head.append(el("span", "t", title));
  if (rubric) head.append(el("span", "r", rubric));
  card.append(head);
  if (sub) card.append(el("div", "chart-sub", sub));
  return card;
}

function legend(items) {
  const box = el("div", "legend");
  for (const [color, label] of items) {
    const row = el("span");
    const swatch = el("i");
    swatch.style.background = color;
    row.append(swatch, document.createTextNode(label));
    box.append(row);
  }
  return box;
}

/* --- 1. Ritmo de preguntas: cuántas por minuto y de qué tipo -------------- */
function chartRhythm() {
  const minutes = Math.max(1, Math.ceil(DURATION / 60));
  const buckets = Array.from({ length: minutes }, () => ({ answered: 0, unanswered: 0, tag: 0 }));
  for (const question of EVENTS) {
    const minute = Math.min(minutes - 1, Math.floor(question.start_s / 60));
    buckets[minute][kindOf(question)]++;
  }
  const peak = Math.max(1, ...buckets.map((b) => b.answered + b.unanswered + b.tag));

  const W = 100, H = 40, PAD_B = 7, PAD_T = 5;
  const bandWidth = W / minutes;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Preguntas por minuto, divididas por si recibieron respuesta",
  });

  for (let line = 0; line <= 2; line++) {
    const y = PAD_T + (H - PAD_T - PAD_B) * (line / 2);
    svg.append(svgEl("line", { x1: 0, y1: y, x2: W, y2: y, class: "c-grid" }));
  }

  // Apiladas, con 0.4 de hueco entre segmentos para que se lean separadas.
  const stack = [["answered", COLOR.answered], ["unanswered", COLOR.unanswered], ["tag", COLOR.tag]];
  buckets.forEach((bucket, index) => {
    const total = bucket.answered + bucket.unanswered + bucket.tag;
    let y = H - PAD_B;
    for (const [key, color] of stack) {
      if (!bucket[key]) continue;
      const height = (H - PAD_T - PAD_B) * (bucket[key] / peak);
      const bar = svgEl("rect", {
        x: index * bandWidth + bandWidth * 0.18, y: y - height,
        width: bandWidth * 0.64, height: Math.max(height - 0.4, 0.4),
        fill: color, rx: 0.8, class: "c-bar",
      });
      tooltip(bar, `Minuto ${index + 1} · ${bucket[key]} ${KIND_LABEL[key]}`);
      svg.append(bar);
      y -= height;
    }
    if (total) {
      svg.append(svgText(total, {
        x: index * bandWidth + bandWidth / 2, y: y - 1.4, "text-anchor": "middle", class: "c-lab",
      }, 3.4));
    }
    svg.append(svgText(`${index + 1}'`, {
      x: index * bandWidth + bandWidth / 2, y: H - 1.4, "text-anchor": "middle", class: "c-axis",
    }, 3.1));
  });

  const card = chartCard({
    title: "Ritmo de preguntas", rubric: "R2 · R3", wide: true,
    sub: "Cuántas preguntas hiciste en cada minuto y cuáles recibieron respuesta.",
  });
  card.append(svg, legend([
    [COLOR.answered, "con respuesta"],
    [COLOR.unanswered, "sin respuesta"],
    [COLOR.tag, "muletilla"],
  ]));
  return card;
}

/* --- 2. Tipo de preguntas (lo clasifica el modelo de lenguaje) ------------ */
function chartQuestionTypes() {
  const labels = (REPORT.llm_analysis || {}).question_labels || [];
  const counts = { question_open: 0, question_closed: 0, question_rhetorical: 0 };
  for (const item of labels) {
    if (item.label in counts) counts[item.label]++;
  }
  const total = counts.question_open + counts.question_closed + counts.question_rhetorical;
  if (!total) return null;

  const rows = [
    ["Abiertas", counts.question_open, COLOR.answered, "piden explicar, comparar o predecir"],
    ["Cerradas", counts.question_closed, COLOR.unanswered, "piden un dato o un sí/no"],
    ["Retóricas", counts.question_rhetorical, COLOR.tag, "no esperan respuesta"],
  ];

  const ROW_H = 13, BAR_X = 21, BAR_W = 58, BAR_H = 6.4;
  const svg = svgEl("svg", {
    viewBox: `0 0 100 ${rows.length * ROW_H}`, role: "img",
    "aria-label": "Preguntas por tipo",
  });

  rows.forEach(([name, count, color, help], index) => {
    const y = index * ROW_H + 2.5;
    svg.append(svgText(name, { x: 0, y: y + 4.6 }, 3.6, "fill:var(--ink-2)"));
    svg.append(svgEl("rect", {
      x: BAR_X, y, width: BAR_W, height: BAR_H, rx: 1.6,
      fill: "var(--panel-2)", stroke: "var(--line)", "stroke-width": 0.4,
    }));
    const bar = svgEl("rect", {
      x: BAR_X, y, width: Math.max(1, BAR_W * (count / total)), height: BAR_H,
      rx: 1.6, fill: color, class: "c-bar",
    });
    tooltip(bar, `${name}: ${count} de ${total} · ${help}`);
    svg.append(bar);
    svg.append(svgText(`${count}   ${Math.round((count / total) * 100)} %`, {
      x: BAR_X + BAR_W + 2.5, y: y + 4.9, class: "c-lab",
    }, 4));
  });

  const card = chartCard({
    title: "Tipo de preguntas", rubric: "R2",
    sub: "Clasificadas una por una por el modelo de lenguaje.",
  });
  card.append(svg);
  return card;
}

/* --- 3. Espera tras preguntar, contra los 3 s de Rowe --------------------- */
const WAIT_EDGES = [0, 1, 2, 3, 5, 8, Infinity];
const WAIT_NAMES = ["0–1 s", "1–2 s", "2–3 s", "3–5 s", "5–8 s", "8 s +"];
const WAIT_OK_FROM = 3;   // índice del primer cubo que cumple el umbral

function chartWait() {
  const waits = EVENTS
    .filter((q) => q.counts_as_question && q.wait_time_s != null)
    .map((q) => q.wait_time_s);
  if (!waits.length) return null;

  const buckets = WAIT_NAMES.map(() => 0);
  for (const wait of waits) {
    for (let i = 0; i < WAIT_NAMES.length; i++) {
      if (wait >= WAIT_EDGES[i] && wait < WAIT_EDGES[i + 1]) { buckets[i]++; break; }
    }
  }
  const peak = Math.max(...buckets);

  const W = 100, H = 42, PAD_B = 8, PAD_T = 6;
  const bandWidth = W / WAIT_NAMES.length;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Distribución del tiempo de espera tras preguntar",
  });

  buckets.forEach((count, index) => {
    const height = peak ? (H - PAD_T - PAD_B) * (count / peak) : 0;
    const meetsThreshold = index >= WAIT_OK_FROM;
    const bar = svgEl("rect", {
      x: index * bandWidth + bandWidth * 0.2, y: H - PAD_B - height,
      width: bandWidth * 0.6, height: Math.max(height, 0.5), rx: 1,
      fill: meetsThreshold ? COLOR.answered : COLOR.tag, class: "c-bar",
    });
    tooltip(bar, `${count} pregunta${count === 1 ? "" : "s"} con espera de ${WAIT_NAMES[index]}`);
    svg.append(bar);

    if (count) {
      svg.append(svgText(count, {
        x: index * bandWidth + bandWidth / 2, y: H - PAD_B - height - 1.4,
        "text-anchor": "middle", class: "c-lab",
      }, 3.6));
    }
    svg.append(svgText(WAIT_NAMES[index], {
      x: index * bandWidth + bandWidth / 2, y: H - 2.6,
      "text-anchor": "middle", class: "c-axis",
    }, 3.1));
  });

  const refX = bandWidth * WAIT_OK_FROM;
  svg.append(svgEl("line", { x1: refX, y1: PAD_T - 4, x2: refX, y2: H - PAD_B, class: "c-ref" }));
  svg.append(svgText("3 s", { x: refX + 1.6, y: PAD_T - 4.4, class: "c-reft" }, 3.2));

  const card = chartCard({
    title: "Cuánto esperaste", rubric: "R2",
    sub: "Desde los 3 s responden más estudiantes y con más detalle (Rowe).",
  });
  card.append(svg, legend([
    [COLOR.answered, "3 s o más"],
    [COLOR.tag, "menos de 3 s"],
  ]));
  return card;
}

/* --- 4. Mapa de calor del recorrido --------------------------------------- */
function chartHeatmap() {
  const card = chartCard({
    title: "Por dónde te moviste", rubric: "R4",
    sub: "Zonas donde más tiempo estuviste, sobre el aula desenfocada.",
  });
  const frame = el("div", "heat");
  const image = document.createElement("img");
  image.src = HEATMAP_URL;
  image.alt = "Mapa de calor del recorrido del docente en el aula";
  image.loading = "lazy";
  // Si no hay heatmap (se corrió sin visión), la tarjeta se quita sola.
  image.addEventListener("error", () => card.remove());
  frame.append(image);
  card.append(frame, el("div", "heat-note",
    "El fondo va desenfocado a propósito: no se reconoce a nadie. No se guardan rostros."));
  return card;
}

function renderCharts() {
  const host = $("#charts");
  if (!host) return;
  host.innerHTML = "";
  for (const card of [chartRhythm(), chartWait(), chartQuestionTypes(), chartHeatmap()]) {
    if (card) host.append(card);
  }
}
