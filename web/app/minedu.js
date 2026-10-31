"use strict";
/* =========================================================================
   Mapeo de las 5 rúbricas del Minedu.

   Una tira de R1 a R5 que se lee de un vistazo: en cuáles hay evidencia y en
   cuáles no se observó nada en el fragmento analizado. Al hacer clic se abre
   esa rúbrica en la pestaña "Rúbricas".

   Regla de lenguaje (del documento rector del equipo): nunca una nota ni un
   nivel I-IV, nunca rojo, nunca "falló". El estado más bajo dice
   "no se observó", que es lo único que los datos permiten afirmar.
   ========================================================================= */

const RUBRIC_SHORT = {
  R1: "Involucra a los estudiantes",
  R2: "Promueve el razonamiento",
  R3: "Evalúa y retroalimenta",
  R4: "Ambiente de respeto",
  R5: "Regula el comportamiento",
};

/* La métrica que mejor resume cada rúbrica, para mostrar un número por tarjeta. */
const RUBRIC_HEADLINE = {
  R1: ["response_rate", "pct", "respondieron"],
  R2: ["open_question_ratio", "pct", "abiertas"],
  R3: ["monitoring_ratio", "pct", "monitoreando"],
  R4: ["path_length_normalized", "num", "de recorrido"],
  R5: [null, null, null],
};

function rubricHeadline(code) {
  const [name, kind, suffix] = RUBRIC_HEADLINE[code] || [];
  if (!name) return null;
  const value = m(name).value;
  if (value == null) return null;
  return `${fmt(value, kind)} ${suffix}`;
}

function renderMinedu() {
  const host = $("#map5");
  if (!host) return;
  host.innerHTML = "";

  const byCode = {};
  for (const item of REPORT.rubric_evidence || []) byCode[item.rubric] = item;

  for (const code of ["R1", "R2", "R3", "R4", "R5"]) {
    const item = byCode[code] || { status: "evidencia insuficiente", citations: [] };
    const hasEvidence = item.status === "con evidencia";
    const citations = (item.citations || []).length;

    const card = el("button", "m5");
    card.type = "button";
    card.style.setProperty("--s", hasEvidence ? "var(--data-teal)" : "var(--line-strong)");
    card.append(el("div", "c", code), el("div", "n", RUBRIC_SHORT[code]));

    const headline = hasEvidence ? rubricHeadline(code) : null;
    card.append(el("div", "v", headline
      || (hasEvidence ? `${citations} cita${citations === 1 ? "" : "s"}` : "no se observó")));

    card.title = hasEvidence
      ? `${item.title || RUBRIC_TITLES[code]} — ${(item.metrics || []).length} métricas, ${citations} citas`
      : `${RUBRIC_TITLES[code]} — sin evidencia en el fragmento analizado`;

    card.addEventListener("click", () => openRubric(code));
    host.append(card);
  }
}

/* Lleva a la pestaña "Rúbricas" y abre la ficha pedida. */
function openRubric(code) {
  const tab = $("#tab-rub");
  if (tab) tab.click();
  const target = document.querySelector(`.rub[data-code="${code}"]`);
  if (!target) return;
  target.open = true;
  target.scrollIntoView({ block: "nearest", behavior: "smooth" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1200);
}
