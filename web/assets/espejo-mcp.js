/* Tarjeta "Conecta tu asistente de IA" (servidor MCP de Espejo).

   Muestra el enlace del docente (assistant_url) con un botón para copiarlo y los pasos
   para Claude, ChatGPT y Claude Code. El enlace incluye la clave: solo se muestra en el
   navegador de quien subió la clase y se avisa que no se comparta. */
(function () {
  "use strict";

  const STEPS = {
    "Claude": [
      "Abre claude.ai (o la app de escritorio) → Configuración → Conectores.",
      "Elige \"Agregar conector personalizado\": nombre Espejo y pega tu enlace.",
      "En un chat, activa Espejo en el menú de herramientas y pregunta: \"¿Qué tal mi última clase?\"",
    ],
    "ChatGPT": [
      "Configuración → Aplicaciones y conectores → Avanzado → activa el modo desarrollador.",
      "Crea un conector con tu enlace (sin autenticación).",
      "Actívalo en el chat y pide: \"Resume mi última clase y dame un plan para la próxima\".",
    ],
    "Claude Code": [
      "En una terminal: claude mcp add --transport http espejo <tu enlace>",
      "Abre claude y pregunta: \"¿Qué evidencias tengo en R2?\"",
    ],
    "Otros": [
      "Cualquier asistente que acepte servidores MCP remotos (transporte HTTP) sirve: agrega uno nuevo con tu enlace.",
      "Para probarlo sin asistente: npx @modelcontextprotocol/inspector → Streamable HTTP → tu enlace.",
    ],
  };
  const QUESTIONS = [
    "¿Qué clases tengo?",
    "Dame el resumen de mi última clase.",
    "Muéstrame las evidencias de R2.",
    "Compara mis dos últimas clases.",
    "Ayúdame a planificar mi próxima clase.",
  ];

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  async function copy(text, button) {
    const before = button.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "¡Enlace copiado!";
    } catch {
      button.textContent = "Selecciónalo y usa Ctrl+C";
    }
    button.classList.add("done");
    setTimeout(() => { button.textContent = before; button.classList.remove("done"); }, 2200);
  }

  function stepsBlock() {
    const box = el("div", "mcp-steps");
    const tabs = el("div", "mcp-tabs");
    const body = el("ol", "mcp-list");
    const show = (name) => {
      for (const tab of tabs.children) tab.classList.toggle("on", tab.dataset.name === name);
      body.innerHTML = "";
      for (const step of STEPS[name]) body.append(el("li", null, step));
    };
    for (const name of Object.keys(STEPS)) {
      const tab = el("button", "mcp-tab", name);
      tab.type = "button";
      tab.dataset.name = name;
      tab.addEventListener("click", () => show(name));
      tabs.append(tab);
    }
    box.append(tabs, body);
    show("Claude");
    return box;
  }

  /* assistantUrl null: explica cómo funciona sin mostrar ningún enlace (demo u otro navegador). */
  function renderMcpCard(host, assistantUrl) {
    const card = el("section", "mcp-card");
    const head = el("div", "mcp-head");
    head.append(el("span", "mcp-badge", "MCP"), el("h3", null, "Conecta tu asistente de IA"));
    card.append(head);
    card.append(el("p", "mcp-lead",
      "Tu asistente (Claude, ChatGPT u otro) puede leer tus reportes y ayudarte a planificar. " +
      "Solo lee: no cambia nada y solo ve datos anonimizados."));

    if (assistantUrl) {
      const row = el("div", "mcp-link");
      const input = el("input", "mcp-url");
      input.value = assistantUrl;
      input.readOnly = true;
      input.setAttribute("aria-label", "Enlace para tu asistente de IA");
      input.addEventListener("focus", () => input.select());
      const button = el("button", "mcp-copy", "Copiar enlace para mi asistente");
      button.type = "button";
      button.addEventListener("click", () => copy(assistantUrl, button));
      row.append(input, button);
      card.append(row);
      card.append(el("p", "mcp-warn",
        "Guárdalo y no lo compartas: quien tenga este enlace puede ver tus reportes. " +
        "Solo se muestra en este navegador."));
    } else {
      card.append(el("p", "mcp-warn",
        "El enlace se entrega una sola vez a quien sube una clase y queda guardado solo en su " +
        "navegador. Sube una clase en \"Analiza tu clase\" para obtener el tuyo."));
    }

    card.append(stepsBlock());
    const ask = el("div", "mcp-ask");
    ask.append(el("span", "mcp-ask-t", "Prueba a preguntarle:"));
    for (const question of QUESTIONS) ask.append(el("span", "mcp-chip", question));
    card.append(ask);
    host.prepend(card);
    return card;
  }

  window.EspejoMcp = { renderMcpCard };
})();
