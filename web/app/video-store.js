"use strict";
/* =========================================================================
   El video de la clase, guardado en el propio navegador.

   La API no devuelve el video, y hace bien: una grabación de clase no debería
   estar disponible para descargar desde un servidor. Pero el docente acaba de
   elegir ese archivo en su máquina, así que se guarda aquí (IndexedDB) y la
   ficha lo reproduce desde el disco del usuario. El video nunca vuelve de la
   red: sale una vez hacia el worker para analizarse y nada más.

   Consecuencia esperada: si se abre el enlace del reporte en otro navegador o
   en otro equipo, no hay video y la ficha lo dice. Las evidencias siguen con su
   minuto y su cita.
   ========================================================================= */

const DB_NOMBRE = "aula-evidente";
const DB_VERSION = 1;
const ALMACEN = "videos";

/* Un video de clase pesa cientos de MB. Se conservan solo los últimos, para no
   llenar el disco del docente con grabaciones que ya revisó. */
const MAX_VIDEOS_GUARDADOS = 2;

function abrirBase() {
  return new Promise((listo, fallo) => {
    const peticion = indexedDB.open(DB_NOMBRE, DB_VERSION);
    peticion.onupgradeneeded = () => {
      const db = peticion.result;
      if (!db.objectStoreNames.contains(ALMACEN)) {
        db.createObjectStore(ALMACEN, { keyPath: "id" });
      }
    };
    peticion.onsuccess = () => listo(peticion.result);
    peticion.onerror = () => fallo(peticion.error);
  });
}

function enTransaccion(db, modo, trabajo) {
  return new Promise((listo, fallo) => {
    const tx = db.transaction(ALMACEN, modo);
    const resultado = trabajo(tx.objectStore(ALMACEN));
    tx.oncomplete = () => listo(resultado && resultado.result);
    tx.onerror = () => fallo(tx.error);
    tx.onabort = () => fallo(tx.error);
  });
}

/* Guarda el video de un análisis. Si no se puede (sin espacio, modo privado),
   no se interrumpe nada: la ficha simplemente saldrá sin reproductor. */
async function guardarVideo(analysisId, file) {
  try {
    const db = await abrirBase();
    await enTransaccion(db, "readwrite", (almacen) =>
      almacen.put({ id: analysisId, nombre: file.name, tipo: file.type,
                    guardado: Date.now(), blob: file }));
    await limpiarAntiguos(db);
    db.close();
    return true;
  } catch (error) {
    console.info("No se pudo guardar el video en este navegador:", error?.name || error);
    return false;
  }
}

/* Devuelve el video de un análisis, o null si no está en este navegador. */
async function recuperarVideo(analysisId) {
  try {
    const db = await abrirBase();
    const guardado = await enTransaccion(db, "readonly", (almacen) => almacen.get(analysisId));
    db.close();
    return guardado ? guardado.blob : null;
  } catch {
    return null;
  }
}

async function limpiarAntiguos(db) {
  const todos = await enTransaccion(db, "readonly", (almacen) => almacen.getAll());
  if (!todos || todos.length <= MAX_VIDEOS_GUARDADOS) return;

  const sobran = todos
    .sort((a, b) => b.guardado - a.guardado)
    .slice(MAX_VIDEOS_GUARDADOS);
  await enTransaccion(db, "readwrite", (almacen) => {
    for (const viejo of sobran) almacen.delete(viejo.id);
  });
}
