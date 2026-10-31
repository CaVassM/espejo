"""Servidor de la demo: sirve la app del reporte, los outputs y el video de la clase.

Un solo comando, desde la raíz del repo:

    python services/serve_demo.py

Luego abrir http://localhost:8000/

Por qué no `python -m http.server`: no responde peticiones Range, así que el navegador
no puede saltar a un minuto del video sin descargarlo entero. Saltar al minuto es
justamente lo que la ficha tiene que demostrar.

Se montan solo tres carpetas, nunca la raíz del repo: `.env` no debe salir por HTTP.
Esto es para desarrollo y demo; la API de producción es services/api.
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# La puerta de entrada es la landing; el reporte vive un nivel más abajo, en /web/app/.
HOME_URL_PATH = "/web/"
APP_URL_PATH = "/web/app/"
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"

# Carpetas servidas: ruta URL -> carpeta en disco. La raíz del repo nunca se monta.
MOUNTS = {
    "/web": REPO_ROOT / "web",
    "/outputs": REPO_ROOT / "outputs",
    "/samples": REPO_ROOT / "samples",
}

REPORT_PATH = REPO_ROOT / "outputs" / "clase" / "class_report.json"

logger = logging.getLogger("espejo.demo")


def check_requirements() -> list[str]:
    """Avisa de lo que falta antes de levantar, en vez de fallar en el navegador."""
    problems = []
    if not REPORT_PATH.exists():
        problems.append(
            f"Falta {REPORT_PATH.relative_to(REPO_ROOT)}\n"
            "    Genéralo con: python scripts/interpret.py --run outputs/clase/ --no-llm"
        )
    samples = REPO_ROOT / "samples"
    if not samples.exists() or not list(samples.glob("*.mp4")):
        problems.append(
            "Falta el video en samples/\n"
            "    Déjalo como samples/frag_02.mp4 (el nombre que registra el reporte)"
        )
    return problems


def build_app():
    # Import local: el servidor de demo no es parte del motor ni de sus dependencias base.
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Aula Evidente · demo", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def sin_cache(request, call_next):
        """Nada de caché: en el demo siempre se ve la última versión.

        Sin Cache-Control, el navegador decide por su cuenta cuánto guardar una
        página y puede seguir mostrando una copia vieja después de editarla. En
        desarrollo eso hace perder horas, y en el demo sería presentar algo que
        ya no es lo que está en disco.
        """
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse(HOME_URL_PATH)

    for url_path, directory in MOUNTS.items():
        directory.mkdir(parents=True, exist_ok=True)
        # StaticFiles de Starlette responde Range: el video se puede recorrer sin descargarlo.
        app.mount(url_path, StaticFiles(directory=directory, html=True), name=url_path.strip("/"))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--no-browser", action="store_true", help="no abre el navegador solo")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    problems = check_requirements()
    if problems:
        print("\nNo se puede levantar la demo todavía:\n")
        for problem in problems:
            print(f"  - {problem}")
        print()
        sys.exit(1)

    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Espejo · demo lista en  {url}\n")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
