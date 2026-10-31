"""Prueba el worker desplegado de punta a punta con un clip corto (5 min).

Sube el clip, espera, descarga el resultado y revisa:
- que detections.json y audio_raw.json tengan el formato de docs/worker_contract.md;
- que no haya texto de alumnos (solo "[ALUMNO responde]").
Mide el arranque en frío, el proceso y el costo estimado. El resultado queda en outputs/.

Uso:  python scripts/test_modal_worker.py --video samples/clip.mp4 [--topic "..."] [--delete-after]
"""

import argparse
import json
import time
from pathlib import Path

import httpx
import modal

from espejo.audio.capture_io import load_audio_capture
from espejo.audio.diarization import assign_speakers
from espejo.audio.schemas import STUDENT_PLACEHOLDER, SpeakerRole
from espejo.audio.speaker_roles import assign_roles
from espejo.common.config_io import CONFIGS_DIR, OUTPUTS_DIR, read_yaml
from espejo.common.env import read_secret

WEB_FUNCTION = "web"
POLL_INTERVAL_S = 10
STUCK_AFTER_S = 20 * 60  # plan B: si pasa de 20 min, avisar
UPLOAD_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 120
# Precios de Modal (modal.com/pricing): GPU T4 y CPU/memoria del contenedor web
T4_USD_PER_S = 0.000164
CPU_CORE_USD_PER_S = 0.0000131
GIB_USD_PER_S = 0.00000222
ASSUMED_CORES = 1.0
ASSUMED_GIB = 8.0
DETECTIONS_KEYS = {"video", "sample_fps", "models", "frames"}
FRAME_KEYS = {"frame_index", "timestamp_s", "teacher", "behaviors", "poses"}
AUDIO_KEYS = {
    "video_path",
    "duration_s",
    "language",
    "topic",
    "transcriber",
    "segments",
    "speech_intervals",
    "diarization",
}
SEGMENT_KEYS = {"start_s", "end_s", "text", "avg_logprob", "no_speech_prob", "words"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--topic", default=None)
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="borra el resultado del Volume al terminar (obligatorio con videos de terceros)",
    )
    return parser.parse_args()


def required_secret(name: str) -> str:
    value = read_secret(name)
    if value is None:
        raise SystemExit(f"Falta {name} en .env")
    return value


def upload(url: str, video: Path, topic: str | None, access_code: str) -> dict:
    data = {}
    if topic:
        data["topic"] = topic
    with video.open("rb") as video_file:
        response = httpx.post(
            f"{url}/jobs",
            files={"video": (video.name, video_file, "video/mp4")},
            data=data,
            headers={"X-Access-Code": access_code},
            timeout=UPLOAD_TIMEOUT_S,
        )
    response.raise_for_status()
    return response.json()


def wait_until_done(url: str, job_id: str, started_s: float) -> dict:
    last_stage = None
    while True:
        view = httpx.get(f"{url}/jobs/{job_id}", timeout=REQUEST_TIMEOUT_S).json()
        elapsed_s = time.monotonic() - started_s
        if view["stage"] != last_stage:
            print(f"  [{elapsed_s:6.0f} s] {view['stage']}")
            last_stage = view["stage"]
        if view["status"] in ("done", "error"):
            return view
        if elapsed_s > STUCK_AFTER_S:
            raise SystemExit("Modal lleva más de 20 minutos con este trabajo: revisar (plan B).")
        time.sleep(POLL_INTERVAL_S)


def check_format(result: dict) -> list[str]:
    problems = []
    detections = result["detections"]
    audio = result["audio_raw"]
    if detections is None or not set(detections) >= DETECTIONS_KEYS:
        problems.append("detections.json sin las claves del contrato")
    elif detections["frames"] and not set(detections["frames"][0]) >= FRAME_KEYS:
        problems.append("los frames no tienen las claves del contrato")
    if audio is None or not set(audio) >= AUDIO_KEYS:
        problems.append("audio_raw.json sin las claves del contrato")
    elif audio["segments"] and not set(audio["segments"][0]) >= SEGMENT_KEYS:
        problems.append("los segmentos no tienen las claves del contrato")
    if result["heatmap_png_base64"] is None:
        problems.append("falta el heatmap")
    return problems


def check_privacy(audio_path: Path) -> tuple[list[str], int, int]:
    """Con la misma regla de roles del motor: lo que no es del docente debe estar tapado."""
    capture = load_audio_capture(audio_path)
    labeled = assign_roles(assign_speakers(capture.segments, capture.diarization))
    problems = []
    students = 0
    for segment in labeled:
        if segment.role == SpeakerRole.TEACHER:
            continue
        students += 1
        if segment.text != STUDENT_PLACEHOLDER or segment.words:
            problems.append(f"texto de alumno sin tapar en {segment.start_s:.0f} s")
    names = 0
    for segment in capture.segments:
        names += segment.text.count("[NOMBRE]")
    return problems, students, names


def estimate_cost(timings: dict, scaledown_s: float) -> float:
    """GPU desde que arranca el contenedor hasta que se apaga (incluye la espera para apagarse)."""
    gpu_s = timings.get("queued_s", 0) + timings.get("video_s", 0) + timings.get("audio_s", 0)
    gpu_s += scaledown_s
    per_s = T4_USD_PER_S + ASSUMED_CORES * CPU_CORE_USD_PER_S + ASSUMED_GIB * GIB_USD_PER_S
    return gpu_s * per_s


def main() -> None:
    args = parse_args()
    settings = read_yaml(CONFIGS_DIR / "worker.yaml")
    url = modal.Function.from_name(settings["app_name"], WEB_FUNCTION).get_web_url()
    access_code = required_secret("UPLOAD_ACCESS_CODE")
    worker_token = required_secret("WORKER_TOKEN")
    print(f"Worker: {url}")

    started_s = time.monotonic()
    created = upload(url, args.video, args.topic, access_code)
    job_id = created["job_id"]
    print(f"Subido en {time.monotonic() - started_s:.0f} s. Trabajo {job_id}")

    view = wait_until_done(url, job_id, started_s)
    if view["status"] == "error":
        raise SystemExit(f"El worker falló: {view['error']}")

    headers = {"Authorization": f"Bearer {worker_token}"}
    response = httpx.get(f"{url}/jobs/{job_id}/result", headers=headers, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    result = response.json()

    out_dir = OUTPUTS_DIR / "modal_test" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detections.json").write_text(json.dumps(result["detections"]), encoding="utf-8")
    audio_path = out_dir / "audio_raw.json"
    audio_path.write_text(json.dumps(result["audio_raw"], ensure_ascii=False), encoding="utf-8")

    problems = check_format(result)
    privacy_problems, students, names = check_privacy(audio_path)
    timings = view["timings"]
    cost = estimate_cost(timings, settings["scaledown_window_s"])

    print("\n=== Resultado ===")
    print(f"Formato del contrato: {problems or 'OK'}")
    print(f"Privacidad: {privacy_problems[:5] or 'OK'}")
    print(f"  segmentos de alumnos tapados: {students}, nombres tapados: {names}")
    print(f"Tiempos: {timings}")
    print(f"Costo estimado en Modal: US$ {cost:.3f}")
    print(f"Resultado guardado en {out_dir}")

    if args.delete_after:
        modal.Volume.from_name(settings["jobs_volume"]).remove_file(job_id, recursive=True)
        print("Resultado borrado del Volume.")


if __name__ == "__main__":
    main()
