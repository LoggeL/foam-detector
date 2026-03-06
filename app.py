import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

# --- Security constants ---
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
VIDEO_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20  # requests per window per IP
_rate_limit_store: dict[str, list[float]] = defaultdict(list)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRAME_DIR = DATA_DIR / "frames"
RESULT_DIR = DATA_DIR / "results"
STATIC_DIR = BASE_DIR / "static"

for path in [UPLOAD_DIR, FRAME_DIR, RESULT_DIR, STATIC_DIR]:
    path.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BASF Vision Detection Demo", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_PATTERN.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    _rate_limit_store[ip].append(now)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Foam mode ---
FOAM_STATUS_ORDER = ["NO_FOAM", "FOAM_STARTING", "MODERATE_FOAM", "HEAVY_FOAM"]
FOAM_STATUS_TO_COLOR = {
    "NO_FOAM": "green",
    "FOAM_STARTING": "yellow",
    "MODERATE_FOAM": "red",
    "HEAVY_FOAM": "red",
}

# --- Leak mode ---
LEAK_STATUS_ORDER = ["NO_LEAK", "LEAK_STARTING", "MODERATE_LEAK", "HEAVY_LEAK"]
LEAK_STATUS_TO_COLOR = {
    "NO_LEAK": "green",
    "LEAK_STARTING": "yellow",
    "MODERATE_LEAK": "orange",
    "HEAVY_LEAK": "red",
}

ANALYSIS_THREADS: dict[str, threading.Thread] = {}
THREAD_LOCK = threading.Lock()


def save_result(video_id: str, payload: dict[str, Any]) -> None:
    result_file = RESULT_DIR / f"{video_id}.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_result(video_id: str) -> dict[str, Any]:
    result_file = RESULT_DIR / f"{video_id}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    return json.loads(result_file.read_text(encoding="utf-8"))


def format_time(seconds: float) -> str:
    sec = int(seconds)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def find_video_path(video_id: str) -> Path:
    candidates = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Video not found")
    return candidates[0]


def normalize_result(raw: dict[str, Any], detection_mode: str = "foam") -> dict[str, Any]:
    if detection_mode == "leak":
        valid_statuses = LEAK_STATUS_ORDER
        color_map = LEAK_STATUS_TO_COLOR
        default_status = "NO_LEAK"
    else:
        valid_statuses = FOAM_STATUS_ORDER
        color_map = FOAM_STATUS_TO_COLOR
        default_status = "NO_FOAM"

    status = str(raw.get("status", default_status)).strip().upper()
    if status not in valid_statuses:
        status = default_status

    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    description = str(raw.get("description", "No description provided.")).strip()
    if not description:
        description = "No description provided."

    return {
        "status": status,
        "confidence": round(confidence, 3),
        "description": description,
        "color": color_map[status],
    }


def _parse_json_from_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON from: {text[:200]}")


def analyze_video_with_gemini(video_path: Path, api_key: str, detection_mode: str = "foam") -> list[dict[str, Any]]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    video_file = genai.upload_file(path=str(video_path))

    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed with state: {video_file.state.name}")

    if detection_mode == "leak":
        prompt = (
            "You are analyzing an industrial process video for LIQUID LEAKAGE detection. "
            "Watch the ENTIRE video and produce a timeline analysis, sampling every 2 seconds from the start. "
            "For each 2-second interval, classify the leakage severity.\n\n"
            "Return ONLY a JSON array (no markdown, no explanation):\n"
            "[\n"
            '  {"timestamp": "00:00", "timestamp_seconds": 0, "status": "NO_LEAK", "confidence": 0.95, "description": "..."}, \n'
            "  ...\n"
            "]\n\n"
            "Status must be one of: NO_LEAK, LEAK_STARTING, MODERATE_LEAK, HEAVY_LEAK\n"
            "confidence is 0.0-1.0\n"
            "description: briefly describe what you see (pipe state, water/liquid, drips, puddles, etc).\n"
            "Cover the entire video duration."
        )
    else:
        prompt = (
            "You are analyzing an industrial process video for foam detection. "
            "Watch the ENTIRE video and produce a timeline analysis, sampling every 2 seconds from the start. "
            "For each 2-second interval, classify the foam level.\n\n"
            "Return ONLY a JSON array (no markdown, no explanation):\n"
            "[\n"
            '  {"timestamp": "00:00", "timestamp_seconds": 0, "status": "NO_FOAM", "confidence": 0.95, "description": "..."}, \n'
            "  ...\n"
            "]\n\n"
            "Status must be one of: NO_FOAM, FOAM_STARTING, MODERATE_FOAM, HEAVY_FOAM\n"
            "confidence is 0.0-1.0\n"
            "description: briefly describe what you see.\n"
            "Cover the entire video duration."
        )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([video_file, prompt])

    parsed = _parse_json_from_text(response.text)
    if not isinstance(parsed, list):
        raise ValueError("Gemini did not return a JSON array")

    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    results = []
    for entry in parsed:
        normalized = normalize_result(entry, detection_mode)
        results.append({
            "timestamp_seconds": int(entry.get("timestamp_seconds", 0)),
            "timestamp": str(entry.get("timestamp", format_time(entry.get("timestamp_seconds", 0)))),
            **normalized,
        })

    return results


def extract_frames(video_path: Path, frame_dir: Path, interval_seconds: int = 2) -> list[dict[str, Any]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_dir.glob("*.jpg"):
        frame.unlink()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
        str(frame_dir / "frame_%06d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    extracted: list[dict[str, Any]] = []
    for idx, frame_path in enumerate(frames):
        timestamp = idx * interval_seconds
        extracted.append({
            "frame_path": str(frame_path),
            "timestamp_seconds": timestamp,
            "timestamp": format_time(timestamp),
        })
    if not extracted:
        raise RuntimeError("No frames extracted. Ensure the uploaded file is a valid video.")
    return extracted


def analyze_frame_with_openrouter(client: Any, frame_path: str, detection_mode: str = "foam") -> dict[str, Any]:
    if detection_mode == "leak":
        prompt = (
            "Analyze this frame from an industrial or household video for LIQUID LEAKAGE detection. "
            "Is there a visible leak, drip, water flow, or puddle? "
            "Rate: NO_LEAK, LEAK_STARTING, MODERATE_LEAK, HEAVY_LEAK. "
            "Respond as JSON with fields: status, confidence (0-1), description."
        )
    else:
        prompt = (
            "Analyze this frame from an industrial process video. Is there foam visible? "
            "Rate: NO_FOAM, FOAM_STARTING, MODERATE_FOAM, HEAVY_FOAM. "
            "Respond as JSON with fields: status, confidence (0-1), description."
        )

    image_bytes = Path(frame_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
    )

    text_output = response.choices[0].message.content.strip()
    try:
        parsed = _parse_json_from_text(text_output)
    except ValueError:
        default = "NO_LEAK" if detection_mode == "leak" else "NO_FOAM"
        parsed = {
            "status": default,
            "confidence": 0.4,
            "description": f"Model returned unstructured output: {text_output[:180]}",
        }

    return normalize_result(parsed, detection_mode)


# --- Demo mode generators ---

def generate_demo_foam_analysis(timestamp_seconds: int, max_seconds: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + timestamp_seconds)
    progress = 0.0 if max_seconds <= 0 else timestamp_seconds / max_seconds

    foam_intensity = 1 / (1 + math.exp(-10 * (progress - 0.55)))
    foam_intensity += rng.uniform(-0.08, 0.08)
    foam_intensity = max(0.0, min(1.0, foam_intensity))

    if foam_intensity < 0.22:
        status = "NO_FOAM"
        confidence = rng.uniform(0.86, 0.98)
        description = "Liquid surface appears calm with negligible bubble activity."
    elif foam_intensity < 0.46:
        status = "FOAM_STARTING"
        confidence = rng.uniform(0.75, 0.94)
        description = "Early foam islands forming near agitation zones."
    elif foam_intensity < 0.74:
        status = "MODERATE_FOAM"
        confidence = rng.uniform(0.73, 0.92)
        description = "Foam layer is expanding and covering a visible portion of the vessel."
    else:
        status = "HEAVY_FOAM"
        confidence = rng.uniform(0.78, 0.96)
        description = "Dense foam blanket observed, indicating high buildup intensity."

    return normalize_result({"status": status, "confidence": confidence, "description": description}, "foam")


def generate_demo_leak_analysis(timestamp_seconds: int, max_seconds: int, seed: int) -> dict[str, Any]:
    """Simulates a faucet/valve that starts closed, opens partway through, leaks increasingly."""
    rng = random.Random(seed + timestamp_seconds + 9999)
    progress = 0.0 if max_seconds <= 0 else timestamp_seconds / max_seconds

    # Leak starts at ~40% of video, builds up with noise
    leak_intensity = 1 / (1 + math.exp(-12 * (progress - 0.42)))
    leak_intensity += rng.uniform(-0.06, 0.06)
    leak_intensity = max(0.0, min(1.0, leak_intensity))

    if leak_intensity < 0.18:
        status = "NO_LEAK"
        confidence = rng.uniform(0.88, 0.98)
        description = "Valve/pipe appears dry, no liquid flow detected."
    elif leak_intensity < 0.40:
        status = "LEAK_STARTING"
        confidence = rng.uniform(0.74, 0.91)
        description = "Slight moisture or initial drip visible near valve outlet."
    elif leak_intensity < 0.70:
        status = "MODERATE_LEAK"
        confidence = rng.uniform(0.72, 0.90)
        description = "Steady drip or thin stream observed, puddle beginning to form."
    else:
        status = "HEAVY_LEAK"
        confidence = rng.uniform(0.80, 0.97)
        description = "Significant water flow visible, immediate intervention recommended."

    return normalize_result({"status": status, "confidence": confidence, "description": description}, "leak")


def stable_seed_from_video_id(video_id: str) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 30.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 30.0


def run_analysis(video_id: str, requested_demo_mode: bool, detection_mode: str = "foam") -> None:
    try:
        video_path = find_video_path(video_id)

        result_payload = load_result(video_id)
        result_payload["status"] = "analyzing"
        result_payload["progress"] = 0
        result_payload["results"] = []
        result_payload["error"] = None
        result_payload["detection_mode"] = detection_mode
        save_result(video_id, result_payload)

        google_key = os.getenv("GOOGLE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        use_demo = requested_demo_mode or (not google_key and not openrouter_key)
        use_native_gemini = bool(google_key) and not use_demo

        result_payload["demo_mode"] = use_demo
        save_result(video_id, result_payload)

        if use_demo:
            duration = _get_video_duration(video_path)
            max_seconds = int(duration)
            seed = stable_seed_from_video_id(video_id)
            analyzed: list[dict[str, Any]] = []
            timestamps = list(range(0, max_seconds + 1, 2))
            if not timestamps:
                timestamps = [0]
            total = len(timestamps)

            generator = generate_demo_leak_analysis if detection_mode == "leak" else generate_demo_foam_analysis

            for i, ts in enumerate(timestamps, start=1):
                model_result = generator(ts, max_seconds, seed)
                row = {
                    "timestamp_seconds": ts,
                    "timestamp": format_time(ts),
                    **model_result,
                }
                analyzed.append(row)

                result_payload["status"] = "analyzing"
                result_payload["progress"] = int((i / total) * 100)
                result_payload["results"] = analyzed
                save_result(video_id, result_payload)

        elif use_native_gemini:
            result_payload["progress"] = 10
            save_result(video_id, result_payload)

            analyzed = analyze_video_with_gemini(video_path, google_key, detection_mode)

            result_payload["progress"] = 95
            result_payload["results"] = analyzed
            save_result(video_id, result_payload)

        else:
            from openai import OpenAI

            frame_dir = FRAME_DIR / video_id
            frame_entries = extract_frames(video_path, frame_dir)

            client = OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            )
            analyzed = []
            total = len(frame_entries)

            for i, frame in enumerate(frame_entries, start=1):
                try:
                    model_result = analyze_frame_with_openrouter(client, frame["frame_path"], detection_mode)
                except Exception as exc:
                    default = "NO_LEAK" if detection_mode == "leak" else "NO_FOAM"
                    model_result = normalize_result(
                        {"status": default, "confidence": 0.2, "description": f"Frame analysis failed: {exc}"},
                        detection_mode,
                    )

                row = {
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "timestamp": frame["timestamp"],
                    **model_result,
                }
                analyzed.append(row)

                result_payload["status"] = "analyzing"
                result_payload["progress"] = int((i / total) * 100)
                result_payload["results"] = analyzed
                save_result(video_id, result_payload)

        result_payload["status"] = "completed"
        result_payload["progress"] = 100
        result_payload["results"] = analyzed
        save_result(video_id, result_payload)

    except Exception as exc:  # noqa: BLE001
        fallback_payload = {
            "video_id": video_id,
            "video_url": f"/video/{video_id}",
            "status": "failed",
            "progress": 0,
            "results": [],
            "error": str(exc),
        }
        save_result(video_id, fallback_payload)
    finally:
        with THREAD_LOCK:
            ANALYSIS_THREADS.pop(video_id, None)


class AnalyzeRequest(BaseModel):
    video_id: str
    demo_mode: bool = False
    detection_mode: str = "foam"  # "foam" | "leak"


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    _rate_limit(request)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower() or ".mp4"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed.")

    video_id = uuid.uuid4().hex[:12]
    video_path = UPLOAD_DIR / f"{video_id}{ext}"

    total = 0
    with video_path.open("wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                video_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large. Max 100 MB.")
            out_file.write(chunk)

    payload = {
        "video_id": video_id,
        "video_url": f"/video/{video_id}",
        "filename": file.filename,
        "status": "uploaded",
        "progress": 0,
        "results": [],
        "demo_mode": False,
        "detection_mode": "foam",
        "error": None,
    }
    save_result(video_id, payload)
    return payload


@app.post("/analyze")
def analyze(request: Request, req: AnalyzeRequest) -> dict[str, Any]:
    _rate_limit(request)
    _validate_video_id(req.video_id)
    _ = find_video_path(req.video_id)

    detection_mode = req.detection_mode if req.detection_mode in ("foam", "leak") else "foam"

    with THREAD_LOCK:
        existing = ANALYSIS_THREADS.get(req.video_id)
        if existing and existing.is_alive():
            return {"video_id": req.video_id, "status": "already_running", "message": "Analysis is already in progress"}

        thread = threading.Thread(
            target=run_analysis,
            args=(req.video_id, req.demo_mode, detection_mode),
            daemon=True,
            name=f"analysis-{req.video_id}",
        )
        ANALYSIS_THREADS[req.video_id] = thread
        thread.start()

    return {"video_id": req.video_id, "status": "started", "demo_mode_requested": req.demo_mode, "detection_mode": detection_mode}


@app.get("/results/{video_id}")
def get_results(video_id: str, request: Request) -> dict[str, Any]:
    _rate_limit(request)
    _validate_video_id(video_id)
    return load_result(video_id)


@app.get("/video/{video_id}")
def get_video(video_id: str, request: Request) -> FileResponse:
    _rate_limit(request)
    _validate_video_id(video_id)
    video_path = find_video_path(video_id)
    return FileResponse(video_path)


class LiveFrameRequest(BaseModel):
    image: str  # base64 JPEG
    detection_mode: str = "foam"


@app.post("/analyze-frame")
def analyze_live_frame(request: Request, req: LiveFrameRequest) -> dict[str, Any]:
    _rate_limit(request)
    detection_mode = req.detection_mode if req.detection_mode in ("foam", "leak") else "foam"
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not openrouter_key:
        import random as _random
        if detection_mode == "leak":
            statuses = ["NO_LEAK", "LEAK_STARTING", "MODERATE_LEAK", "HEAVY_LEAK"]
            color_map = LEAK_STATUS_TO_COLOR
        else:
            statuses = ["NO_FOAM", "FOAM_STARTING", "MODERATE_FOAM", "HEAVY_FOAM"]
            color_map = FOAM_STATUS_TO_COLOR
        status = _random.choice(statuses)
        label = status.replace("_", " ").lower()
        return {
            "status": status,
            "confidence": round(_random.uniform(0.6, 0.95), 2),
            "description": f"Demo mode: {label} detected.",
            "color": color_map[status],
        }

    tmp_path = f"/tmp/live_frame_{uuid.uuid4().hex}.jpg"
    try:
        image_bytes = base64.b64decode(req.image)
        Path(tmp_path).write_bytes(image_bytes)
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        result = analyze_frame_with_openrouter(client, tmp_path, detection_mode)
        return result
    except Exception as exc:
        default = "NO_LEAK" if detection_mode == "leak" else "NO_FOAM"
        color_map = LEAK_STATUS_TO_COLOR if detection_mode == "leak" else FOAM_STATUS_TO_COLOR
        return {"status": default, "confidence": 0.2, "description": f"Analysis error: {exc}", "color": color_map[default]}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
