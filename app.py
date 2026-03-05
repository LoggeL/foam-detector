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

app = FastAPI(title="Foam Detector Demo", docs_url=None, redoc_url=None)

# Restrict CORS to same-origin (cloudflare tunnel serves everything from same host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _validate_video_id(video_id: str) -> None:
    """Reject any video_id that isn't exactly a 12-char hex string."""
    if not VIDEO_ID_PATTERN.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")


def _rate_limit(request: Request) -> None:
    """Simple in-memory rate limiter per client IP."""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_limit_store[ip]
    # Prune old entries
    _rate_limit_store[ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    _rate_limit_store[ip].append(now)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

STATUS_ORDER = ["NO_FOAM", "FOAM_STARTING", "MODERATE_FOAM", "HEAVY_FOAM"]
STATUS_TO_COLOR = {
    "NO_FOAM": "green",
    "FOAM_STARTING": "yellow",
    "MODERATE_FOAM": "red",
    "HEAVY_FOAM": "red",
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


def normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status", "NO_FOAM")).strip().upper()
    if status not in STATUS_ORDER:
        status = "NO_FOAM"

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
        "color": STATUS_TO_COLOR[status],
    }


def _parse_json_from_text(text: str) -> Any:
    """Extract JSON from model response text, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON from: {text[:200]}")


def analyze_video_with_gemini(video_path: Path, api_key: str) -> list[dict[str, Any]]:
    """Upload video to Google Files API and analyze with Gemini in one call."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    video_file = genai.upload_file(path=str(video_path))

    # Wait for file processing
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed with state: {video_file.state.name}")

    prompt = (
        "You are analyzing an industrial process video for foam detection. "
        "Watch the ENTIRE video and produce a timeline analysis, sampling every 2 seconds from the start. "
        "For each 2-second interval, classify the foam level.\n\n"
        "Return ONLY a JSON array (no markdown, no explanation) with one entry per 2-second interval:\n"
        "[\n"
        '  {"timestamp": "00:00", "timestamp_seconds": 0, "status": "NO_FOAM", "confidence": 0.95, "description": "..."}, \n'
        '  {"timestamp": "00:02", "timestamp_seconds": 2, "status": "FOAM_STARTING", "confidence": 0.85, "description": "..."}, \n'
        "  ...\n"
        "]\n\n"
        "Status must be one of: NO_FOAM, FOAM_STARTING, MODERATE_FOAM, HEAVY_FOAM\n"
        "confidence is 0.0-1.0\n"
        "description should briefly describe what you see at that moment.\n"
        "Cover the entire video duration."
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([video_file, prompt])

    parsed = _parse_json_from_text(response.text)
    if not isinstance(parsed, list):
        raise ValueError("Gemini did not return a JSON array")

    # Clean up uploaded file
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    results = []
    for entry in parsed:
        normalized = normalize_result(entry)
        results.append({
            "timestamp_seconds": int(entry.get("timestamp_seconds", 0)),
            "timestamp": str(entry.get("timestamp", format_time(entry.get("timestamp_seconds", 0)))),
            **normalized,
        })

    return results


# --- Fallback: frame-by-frame via OpenRouter ---

def extract_frames(video_path: Path, frame_dir: Path, interval_seconds: int = 2) -> list[dict[str, Any]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_dir.glob("*.jpg"):
        frame.unlink()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds}",
        str(frame_dir / "frame_%06d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    extracted: list[dict[str, Any]] = []
    for idx, frame_path in enumerate(frames):
        timestamp = idx * interval_seconds
        extracted.append(
            {
                "frame_path": str(frame_path),
                "timestamp_seconds": timestamp,
                "timestamp": format_time(timestamp),
            }
        )
    if not extracted:
        raise RuntimeError("No frames extracted. Ensure the uploaded file is a valid video.")
    return extracted


def analyze_frame_with_openrouter(client: Any, frame_path: str) -> dict[str, Any]:
    from openai import OpenAI

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
        parsed = {
            "status": "NO_FOAM",
            "confidence": 0.4,
            "description": f"Model returned unstructured output: {text_output[:180]}",
        }

    return normalize_result(parsed)


# --- Demo mode ---

def generate_demo_analysis(timestamp_seconds: int, max_seconds: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + timestamp_seconds)
    progress = 0.0 if max_seconds <= 0 else timestamp_seconds / max_seconds

    # Logistic-like buildup with slight noise for a realistic trend.
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

    return normalize_result(
        {
            "status": status,
            "confidence": confidence,
            "description": description,
        }
    )


def stable_seed_from_video_id(video_id: str) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 30.0  # fallback default
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 30.0


def run_analysis(video_id: str, requested_demo_mode: bool) -> None:
    try:
        video_path = find_video_path(video_id)

        result_payload = load_result(video_id)
        result_payload["status"] = "analyzing"
        result_payload["progress"] = 0
        result_payload["results"] = []
        result_payload["error"] = None
        save_result(video_id, result_payload)

        google_key = os.getenv("GOOGLE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        use_demo = requested_demo_mode or (not google_key and not openrouter_key)
        use_native_gemini = bool(google_key) and not use_demo

        result_payload["demo_mode"] = use_demo
        save_result(video_id, result_payload)

        if use_demo:
            # Demo mode: generate fake timeline
            duration = _get_video_duration(video_path)
            max_seconds = int(duration)
            seed = stable_seed_from_video_id(video_id)
            analyzed: list[dict[str, Any]] = []
            timestamps = list(range(0, max_seconds + 1, 2))
            if not timestamps:
                timestamps = [0]
            total = len(timestamps)

            for i, ts in enumerate(timestamps, start=1):
                model_result = generate_demo_analysis(ts, max_seconds, seed)
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
            # Native Gemini video analysis: one API call, no frame extraction
            result_payload["progress"] = 10
            save_result(video_id, result_payload)

            analyzed = analyze_video_with_gemini(video_path, google_key)

            result_payload["progress"] = 95
            result_payload["results"] = analyzed
            save_result(video_id, result_payload)

        else:
            # Fallback: frame-by-frame via OpenRouter
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
                    model_result = analyze_frame_with_openrouter(client, frame["frame_path"])
                except Exception as exc:
                    model_result = normalize_result(
                        {
                            "status": "NO_FOAM",
                            "confidence": 0.2,
                            "description": f"Frame analysis failed, fallback used: {exc}",
                        }
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
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    video_id = uuid.uuid4().hex[:12]
    video_path = UPLOAD_DIR / f"{video_id}{ext}"

    # Stream with size limit
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
        "error": None,
    }
    save_result(video_id, payload)
    return payload


@app.post("/analyze")
def analyze(request: Request, req: AnalyzeRequest) -> dict[str, Any]:
    _rate_limit(request)
    _validate_video_id(req.video_id)
    _ = find_video_path(req.video_id)

    with THREAD_LOCK:
        existing = ANALYSIS_THREADS.get(req.video_id)
        if existing and existing.is_alive():
            return {
                "video_id": req.video_id,
                "status": "already_running",
                "message": "Analysis is already in progress",
            }

        thread = threading.Thread(
            target=run_analysis,
            args=(req.video_id, req.demo_mode),
            daemon=True,
            name=f"analysis-{req.video_id}",
        )
        ANALYSIS_THREADS[req.video_id] = thread
        thread.start()

    return {
        "video_id": req.video_id,
        "status": "started",
        "demo_mode_requested": req.demo_mode,
    }


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


@app.post("/analyze-frame")
def analyze_live_frame(request: Request, req: LiveFrameRequest) -> dict[str, Any]:
    _rate_limit(request)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        # Demo fallback
        import random as _random
        statuses = ["NO_FOAM", "FOAM_STARTING", "MODERATE_FOAM", "HEAVY_FOAM"]
        status = _random.choice(statuses)
        return {
            "status": status,
            "confidence": round(_random.uniform(0.6, 0.95), 2),
            "description": f"Demo mode: {status.replace('_', ' ').lower()} detected.",
            "color": STATUS_TO_COLOR[status],
        }

    # Save frame to temp file
    tmp_path = f"/tmp/live_frame_{uuid.uuid4().hex}.jpg"
    try:
        image_bytes = base64.b64decode(req.image)
        Path(tmp_path).write_bytes(image_bytes)
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        result = analyze_frame_with_openrouter(client, tmp_path)
        return result
    except Exception as exc:
        return {
            "status": "NO_FOAM",
            "confidence": 0.2,
            "description": f"Analysis error: {exc}",
            "color": STATUS_TO_COLOR["NO_FOAM"],
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
