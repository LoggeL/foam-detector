# Foam Detection Web App (BASF Demo)

Presentation-ready web app for video-based foam detection using FastAPI + GPT-4o vision (with offline demo mode).

## Features
- Dark industrial single-page UI
- Drag-and-drop video upload + file picker
- Video playback with interactive foam timeline
- Colored timeline bar:
  - Green = `NO_FOAM`
  - Yellow = `FOAM_STARTING`
  - Red = `MODERATE_FOAM` / `HEAVY_FOAM`
- Analysis table with timestamp, status, confidence, and description
- Clickable timeline to seek video
- Live progress updates during analysis
- Demo mode toggle for realistic synthetic foam progression without API key

## Project Structure
- `app.py` - FastAPI backend
- `static/index.html` - frontend UI
- `requirements.txt` - Python dependencies
- `run.sh` - start script

## Requirements
- Python 3.10+
- `ffmpeg` available in PATH

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
./run.sh
```

Open `http://localhost:8000`.

## OpenAI Setup (optional)
Set API key for real frame-by-frame LLM vision analysis:
```bash
export OPENAI_API_KEY="your_key_here"
```

If no API key is set, or if Demo Mode is checked in the UI, the app produces realistic synthetic foam buildup data.

## API Endpoints
- `POST /upload` - upload video file
- `POST /analyze` - trigger async analysis job (`video_id`, `demo_mode`)
- `GET /results/{video_id}` - current/complete analysis results JSON
- `GET /video/{video_id}` - serves uploaded video

## Sample Video
Attempted download command:
```bash
curl -L "https://www.pexels.com/video/855806/download/" -o static/sample_foam_video.mp4
```

If this file is missing (network restrictions or source issue), upload your own sample video in the UI.
