# ARIA — Automated Requirements Intelligence Agent

AI-powered system that converts meeting recordings or silent screen recordings into professional **Process Definition Documents (PDD)** and **Business Requirements Documents (BRD)** for RPA projects.

---

## Architecture

ARIA consists of three services:

```
┌─────────────────────────────────────────────────────────┐
│  Frontend  (Streamlit — port 8501)                      │
│  Thin UI: upload video, configure options, download PDD │
└──────────────────────┬──────────────────────────────────┘
                       │ REST (PDD_BACKEND_URL)
┌──────────────────────▼──────────────────────────────────┐
│  Backend  (FastAPI — port 8001)                         │
│  Business logic: pipelines, LLM orchestration,          │
│  document generation, job management                    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  VectorDB Service  (mounted at /vectordb)        │   │
│  │  OR runs standalone on port 8000                 │   │
│  │  Frame indexing: CLIP + Pinecone + MongoDB        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

The VectorDB service can be run in **two modes**:
- **Mounted (default):** Embedded inside the backend process at `http://localhost:8001/vectordb` — only one service to run.
- **Standalone:** Run as a separate process on port 8000 and set `VECTORDB_URL=http://localhost:8000` in `backend/.env`.

---

## Project Structure

```
aria/
│
├── frontend/                     Streamlit web UI
│   ├── streamlit_app.py          UI entry point
│   └── pyproject.toml
│
├── backend/                      FastAPI — business logic service (port 8001)
│   ├── main.py                   App entry point; mounts vectordb_service at /vectordb
│   ├── pyproject.toml
│   ├── .env                      API keys and config
│   │
│   ├── api/
│   │   ├── models.py             Pydantic request/response models
│   │   └── routes/
│   │       ├── pipeline.py       POST /api/pipeline/process, GET /api/pipeline/status/{job_id}
│   │       ├── documents.py      GET /api/documents/, GET /api/documents/download/{filename}
│   │       └── health.py         GET /health
│   │
│   ├── core/
│   │   ├── config.py             Unified config (Gemini, Whisper, frames, annotation, paths)
│   │   ├── gemini_client.py      Gemini API wrapper with rate limiting
│   │   ├── token_tracker.py      LLM token usage tracking → CSV
│   │   ├── utils.py              Auth screen detection, delta detection
│   │   └── vectordb_client.py    HTTP client for VectorDB service
│   │
│   ├── pipeline/
│   │   ├── audio_pipeline.py     Meeting recording orchestrator (audio → PDD)
│   │   ├── video_pipeline.py     Silent screen recording orchestrator (video → PDD)
│   │   └── common.py             Shared: document assembly, flowchart, file saving
│   │
│   ├── llm_tasks/                All Gemini prompt implementations
│   │   ├── meeting_compact.py    Consolidated audio PDD bundle + DOT generation
│   │   ├── document_sections.py  Document section generation
│   │   ├── process_steps.py      Process step extraction
│   │   ├── step_synthesizer.py   Step decomposition/refinement
│   │   ├── entity_extraction.py  Entity/actor extraction
│   │   ├── requirements.py       Requirements generation
│   │   ├── flowchart_dot.py      Graphviz DOT generation
│   │   ├── vision_describer.py   Frame visual descriptions
│   │   ├── timestamps.py         Transcript timestamp alignment
│   │   └── system_prompts.py     Base system prompts
│   │
│   ├── audio/
│   │   ├── video_to_audio.py     FFmpeg video → audio extraction
│   │   ├── transcriber.py        OpenAI Whisper transcription
│   │   └── frame_extractor.py    Audio-aligned frame extraction
│   │
│   ├── video/
│   │   ├── scene_detector.py     SSIM-based scene change detection
│   │   ├── smart_sampler.py      Key frame selection
│   │   ├── ocr_engine.py         Tesseract OCR
│   │   ├── frame_annotator.py    Bounding box annotation on frames
│   │   ├── frame_extractor.py    Raw frame extraction
│   │   ├── frame_matcher.py      Frame comparison
│   │   └── change_detector.py    UI change delta detection
│   │
│   ├── document/
│   │   ├── pdd_generator.py      python-docx PDD/BRD document builder
│   │   └── flowchart_renderer.py Graphviz DOT → PNG/SVG rendering
│   │
│   └── outputs/                  Generated .docx, .dot, .csv files
│
└── vectordb_service/             FastAPI — frame indexing micro-service (port 8000)
    ├── main.py                   App entry point
    ├── config.py                 Pydantic settings (Pinecone, MongoDB, CLIP, Gemini)
    ├── pyproject.toml
    ├── .env
    │
    ├── api/routes/
    │   ├── video.py              POST /api/videos/upload, DELETE /api/videos/{video_id}
    │   ├── query.py              POST /api/frames/query, POST /api/frames/query/locate
    │   └── frames.py             GET /api/frames/image/{video_id}/{filename}
    │
    ├── services/                 Video processing & query pipeline logic
    ├── infrastructure/           CLIP embedding, Pinecone, MongoDB GridFS, OmniParser, Gemini
    └── domain/models.py          Pydantic domain models
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | [python.org](https://www.python.org/downloads/) |
| uv | latest | `pip install uv` |
| FFmpeg | any recent | `winget install FFmpeg.FFmpeg` or `choco install ffmpeg` |
| Graphviz | any recent | `winget install Graphviz.Graphviz` or `choco install graphviz` |
| Tesseract OCR | 5.x | `winget install UB-Mannheim.TesseractOCR` or `choco install tesseract` |
| MongoDB | 6+ | [mongodb.com](https://www.mongodb.com/try/download/community) |

---

## Environment Variables

**`backend/.env`**
```env
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional — Gemini model selection (defaults shown)
GEMINI_TEXT_MODEL=gemini-2.5-flash
GEMINI_VISION_MODEL=gemini-2.5-flash

# Optional — Rate limiting for free tier
GEMINI_RPM=8
GEMINI_TPM=250000
GEMINI_RPD=1000

# Optional — VectorDB URL (only needed when running vectordb_service standalone)
# Default points to the mounted sub-app inside backend
VECTORDB_URL=http://localhost:8001/vectordb
```

**`vectordb_service/.env`** *(only needed in standalone mode)*
```env
PINECONE_API_KEY=your-pinecone-api-key
REPLICATE_API_TOKEN=your-replicate-token
GEMINI_API_KEY=your-gemini-api-key
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=pdd_vectordb
```

---

## Installation & Running

All services use [uv](https://docs.astral.sh/uv/) for dependency management.

### Option A — Combined (Recommended)

The VectorDB service mounts inside the backend at `/vectordb`. Only two terminals needed.

```bash
# Terminal 1 — Backend (includes VectorDB at /vectordb)
cd backend
uv sync
uv run uvicorn main:app --port 8001 --reload

# Terminal 2 — Frontend
cd frontend
uv sync
uv run streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

> **API Docs:** [http://localhost:8001/docs](http://localhost:8001/docs)

---

### Option B — Separate Services

Run vectordb_service as a standalone process on its own port:

1. Set `VECTORDB_URL=http://localhost:8000` in `backend/.env`

```bash
# Terminal 1 — VectorDB Service
cd vectordb_service
uv sync
uv run uvicorn main:app --port 8000 --reload

# Terminal 2 — Backend
cd backend
uv sync
uv run uvicorn main:app --port 8001 --reload

# Terminal 3 — Frontend
cd frontend
uv sync
uv run streamlit run streamlit_app.py
```

> **VectorDB API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
> **Backend API Docs:** [http://localhost:8001/docs](http://localhost:8001/docs)

---

## API Reference

### Backend (port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — Gemini status |
| `POST` | `/api/pipeline/process` | Submit a PDD generation job |
| `GET` | `/api/pipeline/status/{job_id}` | Poll job status and result |
| `GET` | `/api/documents/` | List generated documents |
| `GET` | `/api/documents/download/{filename}` | Download a generated `.docx` |

### VectorDB Service (port 8000 or at `/vectordb` in combined mode)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/videos/upload` | Upload video, extract frames, index in Pinecone |
| `DELETE` | `/api/videos/{video_id}` | Delete all frames and vectors for a video |
| `POST` | `/api/frames/query` | Search frames by natural-language text (CLIP) |
| `POST` | `/api/frames/query/locate` | Search + Gemini frame pick + OmniParser + highlight |
| `GET` | `/api/frames/image/{video_id}/{filename}` | Serve frame image from MongoDB |

---

## Pipelines

### Meeting Recording (Audio + Video)

```
Video file
  → FFmpeg — extract audio track
  → Whisper — transcribe audio to text  (or use provided transcript)
  → Upload video to VectorDB — index frames (CLIP + Pinecone + MongoDB)
  → Gemini LLM (batch):
      • Document sections (title, scope, roles, systems)
      • Process steps + requirements
      • Step refinement / sub-step decomposition
      • Graphviz DOT flowchart
  → VectorDB query/locate — annotated frame screenshots per step
  → python-docx — assemble PDD/BRD document
  → Output: .docx + .dot + token_usage.csv
```

### Silent Screen Recording (Video Only)

```
Video file
  → SSIM scene detection — identify scene changes
  → Smart frame sampler — select key frames (start/mid/end per scene)
  → Tesseract OCR — extract text from frames (parallel)
  → Auth screen detection — filter login/lock screens
  → UI delta detection — identify what changed between frames
  → Upload video to VectorDB — index frames (CLIP + Pinecone + MongoDB)
  → Gemini Vision (parallel):
      • Visual descriptions per frame
      • Screen-by-screen step synthesis
      • High-level process step inference
      • Document sections and requirements
      • Graphviz DOT flowchart
  → Frame annotation — bounding boxes on key UI elements
  → python-docx — assemble PDD/BRD document
  → Output: .docx + .dot + token_usage.csv
```

### VectorDB: Query & Locate

```
Text query
  → CLIP text embedding
  → Pinecone top-k vector search
  → Gemini — selects best matching frame
  → OmniParser (Replicate) — detects UI elements + bounding boxes
  → Gemini — identifies target element from OmniParser output
  → Annotated frame image saved to MongoDB
  → Returned to pipeline for embedding in document
```

---

## Configuration Reference

Key settings in `backend/.env` / `core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required. Google Gemini API key |
| `GEMINI_TEXT_MODEL` | `gemini-3-flash-preview` | Text model |
| `GEMINI_VISION_MODEL` | `gemini-3-flash-preview` | Vision model |
| `GEMINI_RPM` | `8` | Requests per minute (free tier limit) |
| `VECTORDB_URL` | `http://localhost:8001/vectordb` | VectorDB service URL |

Pipeline options exposed via the UI / API:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `document_type` | `PDD` | `PDD` or `BRD` |
| `whisper_model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `ssim_threshold` | `0.92` | Scene change sensitivity (higher = fewer frames) |
| `max_frames` | `60` | Maximum key frames to extract |
| `enable_micro_frames` | `true` | Extract additional micro-frames between key frames |
| `annotate` | `true` | Draw bounding boxes on screenshot frames |
| `pii_redaction` | `true` | Redact personally identifiable information |

---

## Key Dependencies

| Service | Key Packages |
|---------|-------------|
| **Backend** | FastAPI, google-genai, openai-whisper, python-docx, graphviz, opencv-python, pytesseract, scenedetect, torch, transformers, pinecone, pymongo, pydub, cairosvg |
| **VectorDB Service** | FastAPI, torch, transformers (CLIP), pinecone, pymongo, replicate, google-genai, scenedetect, opencv-python |
| **Frontend** | Streamlit, requests |

See each service's `pyproject.toml` for the full pinned dependency list.
