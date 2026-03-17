# PDD Agent — Video to Process Definition Document

AI-powered system that converts meeting recordings or silent screen recordings into
professional **Process Definition Documents (PDD)** / **Business Requirements Documents (BRD)**
for RPA projects.

## Architecture

```
frontend/          Streamlit UI (thin HTTP client)
    ↓ REST
backend/           FastAPI — business logic, LLM orchestration, document generation
    ↓ REST
vectordb_service/  FastAPI — video frame indexing (CLIP + Pinecone) & MongoDB storage
```

## Project Structure

```
├── frontend/                 Streamlit web UI
│   ├── streamlit_app.py
│   ├── .streamlit/
│   └── requirements.txt
│
├── backend/                  FastAPI business logic service
│   ├── main.py               FastAPI entry point (port 8001)
│   ├── .env                  GEMINI_API_KEY, etc.
│   ├── requirements.txt
│   ├── api/                  REST endpoints
│   │   ├── models.py         Pydantic request/response models
│   │   └── routes/
│   │       ├── pipeline.py   Submit & poll PDD generation jobs
│   │       ├── documents.py  List & download generated docs
│   │       └── health.py     Health check
│   ├── core/                 Config, Gemini client, token tracking, utilities
│   ├── pipeline/             Audio & Video pipeline orchestrators
│   ├── llm_tasks/            LLM prompt implementations
│   ├── audio/                FFmpeg + Whisper transcription
│   ├── document/             python-docx PDD builder, Graphviz flowcharts
│   ├── video/                SSIM scene detection, OCR, frame annotation
│   └── outputs/              Generated .docx / .dot / .csv files
│
└── vectordb_service/         FastAPI frame search micro-service
    ├── main.py               FastAPI entry point (port 8000)
    ├── .env
    ├── pyproject.toml
    ├── api/                  Upload, query, frame serving endpoints
    ├── services/             Video processing & query pipelines
    ├── infrastructure/       CLIP, Pinecone, MongoDB GridFS, OmniParser
    └── domain/               Pydantic models
```

## Setup

### 1. Prerequisites

- Python 3.10+
- FFmpeg (`choco install ffmpeg` or `winget install FFmpeg.FFmpeg`)
- Graphviz (`choco install graphviz`)
- MongoDB (for frame storage in vectordb_service)

### 2. Environment Variables

**Backend** (`backend/.env`):
```
GEMINI_API_KEY=your-key
```

**VectorDB Service** (`vectordb_service/.env`):
```
PINECONE_API_KEY=your-key
REPLICATE_API_TOKEN=your-token
GEMINI_API_KEY=your-key
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=pdd_vectordb
```

### 3. Install & Run

Each service runs in its own terminal. All services use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Terminal 1 — VectorDB Service
cd vectordb_service
uv sync
uv run uvicorn main:app --port 8000

# Terminal 2 — Backend
cd backend
uv sync
uv run uvicorn main:app --port 8001

# Terminal 3 — Frontend
cd frontend
uv sync
uv run streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Pipelines

**Meeting Recording (Audio+Video):**
Video → FFmpeg → Whisper transcript → Gemini LLM (sections + steps) → Flowchart → PDD

**Silent Screen Recording (Video Only):**
Video → SSIM scene detection → Frame extraction → OCR → Gemini Vision → Step synthesis → PDD
