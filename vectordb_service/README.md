# VectorDB Frame Service

FastAPI service that indexes video frames as vector embeddings and supports text-based querying with visual element location.

## What it does

1. **Upload a video** — detects scenes, extracts key frames (start/mid/end per scene), embeds each frame with CLIP ViT-L/14, upserts 768-d vectors into Pinecone, and stores frame images in MongoDB (GridFS).
2. **Query by text** — embeds the query with CLIP, searches Pinecone for the most similar frames, and returns ranked results.
3. **Query & Locate** — extends the query pipeline: Gemini selects the best frame, OmniParser detects UI elements, Gemini locates the target element, and a bounding box is drawn on the frame. Annotated images are stored in MongoDB.

## Pipeline: Query & Locate

```
User query
  → CLIP text embedding → Pinecone top-k search
  → Gemini picks the best matching frame
  → OmniParser (via Replicate) detects UI elements + bounding boxes
  → Gemini identifies the target element from OmniParser output
  → Bounding box drawn on frame → annotated image returned
```

## Setup

```bash
cd vectordb_service
cp .env.example .env   # fill in API keys
uv sync
uv run uvicorn main:app --reload
```

### Required environment variables

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Pinecone API key |
| `REPLICATE_API_TOKEN` | Replicate API token (for OmniParser) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `MONGO_URI` | MongoDB connection URI (default: `mongodb://localhost:27017`) |
| `MONGO_DB_NAME` | MongoDB database name (default: `pdd_vectordb`) |

## API Endpoints

### `POST /api/videos/upload`

Upload a video file for scene detection, frame extraction, and indexing.

- **Body**: `multipart/form-data` with `file` field (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- **Response**: `video_id`, scene/frame counts

### `POST /api/frames/query`

Text-based frame search using CLIP embeddings.

```json
{
  "query": "user typing in search box",
  "top_k": 5,
  "video_id": "abc123"   // optional filter
}
```

### `POST /api/frames/query/locate`

Full pipeline: search + Gemini frame selection + OmniParser + bounding box annotation.

```json
{
  "query": "click on first cell in excel sheet",
  "top_k": 5,
  "video_id": "abc123"
}
```

Returns the selected frame, located element, bounding box coordinates, and annotated image URL.

### `GET /api/frames/image/{video_id}/{filename}`

Serves frame images (including annotated ones) from MongoDB GridFS.

### `GET /health`

Health check.

## Project Structure

```
vectordb_service/
├── main.py                          # FastAPI app entry point
├── config.py                        # Settings (env vars + defaults)
├── api/
│   ├── dependencies.py              # Dependency injection
│   └── routes/
│       ├── video.py                 # Upload endpoint
│       ├── query.py                 # Query + Locate endpoints
│       └── frames.py               # Frame image serving (from MongoDB)
├── services/
│   ├── video_service.py             # Video processing pipeline
│   └── query_service.py             # Query + locate pipeline
├── infrastructure/
│   ├── embedding_service.py         # CLIP ViT-L/14 embeddings
│   ├── vector_store.py              # Pinecone wrapper
│   ├── scene_detector.py            # Scene boundary detection
│   ├── frame_extractor.py           # Frame extraction from video
│   ├── gemini_frame_selector.py     # Gemini frame selection + bbox location
│   ├── omniparser_client.py         # OmniParser via Replicate
│   ├── bbox_drawer.py               # Bounding box drawing utility
│   ├── mongo_frame_store.py         # MongoDB GridFS frame storage
│   └── ...
├── domain/
│   └── models.py                    # Pydantic request/response models
```

## Key Configuration Defaults

| Setting | Default |
|---------|---------|
| CLIP model | `openai/clip-vit-large-patch14` (768-d) |
| Pinecone index | `framesdb-v3` |
| Vector dimension | 768 |
| Pinecone metric | cosine |
| Scene threshold | 9.0 |
| Gemini model | `gemini-2.5-flash` |
| OmniParser | `microsoft/omniparser-v2` (via Replicate) |
