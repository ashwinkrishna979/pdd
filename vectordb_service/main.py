import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vectordb_service.api.routes import query, video, frames
from config import Settings

logging.basicConfig(level=logging.INFO)

settings = Settings()
settings.configure_env()

app = FastAPI(
    title="VectorDB Frame Service",
    description="Upload videos, extract & index frames, and query them by text. Frames stored in MongoDB.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(video.router)
app.include_router(query.router)
app.include_router(frames.router)


@app.get("/health")
def health():
    return {"status": "ok"}
