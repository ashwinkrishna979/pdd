"""
Backend FastAPI Application.
Serves as the business logic layer for PDD generation.
"""

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend dir is in path for importing core/llm_tasks/etc.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv

load_dotenv(os.path.join(_backend_dir, ".env"))

from api.routes import pipeline, health, documents

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="PDD Backend Service",
    description="Business logic API for PDD/BRD document generation from video recordings.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(pipeline.router)
app.include_router(documents.router)
