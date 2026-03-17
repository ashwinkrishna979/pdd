"""
Backend FastAPI Application.
Serves as the business logic layer for PDD generation.
"""


import os
import sys

# Ensure backend dir is in path for importing core/llm_tasks/etc.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
# Ensure workspace root is in path for vectordb_service import
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv(os.path.join(_backend_dir, ".env"))

from api.routes import pipeline, health, documents

# Mount vectordb_service as a sub-API
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../vectordb_service')))
from vectordb_service.main import app as vectordb_app

logging.basicConfig(level=logging.INFO)
# Ensure uvicorn doesn't suppress non-uvicorn loggers (e.g. vectordb service)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for _pkg in ("infrastructure", "services", "vectordb_service"):
    logging.getLogger(_pkg).setLevel(logging.INFO)

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
app.mount("/vectordb", vectordb_app)
