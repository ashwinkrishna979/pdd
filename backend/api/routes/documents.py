"""
Document-related routes — list generated documents, download from persistent outputs.
"""

import os
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.config import config

router = APIRouter(prefix="/api/documents", tags=["documents"])


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int


@router.get("/", response_model=List[DocumentInfo])
def list_documents():
    """List all generated documents in the outputs directory."""
    output_dir = config.paths.output_dir
    if not os.path.exists(output_dir):
        return []
    docs = []
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".docx"):
            path = os.path.join(output_dir, f)
            docs.append(DocumentInfo(filename=f, size_bytes=os.path.getsize(path)))
    return docs


@router.get("/download/{filename}")
def download_document(filename: str):
    """Download a generated document by filename."""
    # Sanitize filename to prevent path traversal
    safe_name = os.path.basename(filename)
    path = os.path.join(config.paths.output_dir, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Document not found: {safe_name}")
    return FileResponse(
        path=path,
        filename=safe_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
