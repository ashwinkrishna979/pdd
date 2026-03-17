import requests
import os
import tempfile
from typing import Dict, Any, List, Optional

VECTORDB_URL = os.getenv("VECTORDB_URL", "http://localhost:8000")


def upload_video(video_path: str) -> Dict[str, Any]:
    url = f"{VECTORDB_URL}/api/videos/upload"
    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        response = requests.post(url, files=files)
    response.raise_for_status()
    return response.json()


def query_locate(query: str, video_id: str, top_k: int = 5) -> Dict[str, Any]:
    url = f"{VECTORDB_URL}/api/frames/query/locate"
    payload = {"query": query, "video_id": video_id, "top_k": top_k}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def download_frame(video_id: str, filename: str, dest_dir: str) -> Optional[str]:
    """Download a frame image from the VectorDB service and save to dest_dir."""
    url = f"{VECTORDB_URL}/api/frames/image/{video_id}/{filename}"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return None
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def download_all_frames(video_id: str, dest_dir: str) -> List[str]:
    """Download all frame images for a video from the VectorDB service.

    Uses the query endpoint with a broad query to discover all frames,
    then downloads each one.
    """
    # Query with a very broad term to get frame list
    url = f"{VECTORDB_URL}/api/frames/query"
    payload = {"query": "screen", "video_id": video_id, "top_k": 500}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception:
        return []

    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    seen = set()
    for match in results:
        filename = match.get("metadata", {}).get("filename", "")
        if not filename or filename in seen:
            continue
        seen.add(filename)
        path = download_frame(video_id, filename, dest_dir)
        if path:
            paths.append(path)
    return sorted(paths)
