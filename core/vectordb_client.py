import requests
import os
from typing import Dict, Any, List

VECTORDB_URL = "http://localhost:8000"


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
