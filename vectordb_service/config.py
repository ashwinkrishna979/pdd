import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str = ""
    replicate_api_token: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    pinecone_index_name: str = "framesdb-v3"
    pinecone_dimension: int = 768
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    clip_model_id: str = "openai/clip-vit-large-patch14"
    data_dir: str = "./data"
    scene_threshold: float = 9.0
    omniparser_model: str = "microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df"
    cors_origins: List[str] = ["*"]

    # MongoDB settings for frame storage (set MONGO_URI in .env)
    mongo_uri: str = ""
    mongo_db_name: str = "pdd_vectordb"

    model_config = {"env_file": ".env"}

    def configure_env(self) -> None:
        """Set environment variables required by third-party libraries."""
        os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_token
