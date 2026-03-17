"""FastAPI dependency injection — provides singleton services to route handlers."""

from functools import lru_cache

from config import Settings
from infrastructure.embedding_service import EmbeddingService
from infrastructure.mongo_frame_store import MongoFrameStore
from infrastructure.vector_store import VectorStore
from services.query_service import QueryService
from services.video_service import VideoService


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.configure_env()
    return settings


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings().clip_model_id)


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(
        api_key=s.pinecone_api_key,
        index_name=s.pinecone_index_name,
        dimension=s.pinecone_dimension,
        cloud=s.pinecone_cloud,
        region=s.pinecone_region,
    )


@lru_cache
def get_mongo_frame_store() -> MongoFrameStore:
    s = get_settings()
    return MongoFrameStore(mongo_uri=s.mongo_uri, db_name=s.mongo_db_name)


def get_video_service() -> VideoService:
    return VideoService(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        frame_store=get_mongo_frame_store(),
    )


def get_query_service() -> QueryService:
    return QueryService(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        frame_store=get_mongo_frame_store(),
    )
