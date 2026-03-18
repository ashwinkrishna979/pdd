import logging
import time

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a Pinecone index."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self._pc = Pinecone(api_key=api_key)

        if not self._pc.has_index(index_name):
            logger.info("Index '%s' not found — creating...", index_name)
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            self._wait_for_ready(index_name)

        self._index = self._pc.Index(index_name)

    def _wait_for_ready(self, index_name: str, timeout: int = 120) -> None:
        """Block until the index status is 'Ready'."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            desc = self._pc.describe_index(index_name)
            if desc.status.get("ready", False):
                logger.info("Index '%s' is ready.", index_name)
                return
            logger.info("Waiting for index '%s' to be ready...", index_name)
            time.sleep(3)
        raise TimeoutError(f"Index '{index_name}' not ready after {timeout}s")

    def upsert(self, vectors: list[tuple]) -> None:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[i : i + batch_size])

    def query(self, vector: list[float], top_k: int = 5, filter: dict | None = None) -> dict:
        kwargs = dict(vector=vector, top_k=top_k, include_metadata=True)
        if filter:
            kwargs["filter"] = filter
        return self._index.query(**kwargs)

    def delete_by_video_id(self, video_id: str) -> None:
        """Delete all vectors whose metadata.video_id matches."""
        self._index.delete(filter={"video_id": {"$eq": video_id}})

    def stats(self) -> dict:
        return self._index.describe_index_stats()
