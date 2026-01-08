import logging
import os
import sys
from typing import Optional
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

loglevel = os.getenv("LOGLEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, loglevel, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Set httpx log level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress neo4j label warnings
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

logger = logging.getLogger("simple_graph_retriever")


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_chunks_collection: str = "chunks"
    qdrant_communities_collection: str = "communities"
    embedder_url: str = "http://localhost:8080"
    vector_size: int = 384
    loglevel: str = "WARNING"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


try:
    settings = Settings(**{})
except ValidationError as e:
    logger.error(e.json())

    sys.exit(1)
