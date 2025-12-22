from typing import Optional, Tuple

from .models import RetrievalConfig
from .indexer import GraphIndexer
from .retriever import GraphRetriever
from .config import settings


class GraphRetrievalClient:
    """
    A client for interacting with the graph retrieval API.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_auth: Optional[Tuple[str, str]] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_chunks_collection: Optional[str] = None,
        qdrant_communities_collection: Optional[str] = None,
        embedder_url: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Initializes the GraphRetrievalClient.
        """
        neo4j_uri = neo4j_uri or settings.neo4j_uri
        neo4j_auth = neo4j_auth or (settings.neo4j_user, settings.neo4j_password)
        qdrant_url = qdrant_url or settings.qdrant_url
        qdrant_api_key = qdrant_api_key or settings.qdrant_api_key
        qdrant_chunks_collection = (
            qdrant_chunks_collection or settings.qdrant_chunks_collection
        )
        qdrant_communities_collection = (
            qdrant_communities_collection or settings.qdrant_communities_collection
        )
        embedder_url = embedder_url or settings.embedder_url
        vector_size = vector_size or settings.vector_size

        self._indexer = GraphIndexer(
            neo4j_uri=neo4j_uri,
            neo4j_auth=neo4j_auth,
            qdrant_url=qdrant_url,
            embedder_url=embedder_url,
            qdrant_api_key=qdrant_api_key,
            vector_size=vector_size,
            chunks_collection=qdrant_chunks_collection,
            communities_collection=qdrant_communities_collection,
        )
        self._retriever = GraphRetriever(
            qdrant_url=qdrant_url,
            neo4j_uri=neo4j_uri,
            neo4j_auth=neo4j_auth,
            embedder_url=embedder_url,
            qdrant_api_key=qdrant_api_key,
            chunks_collection=qdrant_chunks_collection,
            communities_collection=qdrant_communities_collection,
        )

    @property
    def indexer(self) -> GraphIndexer:
        return self._indexer

    @property
    def retriever(self) -> GraphRetriever:
        return self._retriever

    def close(self):
        self._indexer.close()
        self._retriever.close()

    def index(self):
        """
        Runs the complete graph indexing pipeline.
        """
        self.indexer.run_community_detection()
        self.indexer.create_chunks()
        self.indexer.index_chunks()
        self.indexer.index_communities()

    def clear_index(self):
        """
        Clears the Qdrant collections and community data from Neo4j.
        """
        self.indexer._clear_graph_chunks()
        self.indexer._clear_communities()

    def retrieve_graph(
        self, query: str, config: RetrievalConfig, include_chunks: bool = False
    ):
        """
        Retrieves relevant graph data based on the query.
        """
        return self.retriever.retrieve_graph(
            query=query, config=config, include_chunks=include_chunks
        )
