from typing import Optional, Tuple
from .models import RetrievalConfig
from .indexer import GraphIndexer
from .retriever import GraphRetriever
from .config import settings
from .db_adapter import Neo4jAdapter, FalkorDBAdapter, GraphAdapter

class GraphRetrievalClient:
    def __init__(
        self,
        # Common
        graph_db_type: Optional[str] = None,
        # Neo4j
        neo4j_uri: Optional[str] = None,
        neo4j_auth: Optional[Tuple[str, str]] = None,
        # FalkorDB
        falkordb_host: Optional[str] = None,
        falkordb_port: Optional[int] = None,
        falkordb_graph_name: Optional[str] = None,
        # Vector DB
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_chunks_collection: Optional[str] = None,
        qdrant_communities_collection: Optional[str] = None,
        embedder_url: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        # 1. Initialize Database Adapter
        db_type = graph_db_type or settings.graph_db_type
        
        self.db_adapter: GraphAdapter
        
        if db_type.lower() == "falkordb":
            self.db_adapter = FalkorDBAdapter(
                host=falkordb_host or settings.falkordb_host,
                port=falkordb_port or settings.falkordb_port,
                username=settings.falkordb_username, # Optional
                password=settings.falkordb_password, # Optional
                graph_name=falkordb_graph_name or settings.falkordb_graph_name
            )
        else:
            # Default to Neo4j
            uri = neo4j_uri or settings.neo4j_uri
            auth = neo4j_auth or (settings.neo4j_user, settings.neo4j_password)
            self.db_adapter = Neo4jAdapter(uri=uri, auth=auth)

        # 2. Settings
        qdrant_url = qdrant_url or settings.qdrant_url
        qdrant_api_key = qdrant_api_key or settings.qdrant_api_key
        qdrant_chunks_collection = qdrant_chunks_collection or settings.qdrant_chunks_collection
        qdrant_communities_collection = qdrant_communities_collection or settings.qdrant_communities_collection
        embedder_url = embedder_url or settings.embedder_url
        vector_size = vector_size or settings.vector_size

        # 3. Inject Adapter into Indexer and Retriever
        self._indexer = GraphIndexer(
            db_adapter=self.db_adapter,
            qdrant_url=qdrant_url,
            embedder_url=embedder_url,
            qdrant_api_key=qdrant_api_key,
            vector_size=vector_size,
            chunks_collection=qdrant_chunks_collection,
            communities_collection=qdrant_communities_collection,
        )
        self._retriever = GraphRetriever(
            db_adapter=self.db_adapter,
            qdrant_url=qdrant_url,
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
        self.db_adapter.close()
        self._indexer.close_qdrant()
        # self._retriever.close() # Retriever doesn't hold separate connection now

    def index(self):
        self.indexer.run_community_detection()
        self.indexer.create_chunks()
        self.indexer.index_chunks()
        self.indexer.index_communities()

    def clear_index(self):
        self.indexer._clear_graph_chunks()
        self.indexer._clear_communities()

    def retrieve_graph(self, query: str, config: RetrievalConfig, include_chunks: bool = False):
        return self.retriever.retrieve_graph(query=query, config=config, include_chunks=include_chunks)