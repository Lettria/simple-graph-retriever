import json
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import requests
import igraph as ig
import leidenalg as la
from .config import logger
from .db_adapter import GraphAdapter


class GraphIndexer:
    def __init__(
        self,
        db_adapter: GraphAdapter,  # Changed from uri/auth to adapter
        qdrant_url,
        embedder_url,
        qdrant_api_key: Optional[str] = None,
        vector_size=384,
        chunks_collection: str = "chunks",
        communities_collection: str = "communities",
    ):
        self.db = db_adapter
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder_url = embedder_url
        self.vector_batch_size = 100
        self.vector_size = vector_size
        self.chunks_collection = chunks_collection
        self.communities_collection = communities_collection
        self._ensure_qdrant_collections_exist()

    def close_qdrant(self):
        self.qdrant.close()

    # _ensure_qdrant_collections_exist remains the same...
    def _ensure_qdrant_collections_exist(self):
        # ... (same as your original code) ...
        pass

    def _clear_graph_chunks(self):
        logger.info("   Wiping all graph chunk data...")
        self.qdrant.recreate_collection(
            collection_name=self.chunks_collection,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )
        logger.info("   Wiping graph chunk data from DB...")
        self.db.query("MATCH (c:GraphChunk) DETACH DELETE c")

    def _clear_communities(self):
        logger.info("   Wiping all community data...")
        self.qdrant.recreate_collection(
            collection_name=self.communities_collection,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )
        logger.info("   Wiping community data from DB...")
        self.db.query("MATCH (c:Community) DETACH DELETE c")
        self.db.query(
            "MATCH (n) WHERE n.community_id IS NOT NULL REMOVE n.community_id"
        )

    def run_community_detection(self):
        logger.info("1️⃣  Refreshing Community Structure...")
        self._clear_communities()

        # Dynamic ID function (elementId or ID)
        id_fn = self.db.id_function

        logger.info("   Fetching graph data...")
        # Note: toString() handles FalkorDB integers vs Neo4j strings
        logger.info(f"Using ID function: {id_fn}(n)")
        nodes_data = self.db.query(
            f"MATCH (n) WHERE NOT n:GraphChunk AND NOT n:Community RETURN {id_fn}(n) as id"
        )
        logger.info(f"Nodes fetched: {len(nodes_data)}")
        rels_data = self.db.query(
            f"MATCH (a)-[r]->(b) WHERE NOT a:GraphChunk AND NOT a:Community AND NOT b:GraphChunk AND NOT b:Community "
            f"RETURN {id_fn}(r) as id, {id_fn}(a) as source, {id_fn}(b) as target"
        )
        if nodes_data:
            logger.info(f"DEBUG: First node keys: {nodes_data[0].keys()}")
        else:
            logger.warning("DEBUG: nodes_data is empty!")
        node_id_to_idx = {str(node["id"]): i for i, node in enumerate(nodes_data)}
        g = ig.Graph(directed=True)
        g.add_vertices(len(nodes_data))
        g.vs["graph_id"] = [str(node["id"]) for node in nodes_data]

        edges = []
        for rel in rels_data:
            source_id = str(rel.get("source"))
            target_id = str(rel.get("target"))

            source_idx = node_id_to_idx.get(source_id)
            target_idx = node_id_to_idx.get(target_id)
            if source_idx is not None and target_idx is not None:
                edges.append((source_idx, target_idx))
            else:
                logger.warning(
                    f"Skipping edge with missing node: {source_id} -> {target_id}"
                )
        g.add_edges(edges)
        logger.info(f"   Running Leiden algorithm on {g.vcount()} nodes...")
        partition = la.find_partition(g, la.ModularityVertexPartition)
        logger.info(f"✅ Detected {len(partition)} communities.")

        logger.info("   Writing community IDs to DB...")
        batch_size = 1000
        count = 0

        # We can't use transactions easily across adapters, so we run simple queries
        # Or you can batch them in the adapter, but simple loop is fine for now
        for i, community_id in enumerate(partition.membership):
            graph_node_id = g.vs[i]["graph_id"]

            # Handling ID matching: if it's FalkorDB (int ID), we need to handle the conversion
            # In Cypher: ID(n) = 123. In Neo4j: elementId(n) = "4:..."
            # Safest way: pass as parameter and let DB match
            if self.db.id_function == "ID":
                # FalkorDB expects int for ID() check usually, but we cast toString earlier
                # Reverting to simple WHERE ID(n) = int(...)
                # But we stored string version.
                query = f"MATCH (n) WHERE ID(n) = {graph_node_id} SET n.community_id = {community_id}"
            else:
                query = f"MATCH (n) WHERE elementId(n) = '{graph_node_id}' SET n.community_id = {community_id}"

            self.db.query(query)
            count += 1
            if count % 100 == 0:
                print(f"Updates: {count}", end="\r")

        # Materialize Community Nodes
        self.db.query(
            """
            MATCH (n) WHERE n.community_id IS NOT NULL
            WITH n.community_id AS cid, count(n) as size
            MERGE (c:Community {id: cid})
            SET c.size = size
        """
        )
        logger.info("✅ Materialized :Community nodes.")

    # embed_batch method remains the same...
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # ... (same as original) ...
        embeddings = []
        for text in texts:
            truncated_text = text[:500]
            try:
                resp = requests.post(
                    self.embedder_url, json={"inputs": [truncated_text]}, timeout=30
                )
                resp.raise_for_status()
                embeddings.extend(resp.json())
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append([])
        return embeddings

    def create_chunks(self):
        logger.info("2️⃣  Creating GraphChunks...")
        self._clear_graph_chunks()

        # Rewritten to remove dependency on APOC.text.join for compatibility
        query = """
        MATCH (n)
        WHERE NOT n:GraphChunk AND NOT n:Community
        AND NOT (n)<-[:CENTERED_ON]-(:GraphChunk)
        WITH n 

        // 1. Textualize Properties (Standard Cypher)
        WITH n,
             reduce(s = "", l IN labels(n) | s + l + ", ") as lbls,
             reduce(s = "", k IN [k IN keys(properties(n)) WHERE k <> 'uuid' AND k <> 'community_id'] | 
                s + k + ": " + toString(properties(n)[k]) + "\n") as props_text

        // 2. Gather Context
        CALL {
            WITH n
            MATCH (n)-[r]-(m)
            WITH type(r) as rel_type, labels(m) as n_labels, m
            LIMIT 10
            RETURN collect(
                rel_type + " -> " + head(labels(m)) + ":" + 
                coalesce(m.label, m.name, "Node")
            ) as context_list
        }

        // 3. Format
        WITH n, lbls, props_text, 
             reduce(s = "", c IN context_list | s + c + "\n") as context_text

        WITH n, 
             "Node: " + lbls + "\nProps:\n" + props_text + "\nContext:\n" + context_text as chunk_text

        CREATE (c:GraphChunk {
            id: randomUUID(),
            community_id: n.community_id,
            text: chunk_text
        })
        CREATE (c)-[:CENTERED_ON]->(n)
        RETURN count(c) as created_count
        """

        result = self.db.query(query)
        # Handle list of dicts result
        count = result[0]["created_count"] if result else 0
        logger.info(f"✅ Total chunks created: {count}")

    def index_chunks(self):
        logger.info("3️⃣  Indexing Chunks...")
        id_fn = self.db.id_function
        # Using toString for ID compatibility
        fetch_query = f"""
        MATCH (c:GraphChunk)-[:CENTERED_ON]->(n)
        WHERE c.indexed IS NULL
        RETURN c.id as id, c.text as text, c.community_id as comm_id,
               toString({id_fn}(n)) as center_node_id
        LIMIT $batch_size
        """

        mark_done_query = """
        MATCH (c:GraphChunk) WHERE c.id IN $ids
        SET c.indexed = true
        """

        while True:
            records = self.db.query(
                fetch_query, params={"batch_size": self.vector_batch_size}
            )
            if not records:
                break

            texts = [r["text"] for r in records]
            ids = [r["id"] for r in records]  # Chunk IDs are UUID strings

            vectors = self.embed_batch(texts)
            points = []
            for i, rec in enumerate(records):
                if not vectors[i]:
                    continue
                points.append(
                    PointStruct(
                        id=rec["id"],
                        vector=vectors[i],
                        payload={
                            "text": rec["text"],
                            "community_id": rec["comm_id"],
                            "center_node_id": rec["center_node_id"],
                            "type": "chunk",
                        },
                    )
                )

            if points:
                self.qdrant.upsert(
                    collection_name=self.chunks_collection, points=points
                )

            self.db.query(mark_done_query, params={"ids": ids})
            logger.info(f"   Indexed {len(points)} chunks.")

    def index_communities(self):
        logger.info("4️⃣  Indexing Communities...")

        # Corrected Query: Removed '#' comments and used '//' or no comments
        query = """
        MATCH (c:Community)
        WHERE c.indexed IS NULL

        CALL {
            WITH c
            MATCH (n) WHERE n.community_id = c.id AND NOT n:GraphChunk
            
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as degree
            ORDER BY degree DESC LIMIT 5

            WITH n, head(labels(n)) + ":" + coalesce(n.label, n.name, "Item") as summary
            RETURN collect(summary) as summaries
        }

        WITH c, "Community ID: " + toString(c.id) + "\nKey Elements:\n" + 
             reduce(s="", x IN summaries | s + x + "\n") as comm_text
        
        RETURN c.id as id, comm_text as text
        LIMIT $batch_size
        """

        mark_done = "MATCH (c:Community {id: $id}) SET c.indexed = true, c.text = $text"

        while True:
            records = self.db.query(
                query, params={"batch_size": self.vector_batch_size}
            )
            if not records:
                break

            texts = [r["text"] for r in records]
            vectors = self.embed_batch(texts)
            points = []

            for i, rec in enumerate(records):
                if not vectors[i]:
                    continue

                points.append(
                    PointStruct(
                        id=rec["id"],
                        vector=vectors[i],
                        payload={
                            "text": rec["text"],
                            "community_id": rec["id"],
                            "type": "community",
                        },
                    )
                )

            if points:
                self.qdrant.upsert(
                    collection_name=self.communities_collection, points=points
                )

            for r in records:
                self.db.query(mark_done, params={"id": r["id"], "text": r["text"]})

            logger.info(f"   Indexed {len(points)} communities.")
