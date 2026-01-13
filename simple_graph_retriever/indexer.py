import json
import time
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import requests
from tqdm import tqdm
import igraph as ig
import leidenalg as la
from .config import logger


class GraphIndexer:
    def __init__(
        self,
        neo4j_uri,
        neo4j_auth,
        qdrant_url,
        embedder_url,
        qdrant_api_key: Optional[str] = None,
        vector_size=384,
        chunks_collection: str = "chunks",
        communities_collection: str = "communities",
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder_url = embedder_url
        self.vector_batch_size = 100
        self.vector_size = vector_size
        self.chunks_collection = chunks_collection
        self.communities_collection = communities_collection
        self._ensure_qdrant_collections_exist()

    def _ensure_qdrant_collections_exist(self):
        logger.info("Ensuring Qdrant collections exist...")
        existing_collections = [
            c.name for c in self.qdrant.get_collections().collections
        ]

        if self.chunks_collection not in existing_collections:
            self.qdrant.create_collection(
                collection_name=self.chunks_collection,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            logger.info(f"Created '{self.chunks_collection}' collection.")
        else:
            logger.info(f"'{self.chunks_collection}' collection already exists.")

        if self.communities_collection not in existing_collections:
            self.qdrant.create_collection(
                collection_name=self.communities_collection,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            logger.info(f"Created '{self.communities_collection}' collection.")
        else:
            logger.info(f"'{self.communities_collection}' collection already exists.")
        logger.info("✅ Qdrant collections checked/created.")

    def close(self):
        self.driver.close()
        self.qdrant.close()

    def _clear_graph_chunks(self):
        """
        Wipes all graph chunk data from both Neo4j and Qdrant.
        """
        logger.info("   Wiping all graph chunk data...")

        # Wipe old chunk data from Qdrant
        logger.info(f"   Wiping Qdrant collection: '{self.chunks_collection}'...")
        self.qdrant.recreate_collection(
            collection_name=self.chunks_collection,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )

        # Wipe old chunk data from Neo4j
        with self.driver.session() as session:
            logger.info("   Wiping graph chunk data from Neo4j...")
            session.run("MATCH (c:GraphChunk) DETACH DELETE c")
            logger.info("   Graph chunk data wiped.")

    def _clear_communities(self):
        """
        Wipes all community-related data from both Neo4j and Qdrant.
        """
        logger.info("   Wiping all community data...")

        # Wipe old community data from Qdrant
        logger.info(f"   Wiping Qdrant collection: '{self.communities_collection}'...")
        self.qdrant.recreate_collection(
            collection_name=self.communities_collection,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )

        # Wipe old community data from Neo4j
        with self.driver.session() as session:
            logger.info("   Wiping community data from Neo4j...")
            session.run("MATCH (c:Community) DETACH DELETE c")
            session.run(
                "MATCH (n) WHERE n.community_id IS NOT NULL REMOVE n.community_id"
            )
            logger.info("   Community data wiped.")

    def run_community_detection(self):
        logger.info("1️⃣  Refreshing Community Structure...")
        self._clear_communities()

        with self.driver.session() as session:
            # Fetch all nodes and relationships from Neo4j (excluding chunks and communities)
            logger.info(
                "   Fetching graph data from Neo4j (excluding chunks and communities)..."
            )
            nodes_data = session.run(
                "MATCH (n) WHERE NOT n:GraphChunk AND NOT n:Community RETURN elementId(n) as id"
            ).data()
            rels_data = session.run(
                "MATCH (a)-[r]->(b) WHERE NOT a:GraphChunk AND NOT a:Community AND NOT b:GraphChunk AND NOT b:Community RETURN elementId(r) as id, elementId(a) as source, elementId(b) as target"
            ).data()

            node_id_to_idx = {node["id"]: i for i, node in enumerate(nodes_data)}

            # Create an igraph graph
            g = ig.Graph(directed=True)
            g.add_vertices(len(nodes_data))
            g.vs["neo4j_id"] = [node["id"] for node in nodes_data]

            edges = []
            for rel in rels_data:
                source_idx = node_id_to_idx.get(rel["source"])
                target_idx = node_id_to_idx.get(rel["target"])
                if source_idx is not None and target_idx is not None:
                    edges.append((source_idx, target_idx))
            g.add_edges(edges)

            logger.info(
                f"   Created igraph with {g.vcount()} vertices and {g.ecount()} edges."
            )

            # Run Leiden algorithm
            logger.info("   Running Leiden algorithm...")
            partition = la.find_partition(g, la.ModularityVertexPartition)
            logger.info(f"✅ Detected {len(partition)} communities.")

            # Write community IDs back to Neo4j
            logger.info("   Writing community IDs to Neo4j...")
            tx = session.begin_transaction()
            batch_size = 1000
            count = 0
            for i, community_id in enumerate(partition.membership):
                neo4j_node_id = g.vs[i]["neo4j_id"]
                tx.run(
                    "MATCH (n) WHERE elementId(n) = $neo4j_node_id SET n.community_id = $community_id",
                    neo4j_node_id=neo4j_node_id,
                    community_id=community_id,
                )
                count += 1
                if count % batch_size == 0:
                    tx.commit()
                    tx = session.begin_transaction()
            tx.commit()
            logger.info(f"✅ Wrote community IDs for {count} nodes.")

            # Materialize Community Nodes (Optional but good for linkage)
            session.run(
                """
                MATCH (n) WHERE n.community_id IS NOT NULL
                WITH n.community_id AS cid, count(n) as size
                MERGE (c:Community {id: cid})
                SET c.size = size
            """
            )
            logger.info("✅ Materialized :Community nodes.")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding wrapper.
        Adjust payload/response handling based on your specific API.
        """
        # Example assumes an OpenAI-compatible interface or similar list-in/list-out
        embeddings = []
        for text in texts:
            try:
                resp = requests.post(
                    self.embedder_url,
                    json={"inputs": [text]},  # Sending as a list of one string
                    timeout=30,
                )
                resp.raise_for_status()
                # Assuming response is a list of embeddings, even for single input
                # e.g., [[0.1, 0.2, ...]]
                embeddings.extend(
                    resp.json()
                )  # Use extend because response is list of lists
            except Exception as e:
                logger.error(f"Embedding failed for text '{text[:50]}...': {e}")
                embeddings.append([])  # Append empty list for failed embeddings
        return embeddings

    def create_chunks(self):
        logger.info("2️⃣  Creating GraphChunks (Node + Context).")
        self._clear_graph_chunks()
        query = """
        MATCH (n)
        WHERE NOT n:GraphChunk AND NOT n:Community
        // Ensure we only create a chunk if one doesn't already exist for the node
        AND NOT (n)<-[:CENTERED_ON]-(:GraphChunk)
        
        // This query will now run once and create all necessary chunks
        WITH n 

        // 1. Gather properties and textualize them
        WITH n,
             apoc.text.join(labels(n), ", ") as lbls,
             apoc.text.join(
                [key IN keys(properties(n)) WHERE NOT key IN ['uuid', 'community_id'] | key + ": " + toString(properties(n)[key])], 
                "\\n"
             ) as props_text

        // 2. Gather 1-hop context (Relationships + Neighbor Labels)
        CALL (n) {
            MATCH (n)-[r]-(m)
            WITH type(r) as rel_type, labels(m) as n_labels, m
            RETURN collect(
                rel_type + " -> " + head(labels(m)) + ":" + coalesce(apoc.map.get(properties(m), 'label', null), apoc.map.get(properties(m), 'name', null), "Node")
            )[0..10] as context_list // Capping context to 10 neighbors
        }

        // 3. Format Text Blob
        WITH n, lbls, props_text, context_list,
             "Node: " + lbls + "\\nProps:\\n" + props_text + "\\nContext:\\n" +
             apoc.text.join(context_list, "\\n") as chunk_text

        // 4. Create Chunk Node
        CREATE (c:GraphChunk {
            id: randomUUID(),
            community_id: n.community_id,
            text: chunk_text
        })
        CREATE (c)-[:CENTERED_ON]->(n)
        RETURN count(c) as created_count
        """

        with self.driver.session() as session:
            result = session.run(query).single()
            count = result["created_count"] if result else 0
            logger.info(f"✅ Total chunks created: {count}")

    def index_chunks(self):
        logger.info("3️⃣  Indexing Chunks to Qdrant...")

        # Read chunks that are not yet indexed
        fetch_query = """
        MATCH (c:GraphChunk)-[:CENTERED_ON]->(n)
        WHERE c.indexed IS NULL
        RETURN c.id as id, c.text as text, c.community_id as comm_id,
               elementId(n) as center_node_id
        LIMIT $batch_size
        """

        mark_done_query = """
        MATCH (c:GraphChunk) WHERE c.id IN $ids
        SET c.indexed = true
        """

        with self.driver.session() as session:
            while True:
                # 1. Fetch Batch from Neo4j
                records = session.run(
                    fetch_query, batch_size=self.vector_batch_size
                ).data()
                if not records:
                    break

                texts = [r["text"] for r in records]
                ids = [r["id"] for r in records]

                # 2. Embed
                vectors = self.embed_batch(texts)

                # 3. Prepare Points
                points = []
                for i, rec in enumerate(records):
                    if not vectors[i]:
                        logger.warning(
                            f"Skipping chunk {rec['id']} due to failed embedding."
                        )
                        continue  # skip failed embeddings

                    points.append(
                        PointStruct(
                            id=rec["id"],  # Using UUID from Neo4j
                            vector=vectors[i],
                            payload={
                                "text": rec["text"],
                                "community_id": rec["comm_id"],
                                "center_node_id": rec[
                                    "center_node_id"
                                ],  # Store internal ID for retrieval
                                "type": "chunk",
                            },
                        )
                    )

                # 4. Upsert to Qdrant
                if points:
                    self.qdrant.upsert(
                        collection_name=self.chunks_collection, points=points
                    )

                # 5. Mark as done in Neo4j
                session.run(mark_done_query, ids=ids)
                logger.info(f"   Indexed {len(points)} chunks.")

        logger.info("✅ Chunk indexing complete.")

    def index_communities(self):
        logger.info("4️⃣  Indexing Communities...")

        # Strategy: Aggregate the text of the 5 most connected nodes in the community
        query = """
        MATCH (c:Community)
        WHERE c.indexed IS NULL

        CALL (c) {
            MATCH (n) WHERE n.community_id = c.id AND NOT n:GraphChunk
            // Heuristic: Pick 'important' nodes by degree
            WITH n ORDER BY COUNT { (n)--() } DESC LIMIT 5

            // Re-generate basic text for these nodes
            WITH n, head(labels(n)) + ":" + coalesce(apoc.map.get(properties(n), 'label', null), apoc.map.get(properties(n), 'name', null), "Item") as summary
            RETURN collect(summary) as summaries
        }

        WITH c, "Community ID: " + toString(c.id) + "\nKey Elements:\n" + apoc.text.join(summaries, "\n") as comm_text
        RETURN c.id as id, comm_text as text
        LIMIT $batch_size
        """

        mark_done = "MATCH (c:Community {id: $id}) SET c.indexed = true, c.text = $text"

        with self.driver.session() as session:
            while True:
                records = session.run(query, batch_size=self.vector_batch_size).data()
                if not records:
                    break

                texts = [r["text"] for r in records]
                vectors = self.embed_batch(texts)

                points = []
                for i, rec in enumerate(records):
                    if not vectors[i]:
                        continue

                    # Qdrant requires integer or UUID for ID.
                    # If community_id is integer, use it directly.
                    # If it's a string/uuid, you might need to hash it to UUID.
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
                    session.run(mark_done, id=r["id"], text=r["text"])

                logger.info(f"   Indexed {len(points)} communities.")

        logger.info("✅ Community indexing complete.")
