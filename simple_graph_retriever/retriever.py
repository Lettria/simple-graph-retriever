from typing import List, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from neo4j import GraphDatabase

from .models import RetrievalConfig, RetrievalResult
from .config import logger


class GraphRetriever:
    def __init__(
        self,
        qdrant_url,
        neo4j_uri,
        neo4j_auth,
        embedder_url,
        qdrant_api_key: Optional[str] = None,
        chunks_collection: str = "chunks",
        communities_collection: str = "communities",
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.embedder_url = embedder_url
        self.chunks_collection = chunks_collection
        self.communities_collection = communities_collection

    def close(self):
        self.neo4j_driver.close()

    def embed(self, text: str) -> list[float]:
        """
        Embeds a single string using the TEI embedder.
        """
        try:
            resp = requests.post(self.embedder_url, json={"inputs": text}, timeout=5)
            resp.raise_for_status()
            return resp.json()[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def search_qdrant(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        filters=None,
        score_threshold=None,
        score_drop_off_pct=None,
    ):
        try:
            search_result = self.qdrant_client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                query_filter=filters,
                score_threshold=score_threshold,
            )

            # Apply score drop-off filter
            if score_drop_off_pct is not None and len(search_result) > 1:
                filtered_results = [search_result[0]]
                for i in range(1, len(search_result)):
                    prev_score = search_result[i - 1].score
                    current_score = search_result[i].score

                    if prev_score == 0:
                        break

                    drop_off = (prev_score - current_score) / prev_score
                    if drop_off > score_drop_off_pct:
                        break  # Stop including results after a significant drop-off
                    filtered_results.append(search_result[i])
                return filtered_results

            return search_result
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def retrieve_communities(self, query_embedding, config: RetrievalConfig):
        if not config.use_communities:
            return []

        results = self.search_qdrant(
            collection=self.communities_collection,
            vector=query_embedding,
            limit=config.max_communities,
            score_threshold=config.community_score_threshold,
            score_drop_off_pct=config.community_score_drop_off_pct,
        )
        return [
            {"community_id": r.payload["community_id"], "score": r.score}
            for r in results
        ]

    def retrieve_chunks(
        self, query_embedding, community_results: list, config: RetrievalConfig
    ):
        if not config.use_chunks:
            return []

        filters = None
        if community_results:
            community_ids = [c["community_id"] for c in community_results]
            filters = Filter(
                must=[
                    FieldCondition(
                        key="community_id", match=MatchAny(any=community_ids)
                    )
                ]
            )

        results = self.search_qdrant(
            collection=self.chunks_collection,
            vector=query_embedding,
            limit=config.max_chunks,
            filters=filters,
            score_threshold=config.chunk_score_threshold,
            score_drop_off_pct=config.chunk_score_drop_off_pct,
        )

        return [
            {"center_node_id": r.payload["center_node_id"], "score": r.score}
            for r in results
        ]

    def retrieve_nodes_from_communities(
        self, community_ids: list, community_expansion_limit: int
    ):
        query = """
            MATCH (n)
            WHERE n.community_id IN $cids
            RETURN elementId(n) as id
            LIMIT $limit
        """
        with self.neo4j_driver.session() as session:
            result = session.run(
                query, cids=community_ids, limit=community_expansion_limit
            )
            return [record["id"] for record in result]

    def fetch_subgraph(
        self,
        center_node_ids: list,
        max_hops: int,
        allowed_rel_types: Optional[List[str]] = None,
        denied_rel_types: Optional[List[str]] = None,
    ):
        print("Fetching subgraph...")
        rel_type_filter_clauses = []
        if allowed_rel_types:
            rel_type_filter_clauses.append("type(r) IN $allowed_rel_types")
        if denied_rel_types:
            rel_type_filter_clauses.append("NOT type(r) IN $denied_rel_types")

        rel_filter_string = ""
        if rel_type_filter_clauses:
            rel_filter_string = " WHERE " + " AND ".join(rel_type_filter_clauses)

        query = f"""
            MATCH (n)
            WHERE elementId(n) IN $ids
            MATCH p=(n)-[r*1..{max_hops}]-(m)
            {rel_filter_string}
            UNWIND nodes(p) as node
            UNWIND relationships(p) as rel
            RETURN collect(DISTINCT node {{.*, element_id: elementId(node)}}) as nodes,
                   collect(DISTINCT rel {{.*, element_id: elementId(rel), type: type(rel), start_node_element_id: elementId(startNode(rel)), end_node_element_id: elementId(endNode(rel))}}) as relationships
        """
        with self.neo4j_driver.session() as session:
            params = {"ids": center_node_ids}
            if allowed_rel_types:
                params["allowed_rel_types"] = allowed_rel_types
            if denied_rel_types:
                params["denied_rel_types"] = denied_rel_types

            result = session.run(query, params)
            return result.data()

    def retrieve_graph(
        self, query: str, config: RetrievalConfig, include_chunks: bool = False
    ) -> RetrievalResult | None:
        """
        Retrieves a subgraph from the graph database based on a query string.
        """
        query_embedding = self.embed(query)
        if not query_embedding:
            return None

        community_results = self.retrieve_communities(query_embedding, config)
        community_scores = {c["community_id"]: c["score"] for c in community_results}

        chunk_results = []
        center_node_ids = []
        if config.use_chunks:
            chunk_results = self.retrieve_chunks(
                query_embedding, community_results, config
            )
            center_node_ids += [c["center_node_id"] for c in chunk_results]
        if config.use_communities:
            community_ids = [c["community_id"] for c in community_results]
            center_node_ids += self.retrieve_nodes_from_communities(
                community_ids,
                community_expansion_limit=config.community_expansion_limit,
            )

        if not center_node_ids:
            return None
        # print(f"Number of center nodes: {len(center_node_ids)}")
        subgraph = self.fetch_subgraph(
            center_node_ids,
            config.max_hops,
            config.allowed_rel_types,
            config.denied_rel_types,
        )
        # sub_graph_ids = [node["element_id"] for node in subgraph[0]["nodes"]]
        # print(f"Subgraph node IDs: {sub_graph_ids}")
        if subgraph and subgraph[0].get("nodes"):
            chunk_scores = {c["center_node_id"]: c["score"] for c in chunk_results}

            for node in subgraph[0]["nodes"]:
                node_id = node.get("element_id")
                community_id = node.get("community_id")
                if node_id in chunk_scores:
                    node["chunk_score"] = chunk_scores[node_id]
                if community_id in community_scores:
                    node["community_score"] = community_scores[community_id]

            # Sort nodes by chunk_score (descending), with 0 if not present
            subgraph[0]["nodes"] = sorted(
                subgraph[0]["nodes"],
                key=lambda node: node.get("chunk_score", 0),
                reverse=True,
            )

        if not include_chunks:
            nodes_to_keep = [
                node for node in subgraph[0]["nodes"] if "text" not in node
            ]
            node_ids_to_keep = {node["element_id"] for node in nodes_to_keep}
            relationships_to_keep = [
                rel
                for rel in subgraph[0]["relationships"]
                if rel["start_node_element_id"] in node_ids_to_keep
                and rel["end_node_element_id"] in node_ids_to_keep
            ]
            subgraph[0]["nodes"] = nodes_to_keep
            subgraph[0]["relationships"] = relationships_to_keep

        if not config.include_scores:
            for node in subgraph[0]["nodes"]:
                node.pop("chunk_score", None)
                node.pop("community_score", None)
        return RetrievalResult(
            nodes=subgraph[0]["nodes"], relationships=subgraph[0]["relationships"]
        )
