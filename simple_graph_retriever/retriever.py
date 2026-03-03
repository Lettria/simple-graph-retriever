from typing import List, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from .models import RetrievalConfig, RetrievalResult
from .config import logger
from .db_adapter import GraphAdapter


class GraphRetriever:
    def __init__(
        self,
        db_adapter: GraphAdapter,
        qdrant_url,
        embedder_url,
        qdrant_api_key: Optional[str] = None,
        chunks_collection: str = "chunks",
        communities_collection: str = "communities",
    ):
        self.db = db_adapter
        self.qdrant_client = QdrantClient(
            url=qdrant_url, api_key=qdrant_api_key)
        self.embedder_url = embedder_url
        self.chunks_collection = chunks_collection
        self.communities_collection = communities_collection

    def embed(self, text: str) -> list[float]:
        """
        Embeds a single string using the TEI embedder.
        """
        try:
            resp = requests.post(self.embedder_url, json={
                                 "inputs": text}, timeout=5)
            resp.raise_for_status()
            return resp.json()[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def search_qdrant(self, collection, vector, limit, filters=None, score_threshold=None, score_drop_off_pct=None):
        try:
            search_result = self.qdrant_client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                query_filter=filters,
                score_threshold=score_threshold,
            )
            if score_drop_off_pct is not None and len(search_result) > 1:
                filtered_results = [search_result[0]]
                for i in range(1, len(search_result)):
                    prev_score = search_result[i - 1].score
                    current_score = search_result[i].score
                    if prev_score == 0:
                        break
                    drop_off = (prev_score - current_score) / prev_score
                    if drop_off > score_drop_off_pct:
                        break
                    filtered_results.append(search_result[i])
                return filtered_results
            return search_result
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def retrieve_communities(self, query_embedding, config: RetrievalConfig):
        if not config.use_communities:
            return []
        results = self.search_qdrant(self.communities_collection, query_embedding, config.max_communities,
                                     score_threshold=config.community_score_threshold, score_drop_off_pct=config.community_score_drop_off_pct)
        return [{"community_id": r.payload["community_id"], "score": r.score} for r in results]

    def retrieve_chunks(self, query_embedding, community_results: list, config: RetrievalConfig):
        if not config.use_chunks:
            return []
        filters = None
        if community_results:
            community_ids = [c["community_id"] for c in community_results]
            filters = Filter(must=[FieldCondition(
                key="community_id", match=MatchAny(any=community_ids))])

        results = self.search_qdrant(self.chunks_collection, query_embedding, config.max_chunks, filters=filters,
                                     score_threshold=config.chunk_score_threshold, score_drop_off_pct=config.chunk_score_drop_off_pct)
        return [{"center_node_id": r.payload["center_node_id"], "score": r.score} for r in results]

    def retrieve_nodes_from_communities(self, community_ids: list, community_expansion_limit: int):
        id_fn = self.db.id_function
        query = f"""
            MATCH (n)
            WHERE n.community_id IN $cids
            RETURN toString({id_fn}(n)) as id
            LIMIT $limit
        """
        result = self.db.query(
            query, params={'cids': community_ids, 'limit': community_expansion_limit})
        return [record["id"] for record in result]

    def fetch_subgraph(
        self,
        center_node_ids: list,
        max_hops: int,
        allowed_rel_types: Optional[List[str]] = None,
        denied_rel_types: Optional[List[str]] = None,
    ):
        print("Fetching subgraph...")
        id_fn = self.db.id_function

        rel_type_filter_clauses = []
        if allowed_rel_types:
            rel_type_filter_clauses.append("type(r) IN $allowed_rel_types")
        if denied_rel_types:
            rel_type_filter_clauses.append("NOT type(r) IN $denied_rel_types")
        rel_filter_string = " WHERE " + \
            " AND ".join(
                rel_type_filter_clauses) if rel_type_filter_clauses else ""

        # Using ID checks.
        # Since center_node_ids are strings (from Qdrant payload/elementId),
        # If FalkorDB, we must treat them as INTs in the WHERE clause unless we cast ID(n) to string in the match.
        # It is safer to cast ID(n) to string in the WHERE for compatibility: toString(ID(n)) IN $ids

        query = f"""
            MATCH (n)
            WHERE toString({id_fn}(n)) IN $ids
            MATCH p=(n)-[r*1..{max_hops}]-(m)
            {rel_filter_string}
            UNWIND nodes(p) as node
            UNWIND relationships(p) as rel
            RETURN collect(DISTINCT node {{.*, element_id: toString({id_fn}(node))}}) as nodes,
                   collect(DISTINCT rel {{.*, element_id: toString({id_fn}(rel)), type: type(rel), 
                           start_node_element_id: toString({id_fn}(startNode(rel))), 
                           end_node_element_id: toString({id_fn}(endNode(rel)))}}) as relationships
        """

        params = {"ids": center_node_ids}
        if allowed_rel_types:
            params["allowed_rel_types"] = allowed_rel_types
        if denied_rel_types:
            params["denied_rel_types"] = denied_rel_types

        return self.db.query(query, params=params)

    # retrieve_graph logic remains mostly the same, standard python logic
    def retrieve_graph(self, query: str, config: RetrievalConfig, include_chunks: bool = False) -> RetrievalResult | None:
        """
        Retrieves a subgraph from the graph database based on a query string.
        """
        query_embedding = self.embed(query)
        if not query_embedding:
            return None

        community_results = self.retrieve_communities(query_embedding, config)
        community_scores = {c["community_id"]: c["score"]
                            for c in community_results}

        chunk_results = []
        center_node_ids = []
        if config.use_chunks:
            chunk_results = self.retrieve_chunks(
                query_embedding, community_results, config)
            center_node_ids += [c["center_node_id"] for c in chunk_results]
        if config.use_communities:
            community_ids = [c["community_id"] for c in community_results]
            center_node_ids += self.retrieve_nodes_from_communities(
                community_ids, config.community_expansion_limit)

        if not center_node_ids:
            return None

        subgraph = self.fetch_subgraph(
            center_node_ids, config.max_hops, config.allowed_rel_types, config.denied_rel_types)

        if subgraph and subgraph[0].get("nodes"):
            chunk_scores = {c["center_node_id"]: c["score"]
                            for c in chunk_results}
            for node in subgraph[0]["nodes"]:
                node_id = node.get("element_id")
                community_id = node.get("community_id")
                if node_id in chunk_scores:
                    node["chunk_score"] = chunk_scores[node_id]
                if community_id in community_scores:
                    node["community_score"] = community_scores[community_id]

            subgraph[0]["nodes"] = sorted(
                subgraph[0]["nodes"], key=lambda node: node.get("chunk_score", 0), reverse=True)

        if not include_chunks and subgraph:
            nodes_to_keep = [node for node in subgraph[0]
                             ["nodes"] if "text" not in node]
            node_ids_to_keep = {node["element_id"] for node in nodes_to_keep}
            relationships_to_keep = [rel for rel in subgraph[0]["relationships"]
                                     if rel["start_node_element_id"] in node_ids_to_keep and rel["end_node_element_id"] in node_ids_to_keep]
            subgraph[0]["nodes"] = nodes_to_keep
            subgraph[0]["relationships"] = relationships_to_keep

        if not config.include_scores and subgraph:
            for node in subgraph[0]["nodes"]:
                node.pop("chunk_score", None)
                node.pop("community_score", None)

        if not subgraph:
            return RetrievalResult(nodes=[], relationships=[])

        return RetrievalResult(nodes=subgraph[0]["nodes"], relationships=subgraph[0]["relationships"])
