from typing import Callable, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from .models import RetrievalConfig, RetrievalResult, RetrievalStepEvent, RetrievalStepSubEvent
from .config import logger
from .db_adapter import GraphAdapter
from .nl2cypher import CypherGenerator


class GraphRetriever:
    def __init__(
        self,
        db_adapter: GraphAdapter,
        qdrant_url,
        embed_fn: Callable[[List[str]], List[List[float]]],
        qdrant_api_key: Optional[str] = None,
        chunks_collection: str = "chunks",
        communities_collection: str = "communities",
        cypher_generator: Optional[CypherGenerator] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.db = db_adapter
        self.qdrant_client = QdrantClient(
            url=qdrant_url, api_key=qdrant_api_key)
        self.embed_fn = embed_fn
        self.chunks_collection = chunks_collection
        self.communities_collection = communities_collection
        self.cypher_generator = cypher_generator
        self.llm_fn = llm_fn

    def embed(self, text: str) -> list[float]:
        return self.embed_fn([text])[0]

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
        community_results = [{"community_id": r.payload["community_id"], "score": r.score} for r in results]
        logger.info(
            f"Vector search: {len(community_results)} matching communities "
            f"(community_ids={[c['community_id'] for c in community_results]})"
        )
        return community_results

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
        chunk_results = [{"center_node_id": r.payload["center_node_id"], "score": r.score} for r in results]
        logger.info(
            f"Vector search: {len(chunk_results)} matching chunks "
            f"({len({c['center_node_id'] for c in chunk_results})} distinct center nodes)"
        )
        return chunk_results

    def retrieve_nodes_from_communities(self, community_ids: list, community_expansion_limit: int):
        id_fn = self.db.id_function
        # Scoped to :_CommunityMember (set by GraphIndexer.run_community_detection
        # alongside community_id) so the planner can use the index on that label
        # instead of a full node-store scan.
        query = f"""
            MATCH (n:_CommunityMember)
            WHERE n.community_id IN $cids
            RETURN toString({id_fn}(n)) as id
            LIMIT $limit
        """
        params = {'cids': community_ids, 'limit': community_expansion_limit}
        result = self.db.query(query, params=params)
        seed_ids = [record["id"] for record in result]
        logger.info(
            f"Community expansion: {len(seed_ids)} additional seed nodes from {len(community_ids)} communities"
        )
        return seed_ids

    def fetch_subgraph(
        self,
        center_node_ids: list,
        max_hops: int,
        allowed_rel_types: Optional[List[str]] = None,
        denied_rel_types: Optional[List[str]] = None,
    ):
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

    def iter_retrieve_graph(self, query: str, config: RetrievalConfig, include_chunks: bool = False):
        """
        Same retrieval pipeline as retrieve_graph(), yielding a RetrievalStepEvent
        after each phase that actually ran, for live progress. The final event has
        key="done" and its `result` is exactly what retrieve_graph() would return.

        This is lazy (partial iteration = partial execution) and single-use, like
        any generator. Mirrors retrieve_graph()'s control flow below it in this
        file — keep both in sync if the retrieval algorithm changes.
        """
        query_embedding = self.embed(query)
        if not query_embedding:
            yield RetrievalStepEvent(
                key="embed_query", title="Embedding query",
                output="Failed to embed query; aborting retrieval.",
            )
            yield RetrievalStepEvent(key="done", title="Done", result=None)
            return
        yield RetrievalStepEvent(
            key="embed_query", title="Embedding query",
            output=f"Embedded query into a {len(query_embedding)}-dim vector.",
            data={"dimensions": len(query_embedding)},
        )

        community_results = self.retrieve_communities(query_embedding, config)
        community_scores = {c["community_id"]: c["score"]
                            for c in community_results}

        chunk_results = []
        center_node_ids = []
        if config.use_chunks:
            chunk_results = self.retrieve_chunks(
                query_embedding, community_results, config)
            center_node_ids += [c["center_node_id"] for c in chunk_results]
            yield RetrievalStepEvent(
                key="chunks", title="Searching chunks",
                output=(
                    f"Found {len(chunk_results)} matching chunks across "
                    f"{len({c['center_node_id'] for c in chunk_results})} distinct nodes."
                ),
                data={"results": chunk_results},
            )
        if config.use_communities:
            community_ids = [c["community_id"] for c in community_results]
            seed_ids = self.retrieve_nodes_from_communities(
                community_ids, config.community_expansion_limit)
            center_node_ids += seed_ids
            yield RetrievalStepEvent(
                key="communities", title="Searching communities",
                output=(
                    f"Found {len(community_results)} matching communities; "
                    f"expanded to {len(seed_ids)} seed nodes."
                ),
                data={"communities": community_results, "seed_node_ids": seed_ids},
            )

        generated_cypher = []
        cypher_results = []
        if config.use_nl2cypher and self.llm_fn is not None:
            nl_result = self.cypher_generator.query(
                query, self.llm_fn, config.nl2cypher_config,
                execute=True, mode=config.nl2cypher_mode,
            )
            generated_cypher = nl_result.cypher
            if config.nl2cypher_mode == "seed_nodes":
                for rows in nl_result.results:
                    center_node_ids += [
                        str(r["id"]) for r in rows if r.get("id") is not None
                    ]
            else:
                cypher_results = nl_result.results
            total_rows = sum(len(rows) for rows in nl_result.results)
            yield RetrievalStepEvent(
                key="nl2cypher", title="Generating Cypher",
                output=(
                    f"Generated {len(generated_cypher)} Cypher quer"
                    f"{'y' if len(generated_cypher) == 1 else 'ies'}; "
                    f"{total_rows} total rows returned."
                ),
                sub_events=[
                    RetrievalStepSubEvent(
                        title=f"Query {i + 1}",
                        output=f"{cypher}\n-> {len(rows)} rows",
                    )
                    for i, (cypher, rows) in enumerate(zip(generated_cypher, nl_result.results))
                ],
                data={"generated_cypher": generated_cypher, "results": nl_result.results},
            )
        elif config.use_nl2cypher:
            logger.warning(
                "use_nl2cypher is enabled but no llm_fn was configured on the client; skipping NL2Cypher."
            )

        if not center_node_ids and not cypher_results:
            yield RetrievalStepEvent(key="done", title="Done", result=None)
            return

        logger.info(
            f"Expanding subgraph from {len(center_node_ids)} seed nodes (max_hops={config.max_hops})"
        )
        subgraph = (
            self.fetch_subgraph(
                center_node_ids, config.max_hops, config.allowed_rel_types, config.denied_rel_types)
            if center_node_ids else []
        )

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
            logger.info("Retrieval complete: 0 nodes, 0 relationships")
            result = RetrievalResult(
                nodes=[], relationships=[],
                generated_cypher=generated_cypher, cypher_results=cypher_results,
            )
            yield RetrievalStepEvent(
                key="subgraph", title="Expanding subgraph",
                output="Expanded to 0 nodes, 0 relationships.",
                data={"node_count": 0, "relationship_count": 0, "nodes": [], "relationships": []},
            )
            yield RetrievalStepEvent(key="done", title="Done", output="Retrieval complete.", result=result)
            return

        logger.info(
            f"Retrieval complete: {len(subgraph[0]['nodes'])} nodes, "
            f"{len(subgraph[0]['relationships'])} relationships"
        )
        result = RetrievalResult(
            nodes=subgraph[0]["nodes"], relationships=subgraph[0]["relationships"],
            generated_cypher=generated_cypher, cypher_results=cypher_results,
        )
        yield RetrievalStepEvent(
            key="subgraph", title="Expanding subgraph",
            output=(
                f"Expanded from {len(center_node_ids)} seed nodes (max_hops={config.max_hops}) "
                f"to {len(result.nodes)} nodes, {len(result.relationships)} relationships."
            ),
            data={
                "node_count": len(result.nodes),
                "relationship_count": len(result.relationships),
                # Capped so a large subgraph doesn't balloon this event; the
                # full result is still available on the final "done" event.
                "nodes": [n.model_dump() for n in result.nodes[:25]],
                "relationships": [r.model_dump() for r in result.relationships[:25]],
            },
        )
        yield RetrievalStepEvent(key="done", title="Done", output="Retrieval complete.", result=result)

    # Mirrors iter_retrieve_graph() above — keep both in sync.
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

        generated_cypher = []
        cypher_results = []
        if config.use_nl2cypher and self.llm_fn is not None:
            nl_result = self.cypher_generator.query(
                query, self.llm_fn, config.nl2cypher_config,
                execute=True, mode=config.nl2cypher_mode,
            )
            generated_cypher = nl_result.cypher
            if config.nl2cypher_mode == "seed_nodes":
                for rows in nl_result.results:
                    center_node_ids += [
                        str(r["id"]) for r in rows if r.get("id") is not None
                    ]
            else:
                cypher_results = nl_result.results
        elif config.use_nl2cypher:
            logger.warning(
                "use_nl2cypher is enabled but no llm_fn was configured on the client; skipping NL2Cypher."
            )

        if not center_node_ids and not cypher_results:
            return None

        logger.info(
            f"Expanding subgraph from {len(center_node_ids)} seed nodes (max_hops={config.max_hops})"
        )
        subgraph = (
            self.fetch_subgraph(
                center_node_ids, config.max_hops, config.allowed_rel_types, config.denied_rel_types)
            if center_node_ids else []
        )

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
            logger.info("Retrieval complete: 0 nodes, 0 relationships")
            return RetrievalResult(
                nodes=[], relationships=[],
                generated_cypher=generated_cypher, cypher_results=cypher_results,
            )

        logger.info(
            f"Retrieval complete: {len(subgraph[0]['nodes'])} nodes, "
            f"{len(subgraph[0]['relationships'])} relationships"
        )
        return RetrievalResult(
            nodes=subgraph[0]["nodes"], relationships=subgraph[0]["relationships"],
            generated_cypher=generated_cypher, cypher_results=cypher_results,
        )
