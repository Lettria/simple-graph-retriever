import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Literal, Optional
from qdrant_client import QdrantClient
from .models import CypherGenerationConfig, NLQueryResult
from .db_adapter import GraphAdapter
from .config import logger
from .embedding import embed_text

FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)
LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
NONE_RE = re.compile(r"^\s*NONE\s*$", re.IGNORECASE)


class CypherGenerator:
    def __init__(
        self,
        db_adapter: GraphAdapter,
        qdrant_url: str,
        embedder_url: str,
        qdrant_api_key: Optional[str] = None,
        schema_collection: str = "schema",
    ):
        self.db = db_adapter
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder_url = embedder_url
        self.schema_collection = schema_collection

    def _retrieve_schema_context(self, query_embedding: list[float], top_k: int) -> str:
        results = self.qdrant_client.search(
            collection_name=self.schema_collection,
            query_vector=query_embedding,
            limit=top_k,
        )
        return "\n\n".join(r.payload["text"] for r in results)

    def _build_prompt(
        self,
        query: str,
        schema_context: str,
        mode: Literal["seed_nodes", "freeform"],
        max_queries: int,
        result_limit: int,
    ) -> str:
        if mode == "seed_nodes":
            id_fn = self.db.id_function
            mode_rule = (
                f"- Each query must return ONLY a single column named `id`, containing "
                f"toString({id_fn}(n)) for every node that matches. "
                f"Do not return properties, aggregates, or anything else."
            )
        else:
            mode_rule = "- Each query should return whatever columns best answer that part of the question."

        return f"""You are an expert at writing Cypher queries for a property graph database.

Below is the relevant subset of the graph's schema (node labels, relationship types, and their properties):

{schema_context}

Rules:
- Only use the node labels, relationship types, and properties listed above.
- Do not use APOC procedures.
- If none of the schema above is actually relevant to the question, output exactly `NONE` and nothing else.
- You may write up to {max_queries} separate Cypher queries if the question has multiple distinct aspects that are better answered independently — separate each query with a single blank line. Otherwise, write just one.
{mode_rule}
- Each query must end with a `LIMIT {result_limit}` clause (or fewer) to keep it fast.
- Return ONLY the Cypher query/queries. No explanation, no markdown code fences, no numbering.

Question: {query}
Cypher query:"""

    @staticmethod
    def _split_queries(raw: str, max_queries: int) -> List[str]:
        text = FENCE_LINE_RE.sub("", raw).strip()
        if NONE_RE.match(text):
            return []
        pieces = re.split(r"\n\s*\n", text)
        queries = [p.strip() for p in pieces if p.strip()]
        return queries[:max_queries]

    @staticmethod
    def _ensure_limit(cypher: str, result_limit: int) -> str:
        cypher = cypher.strip().rstrip(";").strip()
        if LIMIT_RE.search(cypher):
            return cypher
        return f"{cypher}\nLIMIT {result_limit}"

    def generate_cypher(
        self,
        query: str,
        llm_fn: Callable[[str], str],
        config: Optional[CypherGenerationConfig] = None,
        mode: Literal["seed_nodes", "freeform"] = "freeform",
    ) -> List[str]:
        config = config or CypherGenerationConfig()
        query_embedding = embed_text(self.embedder_url, query)
        schema_context = self._retrieve_schema_context(
            query_embedding, config.top_k_schema
        )
        prompt = self._build_prompt(
            query, schema_context, mode, config.max_queries, config.result_limit
        )
        raw = llm_fn(prompt)
        queries = self._split_queries(raw, config.max_queries)
        return [self._ensure_limit(q, config.result_limit) for q in queries]

    def query(
        self,
        query: str,
        llm_fn: Callable[[str], str],
        config: Optional[CypherGenerationConfig] = None,
        execute: bool = True,
        mode: Literal["seed_nodes", "freeform"] = "freeform",
    ) -> NLQueryResult:
        config = config or CypherGenerationConfig()
        cyphers = self.generate_cypher(query, llm_fn, config, mode=mode)
        if not execute or not cyphers:
            return NLQueryResult(cypher=cyphers, results=[[] for _ in cyphers])

        def run_one(indexed_cypher):
            i, cypher = indexed_cypher
            logger.info(
                f"Running generated Cypher query {i + 1}/{len(cyphers)} (mode={mode}):\n{cypher}"
            )
            try:
                return self.db.query(cypher)
            except Exception as e:
                logger.error(f"Generated Cypher failed: {cypher}\nError: {e}")
                return []

        with ThreadPoolExecutor(max_workers=len(cyphers)) as executor:
            # map preserves input order, so results[i] still corresponds to cyphers[i]
            results = list(executor.map(run_one, enumerate(cyphers)))

        return NLQueryResult(cypher=cyphers, results=results)
