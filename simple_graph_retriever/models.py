from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict


class CypherGenerationConfig(BaseModel):
    top_k_schema: int = 10
    max_queries: int = 3
    result_limit: int = 50


class NLQueryResult(BaseModel):
    cypher: List[str]
    results: List[List[Dict[str, Any]]]  # results[i] came from cypher[i]


class RetrievalConfig(BaseModel):
    use_communities: bool = True
    use_chunks: bool = True
    max_communities: int = 3
    max_chunks: int = 10
    max_hops: int = 1
    community_score_threshold: float = 0.4
    chunk_score_threshold: float = 0.4
    community_score_drop_off_pct: float = 0.07
    chunk_score_drop_off_pct: float = 0.2
    community_expansion_limit: int = 25
    allowed_rel_types: Optional[List[str]] = None
    denied_rel_types: Optional[List[str]] = None
    include_scores: bool = False
    use_nl2cypher: bool = True
    nl2cypher_mode: Literal["seed_nodes", "freeform"] = "seed_nodes"
    nl2cypher_config: Optional[CypherGenerationConfig] = None


class RetrievalNode(BaseModel):
    element_id: str
    community_id: Optional[int] = None
    label: Optional[str] = None
    uuid: Optional[str] = None
    # chunk_score: Optional[float] = None
    # community_score: Optional[float] = None

    model_config = ConfigDict(extra="allow")

    def __iter__(self):
        return iter(self.model_dump().items())


class RetrievalRelationship(BaseModel):
    element_id: str
    start_node_element_id: str
    end_node_element_id: str
    type: str

    model_config = ConfigDict(extra="allow")

    def __iter__(self):
        return iter(self.model_dump().items())


class RetrievalResult(BaseModel):
    nodes: List[RetrievalNode]
    relationships: List[RetrievalRelationship]
    generated_cypher: List[str] = []
    cypher_results: List[List[Dict[str, Any]]] = []  # cypher_results[i] came from generated_cypher[i]

    def __iter__(self):
        return iter(self.model_dump().items())

    def json(self, **kwargs) -> str:
        """
        Serialize the retrieval result to a JSON string.
        """
        return super().model_dump_json(**kwargs)

    def to_markdown(self) -> str:
        """
        Textualize the retrieval result in Markdown format.
        Only prints properties and relationships if they exist.
        """
        lines = ["# Nodes\n"]

        # Map node element_id to label for relationships
        node_label_map = {
            node.element_id: node.label or node.element_id for node in self.nodes
        }

        for node in self.nodes:
            data = node.model_dump()
            label = data.get("label") or node.element_id

            # Node header
            lines.append(f"## {label}\n")

            # Extra properties
            core_fields = {
                "element_id",
                "label",
                "uuid",
                "chunk_score",
                "community_score",
                "community_id",
            }
            extra_properties = {k: v for k, v in data.items() if k not in core_fields}
            if extra_properties:
                lines.append("**Properties:**")
                for k, v in extra_properties.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")  # empty line for spacing

            # Relationships
            outgoing_rels = [
                rel
                for rel in self.relationships
                if rel.start_node_element_id == node.element_id
            ]
            if outgoing_rels:
                lines.append("**Relationships:**")
                for rel in outgoing_rels:
                    target_label = node_label_map.get(
                        rel.end_node_element_id, rel.end_node_element_id
                    )
                    lines.append(f"- {rel.type} => {target_label}")
                lines.append("")  # empty line for spacing

        return "\n".join(lines)
