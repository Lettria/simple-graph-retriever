from typing import List, Optional
from pydantic import BaseModel


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
