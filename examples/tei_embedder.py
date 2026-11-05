"""
Example helper, not part of the SDK: the SDK no longer talks to any
embedding provider itself, so each example here supplies its own `embed_fn`.
This one calls the local TEI (Text Embeddings Inference) service from
docker-compose.yaml. Swap this out for Azure OpenAI, OpenAI, etc. in your
own code — see the README's "Initializing the Client" section.
"""
import os
from typing import Callable, List

import requests


def make_tei_embed_fn(
    embedder_url: str = None, batch_size: int = 32, max_chars: int = 500
) -> Callable[[List[str]], List[List[float]]]:
    """
    TEI rejects a request that's too large (HTTP 413) either because it
    has too many inputs at once (batch_size caps this at TEI's default
    max-client-batch-size, 32) or because the inputs themselves are too
    long (max_chars truncates each one, matching the SDK's old built-in
    behavior before embedding became caller-supplied).
    """
    url = embedder_url or os.getenv("EMBEDDER_URL", "http://localhost:8080")

    def embed_fn(texts: List[str]) -> List[List[float]]:
        truncated = [t[:max_chars] for t in texts]
        embeddings = []
        for i in range(0, len(truncated), batch_size):
            chunk = truncated[i : i + batch_size]
            resp = requests.post(url, json={"inputs": chunk}, timeout=30)
            resp.raise_for_status()
            embeddings.extend(resp.json())
        return embeddings

    return embed_fn
