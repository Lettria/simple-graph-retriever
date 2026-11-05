import requests
from .config import logger


def embed_text(embedder_url: str, text: str) -> list[float]:
    """
    Embeds a single string using the TEI embedder.
    """
    try:
        resp = requests.post(embedder_url, json={"inputs": text}, timeout=5)
        resp.raise_for_status()
        return resp.json()[0]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []
