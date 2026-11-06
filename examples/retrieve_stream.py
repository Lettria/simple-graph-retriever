from simple_graph_retriever.client import GraphRetrievalClient
from simple_graph_retriever.models import RetrievalConfig
from tei_embedder import make_tei_embed_fn


def print_events(events):
    result = None
    for event in events:
        print(f"[{event.key}] {event.title}")
        if event.output:
            print(f"    {event.output}")
        if event.data is not None:
            print(f"    data: {event.data}")
        for sub in event.sub_events:
            print(f"    - {sub.title}: {sub.output}")
        if event.key == "done":
            result = event.result
    return result


if __name__ == "__main__":
    query = "What are the health benefits of regular exercise?"

    print("--- Streaming retrieval (no llm_fn, so nl2cypher is skipped) ---")
    client = GraphRetrievalClient(embed_fn=make_tei_embed_fn())
    try:
        print(f"Streaming retrieval steps for query: '{query}'\n")
        result = print_events(client.iter_retrieve_graph(query, RetrievalConfig()))
        if result:
            print(
                f"\nFinal result: {len(result.nodes)} nodes, "
                f"{len(result.relationships)} relationships."
            )
        else:
            print("\nNo results found.")
    finally:
        client.close()

    # Plug in any LLM provider here (Gemini, OpenAI, a local model, ...).
    # See examples/nl2cypher.py for the full stub_llm_fn pattern.
    def stub_llm_fn(prompt: str) -> str:
        raise NotImplementedError(
            "Replace stub_llm_fn with a real LLM call to see the nl2cypher "
            "step and its per-query sub_events populated."
        )

    print("\n--- Streaming retrieval (with llm_fn, nl2cypher enabled) ---")
    client = GraphRetrievalClient(embed_fn=make_tei_embed_fn(), llm_fn=stub_llm_fn)
    try:
        print(f"Streaming retrieval steps for query: '{query}'\n")
        try:
            print_events(client.iter_retrieve_graph(query, RetrievalConfig()))
        except NotImplementedError as e:
            print(f"(stopped at nl2cypher step: {e})")
    finally:
        client.close()
