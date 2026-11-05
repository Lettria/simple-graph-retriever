from simple_graph_retriever.client import GraphRetrievalClient
from simple_graph_retriever.models import RetrievalConfig

# Plug in any LLM provider here (Gemini, OpenAI, a local model, ...).
# The SDK only needs a callable that takes the assembled prompt string
# and returns the raw text response. See perseus-client's report.py for
# an example that wires in `google.genai`.
def stub_llm_fn(prompt: str) -> str:
    print("\n--- Prompt sent to the LLM ---")
    print(prompt)
    print("--- end prompt ---\n")
    raise NotImplementedError(
        "Replace stub_llm_fn with a real LLM call, e.g.:\n"
        "    from google import genai\n"
        "    client = genai.Client()\n"
        "    def llm_fn(prompt):\n"
        "        return client.models.generate_content(\n"
        "            model='gemini-3-flash-preview', contents=prompt\n"
        "        ).text\n"
    )


if __name__ == "__main__":
    # Passing llm_fn here also activates NL2Cypher by default inside
    # client.retrieve_graph() (RetrievalConfig.use_nl2cypher=True).
    client = GraphRetrievalClient(llm_fn=stub_llm_fn)
    try:
        # discover_schema() already ran as part of client.index(); if you
        # haven't indexed yet, run examples/index.py first.
        query = "What are the health benefits of regular exercise?"

        print("--- Standalone query_natural_language() ---")
        result = client.query_natural_language(query)
        print(f"Generated {len(result.cypher)} Cypher quer{'y' if len(result.cypher) == 1 else 'ies'}:")
        for cypher, rows in zip(result.cypher, result.results):
            print(f"{cypher}\n-> {len(rows)} rows")

        print("\n--- Same thing, folded into retrieve_graph() (seed_nodes mode, default) ---")
        subgraph = client.retrieve_graph(query, RetrievalConfig())
        if subgraph:
            print(f"Generated Cypher:\n{subgraph.generated_cypher}\n")
            print(f"Nodes: {len(subgraph.nodes)}, Relationships: {len(subgraph.relationships)}")

        print("\n--- freeform mode: raw rows, not merged into the subgraph ---")
        subgraph = client.retrieve_graph(
            query, RetrievalConfig(nl2cypher_mode="freeform")
        )
        if subgraph:
            print(f"Generated Cypher:\n{subgraph.generated_cypher}\n")
            print(f"Cypher rows: {subgraph.cypher_results}")
    finally:
        client.close()
