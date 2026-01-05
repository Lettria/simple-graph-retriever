from simple_graph_retriever.client import GraphRetrievalClient
from simple_graph_retriever.models import RetrievalConfig
import json

if __name__ == "__main__":
    client = GraphRetrievalClient()
    try:
        query = "What are the health benefits of regular exercise?"

        print(f"Retrieving subgraph for query: '{query}'")

        print("\n--- Full GraphRAG mode ---")
        full_subgraph = client.retriever.retrieve_graph(
            query,
            RetrievalConfig(
                use_communities=True,
                use_chunks=True,
            ),
        )
        if full_subgraph:
            print(
                f"Retrieved {len(full_subgraph.nodes)} nodes and {len(full_subgraph.relationships)} relationships."
            )
            with open("retrieved_subgraph_full.json", "w") as f:
                f.write(full_subgraph.json(indent=2))
            with open("retrieved_subgraph_full.md", "w") as f:
                f.write(full_subgraph.to_markdown())
        else:
            print("No results found.")

        print("\n--- Chunk-only mode ---")
        chunk_only_subgraph = client.retriever.retrieve_graph(
            query, RetrievalConfig(use_communities=False, use_chunks=True)
        )
        if chunk_only_subgraph:
            print(
                f"Retrieved {len(chunk_only_subgraph.nodes)} nodes and {len(chunk_only_subgraph.relationships)} relationships."
            )
            with open("retrieved_subgraph_chunk_only.json", "w") as f:
                f.write(chunk_only_subgraph.json(indent=2))
        else:
            print("No results found.")

        print("\n--- Community-only mode ---")
        community_only_subgraph = client.retriever.retrieve_graph(
            query, RetrievalConfig(use_communities=True, use_chunks=False)
        )
        if community_only_subgraph:
            print(
                f"Retrieved {len(community_only_subgraph.nodes)} nodes and {len(community_only_subgraph.relationships)} relationships."
            )
            with open("retrieved_subgraph_community_only.json", "w") as f:
                f.write(community_only_subgraph.json(indent=2))
        else:
            print("No results found.")
    finally:
        client.close()
