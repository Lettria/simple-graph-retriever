import sys
import os

from simple_graph_retriever.client import GraphRetrievalClient

if __name__ == "__main__":
    print("Starting graph indexing example...")
    client = None
    try:
        # Initialize the client (settings loaded from .env or defaults)
        client = GraphRetrievalClient()

        # Run the full indexing pipeline
        print(
            "Running client.index()... (This will detect communities, create chunks, and index them)"
        )
        client.clear_index()

        print("✅ Graph indexing complete.")

    except Exception as e:
        print(f"❌ An error occurred during indexing: {e}")
    finally:
        if client:
            client.close()
            print("Client closed.")
