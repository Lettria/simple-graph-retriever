# Simple Graph Retriever

A Python SDK for indexing a Neo4j graph into a vector database (Qdrant) and performing retrieval-augmented generation (RAG) tasks against it.

This SDK handles the heavy lifting of graph processing, including:

- **Community Detection**: Uses the Leiden algorithm to identify communities within your graph structure.
- **Chunking**: Breaks down nodes and their local context into "chunks" suitable for embedding.
- **Vector Indexing**: Embeds and indexes both community summaries and individual chunks into Qdrant.
- **Graph Retrieval**: Retrieves relevant subgraphs based on a natural language query, with fine-grained control over the expansion process.

---

## Table of Contents

- [Simple Graph Retriever](#simple-graph-retriever)
  - [Table of Contents](#table-of-contents)
  - [1. Installation](#1-installation)
  - [2. Configuration](#2-configuration)
  - [3. Usage](#3-usage)
    - [Initializing the Client](#initializing-the-client)
    - [Indexing the Graph](#indexing-the-graph)
    - [Retrieving a Subgraph](#retrieving-a-subgraph)
      - [Basic Retrieval](#basic-retrieval)
      - [Advanced Retrieval with Graph Expansion Control](#advanced-retrieval-with-graph-expansion-control)
  - [4. Project Structure](#4-project-structure)
  - [5. Contributing](#5-contributing)
  - [6. License](#6-license)

---

## 1. Installation

To install the SDK, clone this repository and use pip to install it in editable mode, which is recommended for development.

```bash
pip install simple_graph_retriever==1.0.0-rc.11
```

## 2. Configuration

The SDK is configured via environment variables, which are loaded from a `.env` file in your project's root directory.

| Variable                        | Default Value           | Description                                                                                       |
| ------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------- |
| `NEO4J_URI`                     | `bolt://localhost:7687` | The URI for your Neo4j database.                                                                  |
| `NEO4J_USER`                    | `neo4j`                 | The username for your Neo4j database.                                                             |
| `NEO4J_PASSWORD`                | `password`              | The password for your Neo4j database.                                                             |
| `QDRANT_URL`                    | `http://localhost:6333` | The URL for your Qdrant vector database instance.                                                 |
| `QDRANT_API_KEY`                | `None`                  | Optional: The API key for authenticating with your Qdrant instance.                               |
| `QDRANT_CHUNKS_COLLECTION`      | `chunks`                | The name of the collection for storing graph chunks.                                              |
| `QDRANT_COMMUNITIES_COLLECTION` | `communities`           | The name of the collection for storing community summaries.                                       |
| `EMBEDDER_URL`                  | `http://localhost:8080` | The URL of the text embedding service. It must accept a POST request with `{"inputs": ["text"]}`. |
| `VECTOR_SIZE`                   | `384`                   | The dimension of the vectors produced by your embedding model.                                    |
| `LOGLEVEL`                      | `INFO`                  | The logging level for the SDK (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                              |

**Example `.env` file:**

```dotenv
# Neo4j Database Credentials
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_secure_password"

# Qdrant Vector Database URL
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="your_qdrant_api_key" # Optional
QDRANT_CHUNKS_COLLECTION="chunks"
QDRANT_COMMUNITIES_COLLECTION="communities"

# Text Embedding Model URL
EMBEDDER_URL="http://localhost:8080/embed"
```

## 3. Usage

### Initializing the Client

The main entry point is the `GraphRetrievalClient`. It automatically loads settings from your `.env` file.

```python
from graph_retrieval_sdk.client import GraphRetrievalClient

# Initialize the client
client = GraphRetrievalClient()
```

### Indexing the Graph

The `index()` method runs the full, idempotent pipeline to populate Qdrant with your graph data.

```python
# This runs the full pipeline:
# 1. Community detection
# 2. Chunk creation
# 3. Chunk and Community indexing
client.index()

print("Indexing complete!")
```

To completely reset your vector index, use the `clear_index()` method. **Warning**: This is a destructive operation that deletes and recreates the Qdrant collections.

```python
client.clear_index()
```

### Retrieving a Subgraph

Use the `retrieve_graph()` method to query your indexed graph. The `RetrievalConfig` model allows for fine-grained control over the process.

#### Basic Retrieval

```python
from graph_retrieval_sdk.models import RetrievalConfig
import json

query = "Who was the emperor of Rome?"
config = RetrievalConfig()

subgraph = client.retriever.retrieve_graph(query, config)

if subgraph:
    print(f"Retrieved {len(subgraph[0]['nodes'])} nodes and {len(subgraph[0]['relationships'])} relationships.")
```

#### Advanced Retrieval with Graph Expansion Control

Customize the `RetrievalConfig` to tune the retrieval and expansion behavior.

```python
from graph_retrieval_sdk.models import RetrievalConfig

query = "Tell me about the Roman emperors and their families, but exclude servants."

advanced_config = RetrievalConfig(
    # --- Retrieval Settings ---
    max_communities=5,
    max_chunks=15,

    # --- Graph Expansion Settings ---
    max_hops=2,
    community_expansion_limit=5,

    # --- Relationship Filtering ---
    allowed_rel_types=["HAS_SON", "MARRIED_TO", "SUCCESSOR_OF"],
    denied_rel_types=["HAS_SERVANT"]
)

subgraph = client.retriever.retrieve_graph(query, advanced_config)

if subgraph:
    print(f"Retrieved {len(subgraph[0]['nodes'])} nodes with advanced configuration.")

# Close the client connection when done
client.close()
```

## 4. Project Structure

The project is organized as a standard Python package:

```
├── graph_retrieval_sdk/
│   ├── __init__.py
│   ├── client.py         # Main client class
│   ├── config.py         # Configuration and logging setup
│   ├── indexer.py        # Logic for indexing the graph
│   ├── retriever.py      # Logic for retrieving subgraphs
│   └── models.py         # Pydantic models (e.g., RetrievalConfig)
├── examples/
│   ├── retrieve_graph.py
│   └── run_indexing.py
├── pyproject.toml        # Project metadata and dependencies
└── README.md             # This file
```

## 5. Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 6. License

This project is licensed under the MIT License.
