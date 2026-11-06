# Simple Graph Retriever

A Python SDK for indexing a graph database (**Neo4j** or **FalkorDB**) into a vector database (Qdrant) and performing retrieval-augmented generation (RAG) tasks against it.

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
    - [Common Configuration](#common-configuration)
    - [Neo4j Specific Configuration](#neo4j-specific-configuration)
    - [FalkorDB Specific Configuration](#falkordb-specific-configuration)
    - [Example `.env` files](#example-env-files)
  - [3. Usage](#3-usage)
    - [Initializing the Client](#initializing-the-client)
    - [Indexing the Graph](#indexing-the-graph)
    - [Retrieving a Subgraph](#retrieving-a-subgraph)
      - [Basic Retrieval](#basic-retrieval)
      - [Advanced Retrieval with Graph Expansion Control](#advanced-retrieval-with-graph-expansion-control)
    - [Natural Language to Cypher](#natural-language-to-cypher)
    - [Streaming Retrieval Progress](#streaming-retrieval-progress)
  - [4. Project Structure](#4-project-structure)
  - [5. Contributing](#5-contributing)
  - [6. License](#6-license)

---

## 1. Installation

To install the SDK, clone this repository and use pip. You can choose which database driver to install using optional dependencies.

**Install with Neo4j support:**

```bash
pip install simple_graph_retriever[neo4j]==1.0.0-rc.23
```

**Install with FalkorDB support:**

```bash
pip install simple_graph_retriever[falkordb]==1.0.0-rc.23
```

**Install with support for both:**

```bash
pip install simple_graph_retriever[all]==1.0.0-rc.23
```

## 2. Configuration

The SDK is configured via environment variables, which are loaded from a `.env` file in your project's root directory. You must configure **either** Neo4j **or** FalkorDB.

### Common Configuration

| Variable                        | Default Value           | Description                                                                                       |
| ------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------- |
| `GRAPH_DATABASE_TYPE`           | `neo4j`                 | The type of database to use: `neo4j` or `falkordb`.                                               |
| `QDRANT_URL`                    | `http://localhost:6333` | The URL for your Qdrant vector database instance.                                                 |
| `QDRANT_API_KEY`                | `None`                  | Optional: The API key for authenticating with your Qdrant instance.                               |
| `QDRANT_CHUNKS_COLLECTION`      | `chunks`                | The name of the collection for storing graph chunks.                                              |
| `QDRANT_COMMUNITIES_COLLECTION` | `communities`           | The name of the collection for storing community summaries.                                       |
| `QDRANT_SCHEMA_COLLECTION`      | `schema`                | The name of the collection for storing discovered graph schema elements (for NL-to-Cypher).        |
| `VECTOR_SIZE`                   | `384`                   | The dimension of the vectors produced by your embedding model — must match what `embed_fn` returns.|
| `LOGLEVEL`                      | `INFO`                  | The logging level for the SDK (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                              |

### Neo4j Specific Configuration

| Variable         | Default Value           | Description                           |
| ---------------- | ----------------------- | ------------------------------------- |
| `NEO4J_URI`      | `bolt://localhost:7687` | The URI for your Neo4j database.      |
| `NEO4J_USER`     | `neo4j`                 | The username for your Neo4j database. |
| `NEO4J_PASSWORD` | `password`              | The password for your Neo4j database. |

### FalkorDB Specific Configuration

| Variable            | Default Value | Description                            |
| ------------------- | ------------- | -------------------------------------- |
| `FALKORDB_HOST`     | `localhost`   | The host for your FalkorDB instance.   |
| `FALKORDB_PORT`     | `6379`        | The port for your FalkorDB instance.   |
| `FALKORDB_PASSWORD` | `None`        | Optional: Password for Redis/FalkorDB. |
| `FALKORDB_KEY`      | `graph`       | The key (graph name) used in FalkorDB. |

---

### Example `.env` files

**Option A: Using Neo4j**

```dotenv
GRAPH_DATABASE_TYPE="neo4j"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_secure_password"

QDRANT_URL="http://localhost:6333"
```

**Option B: Using FalkorDB**

```dotenv
GRAPH_DATABASE_TYPE="falkordb"
FALKORDB_HOST="localhost"
FALKORDB_PORT="6379"

QDRANT_URL="http://localhost:6333"
```

## 3. Usage

### Initializing the Client

The main entry point is the `GraphRetrievalClient`. It automatically loads settings from your `.env` file and initializes the appropriate database driver (Neo4j or FalkorDB) based on your configuration.

The SDK doesn't bundle an embedding provider either: you supply `embed_fn`, a callback that takes a list of strings and returns a list of vectors (same order/length), so you can plug in Azure OpenAI, OpenAI, a local model, TEI, etc. — same pattern as `llm_fn` for NL2Cypher.

```python
from simple_graph_retriever.client import GraphRetrievalClient
from openai import AzureOpenAI

azure_client = AzureOpenAI(
    azure_endpoint="https://<your-resource>.openai.azure.com",
    api_key="...",
    api_version="2024-02-01",
)

def embed_fn(texts: list[str]) -> list[list[float]]:
    resp = azure_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [d.embedding for d in resp.data]

# Initialize the client (backend determined by env vars)
# VECTOR_SIZE must match embed_fn's output dimension (1536 for text-embedding-3-small).
client = GraphRetrievalClient(embed_fn=embed_fn)
```

### Indexing the Graph

The `index()` method runs the full, idempotent pipeline to populate Qdrant with data from your connected graph database.

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

### Natural Language to Cypher

`client.index()` also runs a schema discovery pass: it inspects the connected graph's node labels, relationship types, and properties, and indexes a short description of each into a dedicated `schema` Qdrant collection. This lets the LLM see only the relevant slice of the schema, even on large graphs.

The SDK doesn't bundle an LLM client; you provide a callback (`llm_fn`) that takes the assembled prompt and returns the model's raw text response, so you can plug in Gemini, OpenAI, a local model, etc. Pass it once, to the client:

```python
from google import genai

genai_client = genai.Client()

def llm_fn(prompt: str) -> str:
    return genai_client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    ).text

client = GraphRetrievalClient(embed_fn=embed_fn, llm_fn=llm_fn)
```

**It's on by default inside `retrieve_graph()`.** With an `llm_fn` configured, every `retrieve_graph()` call also asks the LLM to find nodes matching the question directly (via the discovered schema), and merges those into the same subgraph as the vector-search results — no separate call needed. The generated Cypher is logged at `INFO` level. If no `llm_fn` is configured, this step is skipped (with a warning) and behavior is unchanged from before this feature existed.

This is controlled by two `RetrievalConfig` fields:

```python
config = RetrievalConfig(
    use_nl2cypher=True,       # default: True: set False to disable entirely
    nl2cypher_mode="seed_nodes",  # default: "seed_nodes" | "freeform"
)
```

- `"seed_nodes"` (default): the LLM is constrained to return only matching node identifiers. These are merged with the vector-search seeds and expanded through the normal hop-traversal — the result stays a single, unified `nodes`/`relationships` subgraph.
- `"freeform"`: the LLM can write any Cypher (aggregates, specific properties, etc.). Its raw rows are returned separately in `result.cypher_results`, alongside (not merged into) `result.nodes`/`result.relationships`.

The LLM may answer with **more than one** Cypher query (e.g. if the question has several distinct aspects), or with none at all if the retrieved schema genuinely isn't relevant to the question. So `result.generated_cypher` is a `List[str]` and, in `"freeform"` mode, `result.cypher_results` is a `List[List[Dict]]` — `cypher_results[i]` holds the rows for `generated_cypher[i]`. Both can be empty lists. This is tunable via `CypherGenerationConfig`:

```python
config = RetrievalConfig(
    nl2cypher_config=CypherGenerationConfig(
        max_queries=3,     # default: at most this many queries per call
        result_limit=50,   # default: a LIMIT is enforced on every query, even if the LLM forgets one
    ),
)
```

You can also call it directly, bypassing `retrieve_graph()` entirely:

```python
result = client.query_natural_language("Who was the emperor of Rome?")  # uses the client's llm_fn

print(result.cypher)   # List[str] — the generated Cypher queries
print(result.results)  # List[List[Dict]] — results[i] came from cypher[i]
```

### Streaming Retrieval Progress

`client.iter_retrieve_graph(query, config)` runs the exact same pipeline as `retrieve_graph()`, but returns a generator of `RetrievalStepEvent` objects instead of only the final result — useful for showing live progress in a UI. Only phases that actually ran emit an event; the last event always has `key="done"` and carries the same result `retrieve_graph()` would have returned. Each event's `output` is a short human-readable summary, while `data` carries the step's raw structured output (e.g. the matched community/chunk records, the generated Cypher and its rows, or a sample of the expanded subgraph's nodes/relationships) for a UI that wants to show more than just the summary line.

```python
for event in client.iter_retrieve_graph(query, RetrievalConfig()):
    print(f"[{event.key}] {event.title}: {event.output}")
    print(f"    data: {event.data}")
    for sub in event.sub_events:
        print(f"    - {sub.title}: {sub.output}")
    if event.key == "done":
        result = event.result  # same RetrievalResult (or None) retrieve_graph() returns
```

See `examples/retrieve_stream.py` for a full runnable example, including the NL2Cypher `sub_events` case.

## 4. Project Structure

The project is organized as a standard Python package:

```
├── graph_retrieval_sdk/
│   ├── __init__.py
│   ├── client.py         # Main client class
│   ├── config.py         # Configuration and logging setup
│   ├── indexer.py        # Logic for indexing the graph
│   ├── retriever.py      # Logic for retrieving subgraphs
│   ├── db/               # Database adapters
│   │   ├── neo4j.py
│   │   └── falkordb.py
│   └── models.py         # Pydantic models (e.g., RetrievalConfig)
├── examples/
├── pyproject.toml        # Project metadata and dependencies
└── README.md             # This file
```

## 5. Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 6. License

This project is licensed under the MIT License.
