"""The community step of the indexer, against a fake graph, Qdrant and embedder."""

from simple_graph_retriever.community_summary import SummaryLimits
from simple_graph_retriever.indexer import COMMUNITY_SEED, GraphIndexer


class FakeDB:
    id_function = "elementId"

    def __init__(self, communities, nodes):
        # communities: id -> key elements text; nodes: id -> rows of the node query
        self.pending = dict(communities)
        self.nodes = nodes
        self.marked = {}

    def query(self, query, params=None):
        params = params or {}
        if "c.indexed IS NULL" in query:
            ids = sorted(self.pending)[: params["batch_size"]]
            return [{"id": i, "elements": self.pending[i]} for i in ids]
        if "(n)<-[r]-(m)" in query:
            return self.nodes[params["cid"]]
        if "SET c.indexed = true" in query:
            self.pending.pop(params["id"], None)
            self.marked[params["id"]] = params
            return []
        raise AssertionError(f"unexpected query: {query[:60]}")


class FakeQdrant:
    def __init__(self):
        self.points = []

    def upsert(self, collection_name, points):
        self.points.extend(points)


def row(name, label="Drug", rels=(), props=None):
    """A row of the node query; rels are (type, target label, target name, dir)."""
    return {
        "labels": [label, "_CommunityMember"],
        "name": name,
        "props": props or {},
        "rels": [{"type": t, "label": lab, "target": x, "dir": d} for t, lab, x, d in rels],
        "uuid": name,
    }


def build(communities, nodes, summarize_fn=None, embed=None, limits=SummaryLimits()):
    embedded = []

    def embed_fn(texts):
        embedded.extend(texts)
        return embed(texts) if embed else [[float(len(t))] for t in texts]

    indexer = GraphIndexer(
        db_adapter=FakeDB(communities, nodes),
        qdrant_url="http://localhost:1",
        embed_fn=embed_fn,
        summarize_fn=summarize_fn,
        summary_limits=limits,
    )
    indexer.qdrant = FakeQdrant()
    return indexer, embedded


COMMUNITIES = {1: "Key Elements:\nDrug:Keytruda\n", 2: "Key Elements:\nDrug:Opdivo\n"}
NODES = {
    1: [
        row("Keytruda", rels=[("TREATS", "Condition", "Melanoma", "out")], props={"class": "PD-1"}),
        row("Melanoma", "Condition", rels=[("TREATS", "Drug", "Keytruda", "in")]),
    ],
    2: [row("Opdivo")],
}


def test_the_summary_is_what_is_embedded_and_stored():
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return "Summary of community."

    indexer, embedded = build(COMMUNITIES, NODES, summarize)
    indexer.index_communities()

    assert embedded == ["Summary of community.", "Summary of community."]
    assert [p.payload for p in indexer.qdrant.points] == [
        {"summary": "Summary of community.", "community_id": 1, "type": "community"},
        {"summary": "Summary of community.", "community_id": 2, "type": "community"},
    ]
    assert indexer.db.marked[1] == {"id": 1, "summary": "Summary of community.", "summarized": True}
    # All the community's nodes were given to the model, with their relationships.
    assert 'Drug "Keytruda" {class: PD-1}; -TREATS-> Condition "Melanoma"' in prompts[0]
    # ... with their properties, and each relationship on both ends, with its direction.
    assert 'Condition "Melanoma"; <-TREATS- Drug "Keytruda"' in prompts[0]


def test_text_is_no_longer_written():
    indexer, _ = build(COMMUNITIES, NODES, lambda p: "S.")
    indexer.index_communities()

    assert all("text" not in p.payload for p in indexer.qdrant.points)
    assert all(set(m) == {"id", "summary", "summarized"} for m in indexer.db.marked.values())


def test_without_an_llm_the_summary_is_the_key_elements():
    indexer, embedded = build(COMMUNITIES, NODES, summarize_fn=None)
    indexer.index_communities()

    assert embedded == list(COMMUNITIES.values())
    assert indexer.db.marked[1]["summary"] == COMMUNITIES[1]
    assert indexer.db.marked[1]["summarized"] is False


def test_a_failed_call_falls_back_for_that_community_only():
    def summarize(prompt):
        if "Opdivo" in prompt:
            raise RuntimeError("throttled")
        return "Summary of the first."

    indexer, _ = build(COMMUNITIES, NODES, summarize)
    indexer.index_communities()

    assert indexer.db.marked[1] == {"id": 1, "summary": "Summary of the first.", "summarized": True}
    assert indexer.db.marked[2] == {"id": 2, "summary": COMMUNITIES[2], "summarized": False}


def test_a_community_whose_vector_is_empty_is_still_marked_so_the_loop_ends():
    indexer, _ = build(
        {1: "Key Elements:\n"}, {1: [row("A")]}, lambda p: "S.", embed=lambda texts: [[] for _ in texts]
    )

    indexer.index_communities()

    assert 1 in indexer.db.marked and not indexer.db.pending
    assert indexer.qdrant.points == []


def test_indexes_every_batch_of_communities():
    communities = {i: f"Key Elements:\nDrug:D{i}\n" for i in range(1, 8)}
    nodes = {i: [row(f"D{i}")] for i in communities}
    indexer, _ = build(communities, nodes, lambda p: "S.")
    indexer.vector_batch_size = 3

    indexer.index_communities()

    assert sorted(indexer.db.marked) == list(communities)
    assert len(indexer.qdrant.points) == 7


def test_the_run_budget_cuts_the_communities_that_come_after_it_is_used_up():
    big = [row(f"Drug{i:03d}") for i in range(100)]
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return "S."

    limits = SummaryLimits(max_chars_per_run=300, min_chars_per_community=50)
    indexer, _ = build(
        {1: "Key Elements:\n", 2: "Key Elements:\n"}, {1: [row("Small")], 2: big}, summarize, limits=limits
    )
    indexer.index_communities()

    assert 'Drug "Small"' in prompts[0]
    # The large community (indexed last) is cut to its best-connected nodes.
    assert 0 < prompts[1].count("\nDrug") < 100


def test_communities_are_taken_smallest_first():
    assert "ORDER BY c.size, c.id" in GraphIndexer._COMMUNITIES_QUERY


def test_the_leiden_seed_is_fixed():
    assert isinstance(COMMUNITY_SEED, int)


def test_the_node_query_reads_both_directions_and_the_other_entitys_label():
    query = GraphIndexer._COMMUNITY_NODES_QUERY

    assert "(n)-[r]->(m)" in query and "(n)<-[r]-(m)" in query
    assert "head([l IN labels(m)" in query
