"""Summaries of communities, written by an LLM from the community's own nodes.

A community's summary says what the community comprises. It is built from
*every* node of the community, as compact lines (labels, name, a few
properties, outgoing relationships), so it is not limited to the best-connected
nodes. A community too large for one prompt is summarized in batches, and the
batches' summaries are summarized in turn (map-reduce) until one is left.

Nothing here touches a database or a model: the caller supplies the lines and a
`summarize_fn(prompt) -> text`. A failure of that function is not an error of
the index: the community gets no summary from this module (`None`), and the
indexer falls back to the list of its key elements.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("simple_graph_retriever")

SUMMARY_PROMPT = """You are a Strict Logic-Based Data knowledge graph analyst. Your task is to analyze a list of entities and their relationships that belong to the same community and produce a coherent summary of what this community comprises.

STRICT RULES:
1. Zero External Knowledge: You are forbidden from using any names or facts that do not appear in the text below.
2. Adjectives: Do not add descriptive words like 'major' or 'important'.
3. Verbatim: Use only the definitions of labels and relationship types provided.
4. Identifiers: Never write identifiers such as "BN:Employment_1" or any name that starts with "BN:". Refer to such an entity by its label (for example "an Employment record") and its properties.
5. Plain words: Write relationship types as plain words (for example "has employment" for hasEmployment).
6. Direct start: Start with the subject itself. Do not write "this community" or "the community", and do not use the word "central".
7. Length: Use at most 100 words.

GOAL: Produce a 3-to-5-sentence summary exclusively within provided data."""

NODES_HEADER = (
    "ENTITIES AND RELATIONSHIPS (one entity per line: its labels, its name, "
    "all its properties in braces, then all its relationships: -TYPE-> for an "
    "outgoing one and <-TYPE- for an incoming one, followed by the other "
    "entity's label and name):"
)

# Said when the text to summarize is itself a list of summaries (the reduce
# step of a community too large for one prompt).
SUMMARIES_HEADER = (
    "The text below is a list of summaries, each written for a different part "
    "of the same community. Produce ONE summary of the whole community, under "
    "the same rules and goal, using only what these summaries state.\n\n"
    "SUMMARIES:"
)

# Properties that are bookkeeping, or already shown as the node's name.
SKIPPED_PROPERTIES = frozenset(
    {
        "uuid",
        "community_id",
        "sourceJobIds",
        "embedding",
        "indexed",
        "label",
        "name",
    }
)


@dataclass(frozen=True)
class SummaryLimits:
    """How much text goes to the model, and how much of it."""

    #: Characters of one prompt's data (a batch of lines, or of summaries).
    batch_chars: int = 40_000
    #: Characters of a finished summary; longer ones are cut at a sentence.
    max_summary_chars: int = 1_500
    #: Characters of node lines read over a whole indexing run. A community
    #: that would exceed it is cut to its best-connected nodes (below), so one
    #: huge graph cannot make the run's cost unbounded.
    max_chars_per_run: int = 12_000_000
    #: A community always gets at least this much, whatever the run has used.
    min_chars_per_community: int = 20_000
    #: What is kept of a node's line: everything, within safety limits that
    #: only an abnormal node reaches (a value is cut, a node with hundreds of
    #: relationships gets "(+N more)").
    max_properties: int = 100
    max_relationships: int = 100
    value_chars: int = 300
    line_chars: int = 6_000
    #: Communities summarized at the same time.
    concurrency: int = 4

    def __post_init__(self) -> None:
        # A reduce step must shrink its input: at least two summaries per batch.
        if self.batch_chars < 2 * self.max_summary_chars:
            raise ValueError(
                "batch_chars must hold at least two summaries "
                f"(>= {2 * self.max_summary_chars})"
            )


DEFAULT_LIMITS = SummaryLimits()

SummarizeFn = Callable[[str], str]


def _short(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def node_line(
    labels: Sequence[str],
    name: str,
    props: Dict[str, Any],
    relationships: Sequence[Dict[str, Any]],
    limits: SummaryLimits = DEFAULT_LIMITS,
) -> str:
    """One node, with all its data, as a line, for example::

        Drug "Keytruda" {class: PD-1}; -TREATS-> Condition "Melanoma"; <-MAKES- Company "Merck"

    A relationship is `-TYPE-> label "name"` when it leaves the node and
    `<-TYPE- label "name"` when it arrives (its `dir` is `out` or `in`; `out`
    when absent), where label and name are the other entity's.
    """
    domain_labels = [l for l in labels if not l.startswith("_")]
    head = "/".join(domain_labels) or "Item"
    if name:
        head += f' "{_short(name, limits.value_chars)}"'

    properties = [
        f"{key}: {_short(value, limits.value_chars)}"
        for key, value in sorted(props.items())
        if key not in SKIPPED_PROPERTIES
        and not key.startswith("_")
        and value is not None
        and value != ""
    ][: limits.max_properties]
    if properties:
        head += " {" + ", ".join(properties) + "}"

    # The same relationship to entities of the same label and name is written
    # once, counted.
    counted: Dict[str, int] = {}
    for rel in relationships:
        if not rel or not rel.get("type"):
            continue
        other = _short(rel.get("target") or "Item", limits.value_chars)
        if rel.get("label"):
            other = f'{rel["label"]} "{other}"'
        text = (
            f"<-{rel['type']}- {other}"
            if rel.get("dir") == "in"
            else f"-{rel['type']}-> {other}"
        )
        counted[text] = counted.get(text, 0) + 1
    shown = [
        text if count == 1 else f"{text} (x{count})"
        for text, count in list(counted.items())[: limits.max_relationships]
    ]
    extra = len(counted) - limits.max_relationships
    if extra > 0:
        shown.append(f"(+{extra} more)")

    line = "; ".join([head, *shown])
    return line if len(line) <= limits.line_chars else line[: limits.line_chars - 3] + "..."


def pack_batches(items: Sequence[str], max_chars: int) -> List[List[str]]:
    """Groups items, in order, into batches of at most `max_chars` characters.

    An item longer than the limit is a batch of its own, never dropped.
    """
    batches: List[List[str]] = []
    current: List[str] = []
    size = 0
    for item in items:
        cost = len(item) + 1  # the newline that joins it to the next one
        if current and size + cost > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += cost
    if current:
        batches.append(current)
    return batches


def fit_budget(
    lines: Sequence[str], remaining_chars: int, limits: SummaryLimits = DEFAULT_LIMITS
) -> Tuple[List[str], bool]:
    """Keeps the first lines (the best-connected nodes) that fit the budget.

    Returns the lines kept and whether any were dropped. A community always
    keeps at least `min_chars_per_community`.
    """
    allowed = max(limits.min_chars_per_community, remaining_chars)
    kept: List[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > allowed and kept:
            return kept, True
        kept.append(line)
        used += len(line) + 1
    return kept, False


def clean_summary(text: Optional[str], max_chars: int) -> Optional[str]:
    """Trims the model's answer; None when there is nothing in it."""
    if not text:
        return None
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    sentence_end = max(cut.rfind(". "), cut.rfind(".\n"))
    # Cut at a sentence when one ends in the second half; else at a word.
    if sentence_end >= max_chars // 2:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0].rstrip() + "..."


def _map_prompt(lines: Sequence[str]) -> str:
    return f"{SUMMARY_PROMPT}\n\n{NODES_HEADER}\n" + "\n".join(lines)


def _reduce_prompt(summaries: Sequence[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(summaries, start=1))
    return f"{SUMMARY_PROMPT}\n\n{SUMMARIES_HEADER}\n{numbered}"


def summarize_community(
    lines: Sequence[str],
    summarize_fn: SummarizeFn,
    limits: SummaryLimits = DEFAULT_LIMITS,
) -> Optional[str]:
    """Summarizes a community from its node lines; None if it cannot.

    One prompt when the lines fit a batch. Otherwise each batch is summarized,
    then the summaries are summarized, again and again, until one is left. If
    any call fails, the community has no summary: a partial one would leave out
    a part of the community without saying so.
    """
    if not lines:
        return None

    def ask(prompt: str) -> Optional[str]:
        try:
            return clean_summary(summarize_fn(prompt), limits.max_summary_chars)
        except Exception as error:  # noqa: BLE001 - a failed call is a fallback, not a crash
            logger.warning(f"   Community summary call failed: {error}")
            return None

    parts: List[str] = []
    for batch in pack_batches(lines, limits.batch_chars):
        summary = ask(_map_prompt(batch))
        if summary is None:
            return None
        parts.append(summary)

    while len(parts) > 1:
        reduced: List[str] = []
        for batch in pack_batches(parts, limits.batch_chars):
            summary = ask(_reduce_prompt(batch)) if len(batch) > 1 else batch[0]
            if summary is None:
                return None
            reduced.append(summary)
        parts = reduced
    return parts[0]


def summarize_communities(
    communities: Iterable[Tuple[Any, Sequence[str]]],
    summarize_fn: SummarizeFn,
    limits: SummaryLimits = DEFAULT_LIMITS,
) -> Dict[Any, Optional[str]]:
    """Summarizes several communities at once, `limits.concurrency` at a time."""
    items = list(communities)
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, limits.concurrency)) as pool:
        results = pool.map(
            lambda item: summarize_community(item[1], summarize_fn, limits), items
        )
        return {item[0]: result for item, result in zip(items, results)}
