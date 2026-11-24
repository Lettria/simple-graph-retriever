import pytest

from simple_graph_retriever.community_summary import (
    NODES_HEADER,
    SUMMARIES_HEADER,
    SUMMARY_PROMPT,
    SummaryLimits,
    clean_summary,
    fit_budget,
    node_line,
    pack_batches,
    summarize_communities,
    summarize_community,
)

SMALL = SummaryLimits(batch_chars=400, max_summary_chars=100)


def test_prompt_is_the_agreed_one():
    assert SUMMARY_PROMPT.startswith("You are a Strict Logic-Based Data knowledge graph analyst.")
    assert "1. Zero External Knowledge" in SUMMARY_PROMPT
    assert "2. Adjectives: Do not add descriptive words like 'major' or 'important'." in SUMMARY_PROMPT
    assert "3. Verbatim: Use only the definitions of labels and relationship types provided." in SUMMARY_PROMPT
    assert SUMMARY_PROMPT.endswith("GOAL: Produce a 3-to-5-sentence summary exclusively within provided data.")


def test_prompt_adds_rules_for_identifiers_style_length_and_output_after_the_original_three():
    rules = SUMMARY_PROMPT.split("STRICT RULES:\n", 1)[1].split("\n\nGOAL", 1)[0].split("\n")

    assert len(rules) == 8
    assert [r.split(".", 1)[0] for r in rules] == ["1", "2", "3", "4", "5", "6", "7", "8"]
    assert "BN:" in rules[3] and "label" in rules[3]
    assert "plain words" in rules[4].lower()
    assert "this community" in rules[5] and "central" in rules[5]
    assert "100 words" in rules[6]
    assert "only the final summary" in rules[7] and "drafts" in rules[7]


class TestNodeLine:
    def test_labels_name_properties_and_relationships_with_their_direction(self):
        line = node_line(
            ["Drug", "_CommunityMember"],
            "Keytruda",
            {"class": "PD-1", "uuid": "u1", "community_id": 3, "sourceJobIds": ["j"]},
            [
                {"type": "TREATS", "target": "Melanoma", "label": "Condition", "dir": "out"},
                {"type": "MAKES", "target": "Merck", "label": "Company", "dir": "in"},
            ],
        )
        assert line == (
            'Drug "Keytruda" {class: PD-1}; -TREATS-> Condition "Melanoma"; <-MAKES- Company "Merck"'
        )

    def test_a_relationship_without_direction_or_label_leaves_the_node(self):
        line = node_line(["A"], "a", {}, [{"type": "R", "target": "b"}])
        assert line == 'A "a"; -R-> b'

    def test_a_node_without_name_or_extras_is_just_its_labels(self):
        assert node_line(["Person"], "", {}, []) == "Person"
        assert node_line(["_CommunityMember"], "", {}, []) == "Item"

    def test_skips_empty_values_and_the_name_property(self):
        line = node_line(["A"], "x", {"label": "x", "name": "x", "k": "", "n": None, "v": 2}, [])
        assert line == 'A "x" {v: 2}'

    def test_keeps_all_the_properties_and_relationships_of_an_ordinary_node(self):
        props = {f"property{i:02d}": f"value {i}" for i in range(30)}
        rels = [{"type": f"REL{i}", "target": f"t{i}", "label": "L", "dir": "out"} for i in range(40)]

        line = node_line(["A"], "n", props, rels)

        assert all(f"property{i:02d}: value {i}" in line for i in range(30))
        assert all(f'-REL{i}-> L "t{i}"' in line for i in range(40))
        assert "more)" not in line

    def test_safety_limits_cut_values_properties_and_relationships(self):
        limits = SummaryLimits(max_properties=2, max_relationships=1, value_chars=10)
        line = node_line(
            ["A"],
            "n",
            {"a": "1", "b": "2", "c": "3", "long": "x" * 50},
            [{"type": "R1", "target": "t1"}, {"type": "R2", "target": "t2"}, {"type": "R3", "target": "t3"}],
            limits,
        )
        assert line == 'A "n" {a: 1, b: 2}; -R1-> t1; (+2 more)'
        assert "x" * 11 not in node_line(["A"], "n", {"long": "x" * 50}, [], limits)

    def test_a_long_value_is_cut_at_300_characters_by_default(self):
        line = node_line(["A"], "n", {"text": "y" * 1000}, [])

        assert "y" * 300 not in line and "y" * 290 in line

    def test_a_repeated_relationship_is_written_once_with_its_count(self):
        line = node_line(
            ["Doc"],
            "d",
            {},
            [{"type": "CITES", "target": "X"}] * 3 + [{"type": "CITES", "target": "Y"}],
        )
        assert line == 'Doc "d"; -CITES-> X (x3); -CITES-> Y'

    def test_flattens_whitespace_in_values(self):
        assert node_line(["A"], "two\nlines", {}, []) == 'A "two lines"'


class TestPackBatches:
    def test_groups_in_order_within_the_limit(self):
        assert pack_batches(["aaaa", "bbbb", "cccc"], 10) == [["aaaa", "bbbb"], ["cccc"]]

    def test_an_oversized_item_is_a_batch_of_its_own(self):
        assert pack_batches(["a", "b" * 50, "c"], 10) == [["a"], ["b" * 50], ["c"]]

    def test_nothing_is_dropped(self):
        items = [f"item-{i}" for i in range(100)]
        assert [x for batch in pack_batches(items, 60) for x in batch] == items

    def test_empty(self):
        assert pack_batches([], 10) == []


class TestFitBudget:
    LINES = ["a" * 9, "b" * 9, "c" * 9]  # 10 chars each with its newline

    def test_keeps_the_first_lines_that_fit(self):
        limits = SummaryLimits(min_chars_per_community=0)
        assert fit_budget(self.LINES, 25, limits) == (self.LINES[:2], True)

    def test_keeps_everything_when_it_fits(self):
        limits = SummaryLimits(min_chars_per_community=0)
        assert fit_budget(self.LINES, 1000, limits) == (self.LINES, False)

    def test_a_community_always_gets_its_minimum(self):
        limits = SummaryLimits(min_chars_per_community=20)
        kept, cut = fit_budget(self.LINES, -500, limits)
        assert kept == self.LINES[:2] and cut

    def test_the_first_line_is_kept_even_when_too_long(self):
        limits = SummaryLimits(min_chars_per_community=0)
        assert fit_budget(["x" * 100], 5, limits) == (["x" * 100], False)


class TestCleanSummary:
    def test_none_for_nothing(self):
        assert clean_summary(None, 100) is None
        assert clean_summary("  \n ", 100) is None

    def test_trims_and_collapses_blank_runs(self):
        assert clean_summary("  One.\n\n\n\nTwo.  ", 100) == "One.\n\nTwo."

    def test_cuts_a_long_answer_at_a_sentence(self):
        text = "First sentence here. Second sentence here. Third sentence is cut mid-way"
        assert clean_summary(text, 45) == "First sentence here. Second sentence here."

    def test_cuts_at_a_word_when_no_sentence_ends_late_enough(self):
        assert clean_summary("word " * 40, 22) == "word word word word..."


class TestSummarizeCommunity:
    def test_one_prompt_when_the_lines_fit(self):
        prompts = []

        def fn(prompt):
            prompts.append(prompt)
            return "A summary."

        assert summarize_community(["Drug \"A\"", "Drug \"B\""], fn, SMALL) == "A summary."
        assert len(prompts) == 1
        assert prompts[0].startswith(SUMMARY_PROMPT)
        assert NODES_HEADER in prompts[0]
        assert prompts[0].endswith('Drug "A"\nDrug "B"')

    def test_every_line_reaches_the_model_when_the_community_is_large(self):
        seen = []

        def fn(prompt):
            if SUMMARIES_HEADER in prompt:
                return "Combined."
            seen.extend(prompt.split(NODES_HEADER + "\n", 1)[1].split("\n"))
            return "Part."

        lines = [f"Node {i:03d}" for i in range(200)]  # several batches of 400 chars
        assert summarize_community(lines, fn, SMALL) == "Combined."
        assert seen == lines

    def test_summaries_are_reduced_until_one_is_left(self):
        calls = {"map": 0, "reduce": 0}

        def fn(prompt):
            if SUMMARIES_HEADER in prompt:
                calls["reduce"] += 1
                return "r" * 90
            calls["map"] += 1
            return "m" * 90

        lines = ["x" * 50] * 200  # many map batches, whose summaries need several reduce levels
        result = summarize_community(lines, fn, SMALL)

        assert result == "r" * 90
        assert calls["map"] > 4 and calls["reduce"] >= 2

    def test_reduce_prompt_lists_the_partial_summaries(self):
        prompts = []

        def fn(prompt):
            prompts.append(prompt)
            return f"S{len(prompts)}."

        summarize_community(["x" * 150] * 5, fn, SMALL)

        reduce_prompt = next(p for p in prompts if SUMMARIES_HEADER in p)
        assert "1. S1." in reduce_prompt and "2. S2." in reduce_prompt

    def test_none_when_a_call_fails(self):
        def fn(prompt):
            raise RuntimeError("throttled")

        assert summarize_community(["Drug"], fn, SMALL) is None

    def test_none_when_one_batch_fails_rather_than_a_partial_summary(self):
        calls = []

        def fn(prompt):
            calls.append(prompt)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return "Part."

        assert summarize_community(["x" * 150] * 5, fn, SMALL) is None

    def test_none_for_an_empty_answer_or_no_lines(self):
        assert summarize_community(["Drug"], lambda p: "   ", SMALL) is None
        assert summarize_community([], lambda p: "x", SMALL) is None

    def test_a_long_answer_is_cut(self):
        assert len(summarize_community(["Drug"], lambda p: "word " * 100, SMALL)) <= 103


def test_batch_must_hold_two_summaries():
    with pytest.raises(ValueError):
        SummaryLimits(batch_chars=100, max_summary_chars=100)


def test_summarize_communities_keys_the_results_by_id():
    result = summarize_communities(
        [(1, ["A"]), (2, ["B"]), (3, [])],
        lambda prompt: "Summary of " + prompt.rsplit("\n", 1)[1],
        SMALL,
    )
    assert result == {1: "Summary of A", 2: "Summary of B", 3: None}
