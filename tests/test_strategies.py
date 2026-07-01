"""All six context-fitting strategies."""

import unittest

from contexttrim.strategies import (
    ImportanceWeighted,
    MiddleDrop,
    RecencyDrop,
    RoleWeighted,
    SemanticCluster,
    ToolResultMerge,
)

CHARS = lambda m: len(m.get("content", ""))  # noqa: E731 - deterministic counter


def m(role, content):
    return {"role": role, "content": content}


def roles(messages):
    return [x["role"] for x in messages]


class TestRecencyDrop(unittest.TestCase):
    def test_keeps_newest_within_budget(self):
        msgs = [m("user", "a" * 10), m("assistant", "b" * 10), m("user", "c" * 10)]
        kept, dropped = RecencyDrop().fit(msgs, budget=20, count=CHARS)
        self.assertEqual([x["content"] for x in kept], ["b" * 10, "c" * 10])
        self.assertEqual(len(dropped), 1)
        self.assertIn("recency", dropped[0].reason)


class TestMiddleDrop(unittest.TestCase):
    def test_preserves_head_and_tail(self):
        msgs = [m("user", "x" * 10) for _ in range(5)]
        for i, x in enumerate(msgs):
            x["content"] = str(i) + x["content"][1:]  # tag first char with index
        kept, dropped = MiddleDrop().fit(msgs, budget=30, count=CHARS)
        kept_idx = [int(x["content"][0]) for x in kept]
        self.assertIn(0, kept_idx)  # head kept
        self.assertIn(4, kept_idx)  # tail kept
        self.assertNotIn(2, kept_idx)  # middle dropped first
        self.assertEqual(len(kept), 3)


class TestRoleWeighted(unittest.TestCase):
    def test_drops_low_role_first_and_pins_system(self):
        msgs = [m("system", "s" * 10), m("user", "u" * 10),
                m("assistant", "a" * 10), m("tool", "t" * 10)]
        kept, dropped = RoleWeighted().fit(msgs, budget=20, count=CHARS)
        self.assertEqual(roles(kept), ["system", "user"])
        self.assertEqual(roles([d.message for d in dropped]), ["assistant", "tool"])

    def test_pinned_system_kept_even_when_over_budget(self):
        msgs = [m("system", "s" * 30)]
        kept, dropped = RoleWeighted().fit(msgs, budget=5, count=CHARS)
        self.assertEqual(roles(kept), ["system"])
        self.assertEqual(dropped, [])


class TestImportanceWeighted(unittest.TestCase):
    def test_drops_old_long_before_new_short(self):
        msgs = [
            m("system", "s" * 5),
            m("assistant", "L" * 40),  # old + long → lowest importance
            m("assistant", "S" * 4),   # new + short → highest importance
        ]
        kept, dropped = ImportanceWeighted().fit(msgs, budget=15, count=CHARS)
        self.assertEqual([x["content"] for x in kept], ["s" * 5, "S" * 4])
        self.assertEqual(dropped[0].message["content"], "L" * 40)

    def test_system_pinned(self):
        msgs = [m("system", "s" * 100), m("user", "u" * 4)]
        kept, _ = ImportanceWeighted().fit(msgs, budget=10, count=CHARS)
        self.assertIn("system", roles(kept))


class TestToolResultMerge(unittest.TestCase):
    def test_merges_adjacent_tool_runs_and_dedups(self):
        msgs = [
            m("user", "u" * 10),
            m("tool", "AAAA"),
            m("tool", "AAAA"),  # duplicate
            m("tool", "BBBB"),
            m("assistant", "a" * 10),
        ]
        kept, dropped = ToolResultMerge().fit(msgs, budget=1000, count=CHARS)
        self.assertEqual(roles(kept), ["user", "tool", "assistant"])  # run collapsed to one
        tool_msg = [x for x in kept if x["role"] == "tool"][0]
        self.assertEqual(tool_msg["content"], "AAAA\n---\nBBBB")
        reasons = sorted(d.reason for d in dropped)
        self.assertTrue(any("duplicate" in r for r in reasons))
        self.assertTrue(any("merged into" in r for r in reasons))

    def test_truncates_oversized_tool_and_terminates(self):
        msgs = [m("tool", "x" * 400)]
        kept, dropped = ToolResultMerge().fit(msgs, budget=20, count=CHARS)
        content = kept[0]["content"]
        self.assertTrue(content.endswith("…[truncated]"))
        self.assertLess(len(content), 400)
        self.assertTrue(any("truncated" in d.reason for d in dropped))

    def test_non_tool_messages_preserved(self):
        msgs = [m("user", "u" * 5), m("assistant", "a" * 5)]
        kept, dropped = ToolResultMerge().fit(msgs, budget=1000, count=CHARS)
        self.assertEqual(roles(kept), ["user", "assistant"])
        self.assertEqual(dropped, [])


class TestSemanticCluster(unittest.TestCase):
    def test_keeps_relevant_drops_irrelevant(self):
        msgs = [
            m("system", "system prompt"),
            m("assistant", "the weather in paris is sunny and warm today"),  # irrelevant
            m("assistant", "flights from boston to new york depart hourly"),  # relevant
            m("user", "find me flights from boston to new york"),            # focus
        ]
        # budget fits system + the two relevant messages, but not the weather one
        kept, dropped = SemanticCluster().fit(msgs, budget=100, count=CHARS)
        kept_contents = " ".join(x["content"] for x in kept)
        self.assertIn("flights", kept_contents)         # relevant kept
        self.assertNotIn("weather", kept_contents)      # least relevant dropped
        self.assertIn("system", roles(kept))            # system pinned

    def test_empty_messages(self):
        self.assertEqual(SemanticCluster().fit([], 100, CHARS), ([], []))


class TestNoMutation(unittest.TestCase):
    def test_strategies_do_not_mutate_inputs(self):
        for strat in (RecencyDrop(), MiddleDrop(), RoleWeighted(),
                      ImportanceWeighted(), ToolResultMerge(), SemanticCluster()):
            msgs = [m("system", "s" * 5), m("tool", "t" * 30), m("tool", "t" * 30),
                    m("user", "hello world")]
            snapshot = [dict(x) for x in msgs]
            strat.fit(msgs, budget=10, count=CHARS)
            self.assertEqual(msgs, snapshot, f"{type(strat).__name__} mutated its input")


if __name__ == "__main__":
    unittest.main()
