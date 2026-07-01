"""ContextManager and FitReport behavior."""

import unittest

from contexttrim import ContextManager, FitReport
from contexttrim.strategies import RecencyDrop

CHARS = lambda m: len(m.get("content", ""))  # noqa: E731 - deterministic counter


def msg(role, n):
    return {"role": role, "content": "x" * n}


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.ctx = ContextManager(token_budget=20, strategy=RecencyDrop(), token_counter=CHARS)

    def test_add_extend_clear_and_messages_copy(self):
        self.ctx.add(msg("user", 5)).extend([msg("assistant", 5)])
        self.assertEqual(len(self.ctx.messages), 2)
        # messages returns a copy — mutating it must not affect the manager
        self.ctx.messages.append(msg("user", 1))
        self.assertEqual(len(self.ctx.messages), 2)
        self.assertEqual(self.ctx.total_tokens(), 10)
        self.ctx.clear()
        self.assertEqual(self.ctx.messages, [])

    def test_fit_returns_new_list_and_does_not_mutate(self):
        originals = [msg("user", 10), msg("assistant", 10), msg("user", 10)]
        self.ctx.extend(originals)
        before = [dict(m) for m in originals]
        trimmed = self.ctx.fit()
        self.assertEqual(len(trimmed), 2)  # budget 20 keeps the two newest
        self.assertEqual(self.ctx.messages, before)  # untouched

    def test_last_fit_report(self):
        self.ctx.extend([msg("user", 10), msg("user", 10), msg("user", 10)])
        self.ctx.fit()
        report = self.ctx.last_fit_report()
        self.assertIsInstance(report, FitReport)
        self.assertEqual(report.tokens_used, 20)
        self.assertEqual(report.tokens_budget, 20)
        self.assertEqual(report.num_dropped, 1)
        self.assertTrue(report.fits)

    def test_report_before_fit_is_none(self):
        self.assertIsNone(self.ctx.last_fit_report())

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            ContextManager(token_budget=-1, strategy=RecencyDrop())

    def test_everything_fits_drops_nothing(self):
        self.ctx.extend([msg("user", 5), msg("user", 5)])
        trimmed = self.ctx.fit()
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(self.ctx.last_fit_report().num_dropped, 0)


if __name__ == "__main__":
    unittest.main()
