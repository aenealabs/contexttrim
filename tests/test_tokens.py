"""Token counting: default heuristic and content extraction."""

import unittest

from contexttrim import content_to_text, default_token_counter


class TestTokens(unittest.TestCase):
    def test_empty_content_is_zero(self):
        self.assertEqual(default_token_counter({"role": "user", "content": ""}), 0)
        self.assertEqual(default_token_counter({"role": "user"}), 0)

    def test_four_chars_per_token(self):
        self.assertEqual(default_token_counter({"content": "a" * 40}), 10)

    def test_nonempty_is_at_least_one(self):
        self.assertEqual(default_token_counter({"content": "hi"}), 1)

    def test_non_string_content_serialized(self):
        msg = {"content": [{"type": "text", "text": "hello"}]}
        self.assertGreater(default_token_counter(msg), 0)

    def test_content_to_text_none(self):
        self.assertEqual(content_to_text({"content": None}), "")

    def test_content_to_text_passthrough(self):
        self.assertEqual(content_to_text({"content": "hi"}), "hi")


if __name__ == "__main__":
    unittest.main()
