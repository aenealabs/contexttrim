"""Token accounting for contexttrim.

Zero dependencies means no real tokenizer (tiktoken is third-party). The default
counter uses a deterministic ~4-characters-per-token heuristic over a message's
content — good enough for budgeting. Inject your own counter for exact counts::

    ContextManager(token_budget=8000, token_counter=lambda m: my_tokenizer(m))
"""

from __future__ import annotations

import json
import math

from ._types import Message

_CHARS_PER_TOKEN = 4


def content_to_text(message: Message) -> str:
    """Return a message's content as text, serializing non-string content."""
    content = message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def default_token_counter(message: Message) -> int:
    """Estimate a message's token count via the character heuristic."""
    text = content_to_text(message)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))
