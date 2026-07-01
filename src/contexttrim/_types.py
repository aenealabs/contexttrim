"""Shared types for contexttrim: the message shape, drop records, and the
Strategy interface every truncation strategy implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

# A chat message in the common OpenAI/Anthropic dict shape:
# {"role": "system"|"user"|"assistant"|"tool", "content": str | ...}
Message = dict

# A token counter maps a single message to an integer token estimate.
TokenCounter = Callable[[Message], int]


@dataclass
class Dropped:
    """A message removed (or merged/truncated) during a fit, with the reason."""

    message: Message
    reason: str


class Strategy:
    """Base class for context-fitting strategies.

    A strategy decides which messages to keep so the total stays within a token
    budget. Implement :meth:`fit` to return ``(kept, dropped)`` where ``kept`` is
    the surviving message list (in conversation order) and ``dropped`` is a list
    of :class:`Dropped` records explaining what was removed and why.

    Strategies must never mutate the input messages in place — return new lists
    (and new dicts if a message's content is rewritten).
    """

    def fit(
        self,
        messages: List[Message],
        budget: int,
        count: TokenCounter,
    ) -> "Tuple[List[Message], List[Dropped]]":
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"
