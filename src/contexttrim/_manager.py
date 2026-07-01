"""The ContextManager and its fit report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ._tokens import default_token_counter
from ._types import Dropped, Message, Strategy, TokenCounter


@dataclass
class FitReport:
    """The outcome of a :meth:`ContextManager.fit` call."""

    kept: List[Message]
    dropped: List[Dropped]
    tokens_used: int
    tokens_budget: int

    @property
    def fits(self) -> bool:
        """True if the kept messages are within budget (may be False if pinned
        messages alone exceed it)."""
        return self.tokens_used <= self.tokens_budget

    @property
    def num_dropped(self) -> int:
        return len(self.dropped)

    def __repr__(self) -> str:
        return (
            f"FitReport(kept={len(self.kept)}, dropped={self.num_dropped}, "
            f"tokens_used={self.tokens_used}/{self.tokens_budget})"
        )


class ContextManager:
    """Manage a conversation and trim it to fit a token budget on demand.

    Add messages as they accumulate, then call :meth:`fit` to get a pruned list
    that fits ``token_budget`` according to the chosen strategy. Messages are
    never mutated; ``fit`` returns a new list.

    Example
    -------
    ::

        from contexttrim import ContextManager
        from contexttrim.strategies import RoleWeighted

        ctx = ContextManager(token_budget=8000, strategy=RoleWeighted())
        ctx.add({"role": "system", "content": "You are helpful."})
        ctx.add({"role": "user", "content": "Find flights to NYC."})
        trimmed = ctx.fit()
        print(ctx.last_fit_report().dropped)
    """

    def __init__(
        self,
        token_budget: int,
        strategy: Strategy,
        token_counter: "Optional[TokenCounter]" = None,
    ) -> None:
        if token_budget < 0:
            raise ValueError("token_budget must be non-negative")
        self.token_budget = token_budget
        self.strategy = strategy
        self.count: TokenCounter = token_counter or default_token_counter
        self._messages: List[Message] = []
        self._last_report: Optional[FitReport] = None

    # ---- building the conversation --------------------------------------
    def add(self, message: Message) -> "ContextManager":
        """Append a message. Returns self for chaining."""
        self._messages.append(message)
        return self

    def extend(self, messages) -> "ContextManager":
        """Append several messages."""
        self._messages.extend(messages)
        return self

    def clear(self) -> "ContextManager":
        self._messages.clear()
        self._last_report = None
        return self

    @property
    def messages(self) -> List[Message]:
        """A copy of the current (untrimmed) messages."""
        return list(self._messages)

    def total_tokens(self) -> int:
        """Total estimated tokens of the current (untrimmed) messages."""
        return sum(self.count(m) for m in self._messages)

    # ---- fitting --------------------------------------------------------
    def fit(self) -> List[Message]:
        """Return a pruned copy of the messages that fits the token budget."""
        kept, dropped = self.strategy.fit(self._messages, self.token_budget, self.count)
        tokens_used = sum(self.count(m) for m in kept)
        self._last_report = FitReport(
            kept=kept,
            dropped=dropped,
            tokens_used=tokens_used,
            tokens_budget=self.token_budget,
        )
        return kept

    def last_fit_report(self) -> "Optional[FitReport]":
        """The :class:`FitReport` from the most recent :meth:`fit`, or None."""
        return self._last_report
