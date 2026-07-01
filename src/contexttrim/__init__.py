"""
contexttrim — Pluggable context window management for LLM agents.

Zero dependencies. Pure Python stdlib.

When a conversation exceeds a model's token limit, dropping the oldest messages
(FIFO) is often wrong — you might drop a critical system prompt while keeping
ten redundant tool results. contexttrim gives you a ``ContextManager`` with
swappable truncation strategies so you decide *what* gets dropped and *why*.

Quick start
-----------
::

    from contexttrim import ContextManager
    from contexttrim.strategies import ImportanceWeighted

    ctx = ContextManager(token_budget=8_000, strategy=ImportanceWeighted())
    ctx.add({"role": "system", "content": "You are a helpful assistant."})
    ctx.add({"role": "user", "content": "Find me flights to NYC."})
    ctx.add({"role": "tool", "content": "<very long search result>"})

    trimmed = ctx.fit()                 # a new list that fits the budget
    report = ctx.last_fit_report()      # what was dropped and why
    print(report.dropped, report.tokens_used)

Strategies (in ``contexttrim.strategies``)
-----------------------------------------
- ``RecencyDrop``        — drop the oldest messages first
- ``MiddleDrop``         — drop from the middle ("lost in the middle")
- ``RoleWeighted``       — drop lowest-scored roles first; pin ``system``
- ``ImportanceWeighted`` — role weight × recency decay ÷ length penalty
- ``ToolResultMerge``    — merge/truncate redundant tool results (no context lost)
- ``SemanticCluster``    — drop least topically-relevant messages (TF-IDF, no ML)

Token counting defaults to a ~4-chars-per-token heuristic; inject your own via
``ContextManager(..., token_counter=...)``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

from . import strategies
from ._manager import ContextManager, FitReport
from ._tokens import content_to_text, default_token_counter
from ._types import Dropped, Message, Strategy, TokenCounter

__all__ = [
    "ContextManager",
    "FitReport",
    "Dropped",
    "Strategy",
    "Message",
    "TokenCounter",
    "default_token_counter",
    "content_to_text",
    "strategies",
]

try:
    __version__ = _version("contexttrim")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = "0.0.0"
