"""Context-fitting strategies.

Each strategy decides which messages to keep within a token budget when a
conversation grows too large. All are pure Python stdlib — no ML, no tokenizer,
no network.

    from contexttrim.strategies import (
        RecencyDrop, MiddleDrop, RoleWeighted,
        ImportanceWeighted, ToolResultMerge, SemanticCluster,
    )

Every strategy returns ``(kept, dropped)`` and never mutates its inputs.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, FrozenSet, List, Tuple

from ._tokens import content_to_text
from ._types import Dropped, Message, Strategy, TokenCounter

__all__ = [
    "Strategy",
    "RecencyDrop",
    "MiddleDrop",
    "RoleWeighted",
    "ImportanceWeighted",
    "ToolResultMerge",
    "SemanticCluster",
]

_DEFAULT_ROLE_WEIGHTS = {"system": 10.0, "user": 2.0, "assistant": 1.0, "tool": 0.5}
_DEFAULT_PINNED = frozenset({"system"})


def _total(messages: List[Message], count: TokenCounter) -> int:
    return sum(count(m) for m in messages)


def _drop_in_order(
    messages: List[Message],
    budget: int,
    count: TokenCounter,
    drop_order: List[int],
    reason: Callable[[Message], str],
) -> "Tuple[List[Message], List[Dropped]]":
    """Drop messages by index in ``drop_order`` until the total fits the budget.

    Indices absent from ``drop_order`` are pinned (never dropped). Kept messages
    preserve their original order.
    """
    used = _total(messages, count)
    to_drop = set()
    for idx in drop_order:
        if used <= budget:
            break
        used -= count(messages[idx])
        to_drop.add(idx)
    kept = [m for i, m in enumerate(messages) if i not in to_drop]
    dropped = [Dropped(messages[i], reason(messages[i])) for i in sorted(to_drop)]
    return kept, dropped


def _role(message: Message) -> str:
    return message.get("role", "")


# ---------------------------------------------------------------------------
# RecencyDrop
# ---------------------------------------------------------------------------
class RecencyDrop(Strategy):
    """Drop the oldest messages first. Fast, simple, often wrong."""

    def fit(self, messages, budget, count):
        # Oldest-first drop order (index 0 is the oldest).
        drop_order = list(range(len(messages)))
        return _drop_in_order(
            messages, budget, count, drop_order,
            lambda m: "dropped as oldest (recency)",
        )


# ---------------------------------------------------------------------------
# MiddleDrop
# ---------------------------------------------------------------------------
class MiddleDrop(Strategy):
    """Drop messages from the middle first.

    Models attend least to the middle of a long context (the "lost in the
    middle" effect), so the head and tail are preserved longest.
    """

    def fit(self, messages, budget, count):
        n = len(messages)
        center = (n - 1) / 2
        # Closest-to-center indices dropped first; head/tail dropped last.
        drop_order = sorted(range(n), key=lambda i: (abs(i - center), i))
        return _drop_in_order(
            messages, budget, count, drop_order,
            lambda m: "dropped from the middle (lost-in-the-middle)",
        )


# ---------------------------------------------------------------------------
# RoleWeighted
# ---------------------------------------------------------------------------
class RoleWeighted(Strategy):
    """Score messages by role and drop the lowest-scored first.

    Pinned roles (``system`` by default) are never dropped.
    """

    def __init__(
        self,
        role_weights: "Dict[str, float] | None" = None,
        pin_roles: "FrozenSet[str] | None" = None,
        default_weight: float = 1.0,
    ) -> None:
        self.role_weights = dict(role_weights) if role_weights else dict(_DEFAULT_ROLE_WEIGHTS)
        self.pin_roles = frozenset(pin_roles) if pin_roles is not None else _DEFAULT_PINNED
        self.default_weight = default_weight

    def _score(self, message: Message) -> float:
        return self.role_weights.get(_role(message), self.default_weight)

    def fit(self, messages, budget, count):
        droppable = [i for i, m in enumerate(messages) if _role(m) not in self.pin_roles]
        # Lowest score first; ties broken by oldest first.
        droppable.sort(key=lambda i: (self._score(messages[i]), i))
        return _drop_in_order(
            messages, budget, count, droppable,
            lambda m: f"low role weight ({_role(m) or 'unknown'})",
        )


# ---------------------------------------------------------------------------
# ImportanceWeighted
# ---------------------------------------------------------------------------
class ImportanceWeighted(Strategy):
    """Combine role weight, recency decay, and a length penalty.

    Importance = ``role_weight * recency_decay**age / (1 + length_penalty*tokens)``.
    Keeps short, recent, high-role messages; drops long, stale, low-role ones.
    Pinned roles (``system`` by default) are never dropped.
    """

    def __init__(
        self,
        role_weights: "Dict[str, float] | None" = None,
        pin_roles: "FrozenSet[str] | None" = None,
        recency_decay: float = 0.9,
        length_penalty: float = 0.01,
        default_weight: float = 1.0,
    ) -> None:
        self.role_weights = dict(role_weights) if role_weights else dict(_DEFAULT_ROLE_WEIGHTS)
        self.pin_roles = frozenset(pin_roles) if pin_roles is not None else _DEFAULT_PINNED
        self.recency_decay = recency_decay
        self.length_penalty = length_penalty
        self.default_weight = default_weight

    def _importance(self, message: Message, index: int, n: int, count: TokenCounter) -> float:
        role_weight = self.role_weights.get(_role(message), self.default_weight)
        age = (n - 1) - index  # 0 = newest
        recency = self.recency_decay ** age
        length = 1.0 + self.length_penalty * count(message)
        return role_weight * recency / length

    def fit(self, messages, budget, count):
        n = len(messages)
        droppable = [i for i, m in enumerate(messages) if _role(m) not in self.pin_roles]
        # Lowest importance first; ties broken by oldest first.
        droppable.sort(key=lambda i: (self._importance(messages[i], i, n, count), i))
        return _drop_in_order(
            messages, budget, count, droppable,
            lambda m: "low importance (role/recency/length)",
        )


# ---------------------------------------------------------------------------
# ToolResultMerge
# ---------------------------------------------------------------------------
class ToolResultMerge(Strategy):
    """Merge redundant adjacent tool results, then truncate if still over budget.

    Reduces token count without dropping conversational context:
    1. Runs of adjacent ``tool`` messages are merged into one; exact-duplicate
       contents are de-duplicated.
    2. If still over budget, the largest tool messages are truncated.

    Non-tool messages are always preserved.
    """

    def __init__(self, separator: str = "\n---\n", truncation_marker: str = " …[truncated]") -> None:
        self.separator = separator
        self.truncation_marker = truncation_marker

    def _merge_runs(self, messages, count):
        merged: List[Message] = []
        dropped: List[Dropped] = []
        i = 0
        n = len(messages)
        while i < n:
            if _role(messages[i]) != "tool":
                merged.append(messages[i])
                i += 1
                continue
            # Collect the adjacent run of tool messages.
            j = i
            run = []
            while j < n and _role(messages[j]) == "tool":
                run.append(messages[j])
                j += 1
            if len(run) == 1:
                merged.append(run[0])
            else:
                seen = set()
                distinct_texts = []
                for k, msg in enumerate(run):
                    text = content_to_text(msg)
                    is_dup = text in seen
                    if not is_dup:
                        seen.add(text)
                        distinct_texts.append(text)
                    if k > 0:
                        # run[0] becomes the combined message; the rest fold in.
                        dropped.append(Dropped(
                            msg,
                            "merged (duplicate tool result)" if is_dup
                            else "merged into adjacent tool result",
                        ))
                combined = dict(run[0])
                combined["content"] = self.separator.join(distinct_texts)
                merged.append(combined)
            i = j
        return merged, dropped

    def fit(self, messages, budget, count):
        merged, dropped = self._merge_runs(messages, count)

        # Truncate the largest tool messages until we fit (or can't shrink more).
        used = _total(merged, count)
        truncated_idxs = set()
        while used > budget:
            # Largest tool message that still has content to cut.
            candidates = [
                (count(m), idx)
                for idx, m in enumerate(merged)
                if _role(m) == "tool" and len(content_to_text(m)) > 1
            ]
            if not candidates:
                break
            candidates.sort(reverse=True)
            _, idx = candidates[0]

            text = content_to_text(merged[idx])
            if text.endswith(self.truncation_marker):  # don't compound the marker
                text = text[: -len(self.truncation_marker)]
            new_len = len(text) // 2
            if new_len <= 0:
                break
            new_msg = dict(merged[idx])
            new_msg["content"] = text[:new_len] + self.truncation_marker
            if count(new_msg) >= count(merged[idx]):
                break  # no progress (marker overhead dominates) — avoid looping
            merged[idx] = new_msg
            truncated_idxs.add(idx)
            used = _total(merged, count)

        for idx in sorted(truncated_idxs):
            dropped.append(Dropped(merged[idx], "truncated tool result (over budget)"))

        return merged, dropped


# ---------------------------------------------------------------------------
# SemanticCluster (pure-Python TF-IDF)
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _tfidf_vectors(docs: List[List[str]]) -> List[Dict[str, float]]:
    n = len(docs)
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    idf = {term: math.log((1 + n) / (1 + d)) + 1.0 for term, d in df.items()}
    vectors: List[Dict[str, float]] = []
    for doc in docs:
        if not doc:
            vectors.append({})
            continue
        tf: Dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        vectors.append({term: (c / len(doc)) * idf[term] for term, c in tf.items()})
    return vectors


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCluster(Strategy):
    """Drop the least topically-relevant messages first, via TF-IDF similarity.

    Each message is scored by its TF-IDF cosine similarity to the current focus
    of the conversation (the most recent ``user`` message, or the last message
    if there is none). Messages least similar to that focus are dropped first.
    Pinned roles (``system`` by default) are never dropped. Pure stdlib — no ML.
    """

    def __init__(self, pin_roles: "FrozenSet[str] | None" = None) -> None:
        self.pin_roles = frozenset(pin_roles) if pin_roles is not None else _DEFAULT_PINNED

    def _focus_index(self, messages: List[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if _role(messages[i]) == "user":
                return i
        return len(messages) - 1

    def fit(self, messages, budget, count):
        if not messages:
            return [], []
        docs = [_tokenize(content_to_text(m)) for m in messages]
        vectors = _tfidf_vectors(docs)
        focus = vectors[self._focus_index(messages)]
        relevance = [_cosine(v, focus) for v in vectors]

        droppable = [i for i, m in enumerate(messages) if _role(m) not in self.pin_roles]
        # Least relevant first; ties broken by oldest first.
        droppable.sort(key=lambda i: (relevance[i], i))
        return _drop_in_order(
            messages, budget, count, droppable,
            lambda m: "low semantic relevance to recent context",
        )
