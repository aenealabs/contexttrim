# contexttrim

[![PyPI](https://img.shields.io/pypi/v/contexttrim?color=blue)](https://pypi.org/project/contexttrim/)
[![Python](https://img.shields.io/pypi/pyversions/contexttrim)](https://pypi.org/project/contexttrim/)
[![CI](https://img.shields.io/github/actions/workflow/status/aenealabs/contexttrim/ci.yml?label=CI)](https://github.com/aenealabs/contexttrim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)

**Pluggable context-window management for LLM agents.**

When a conversation exceeds the model's token limit, most code drops the oldest messages (FIFO) — often dropping a critical system prompt while keeping ten redundant tool results. contexttrim lets you choose *what* gets dropped and *why*, with swappable strategies. Zero dependencies, no ML, no tokenizer required.

```python
from contexttrim import ContextManager
from contexttrim.strategies import ImportanceWeighted

ctx = ContextManager(token_budget=8_000, strategy=ImportanceWeighted())
ctx.add({"role": "system", "content": "You are a helpful assistant."})
ctx.add({"role": "user", "content": "Find me flights to NYC."})
ctx.add({"role": "tool", "content": "<very long search result>"})

trimmed = ctx.fit()              # a new list that fits the budget
report = ctx.last_fit_report()   # what was dropped and why
print(report.dropped, report.tokens_used)
```

## Why contexttrim?

Naive FIFO truncation throws away the wrong things. contexttrim gives you a `ContextManager` plus a set of strategies that make deliberate, inspectable decisions — and it's pure Python stdlib, so it adds nothing to your dependency tree.

## Installation

```bash
pip install contexttrim
```

Requires Python 3.9+. No other dependencies, ever.

## Strategies

Import from `contexttrim.strategies`:

| Strategy | What it does |
|---|---|
| `RecencyDrop` | Drop the oldest messages first. Fast, simple, often wrong. |
| `MiddleDrop` | Drop from the middle — models attend least there ("lost in the middle"). Head and tail preserved longest. |
| `RoleWeighted` | Score by role; drop lowest-scored first. `system` pinned by default. |
| `ImportanceWeighted` | `role_weight × recency_decay^age ÷ (1 + length_penalty·tokens)`. Keeps short, recent, high-role messages. |
| `ToolResultMerge` | Merge redundant adjacent tool results (dedup), then truncate the largest if still over budget — no conversational context dropped. |
| `SemanticCluster` | Drop messages least topically relevant to the recent conversation, via TF-IDF cosine similarity (pure stdlib, no ML). `system` pinned. |

```python
from contexttrim.strategies import RoleWeighted

RoleWeighted(
    role_weights={"system": 10.0, "user": 2.0, "assistant": 1.0, "tool": 0.5},
    pin_roles=frozenset({"system"}),   # never dropped
)
```

## The fit report

Every `fit()` records what happened:

```python
trimmed = ctx.fit()
report = ctx.last_fit_report()

report.kept           # the messages that survived
report.dropped        # list of Dropped(message, reason)
report.tokens_used    # total tokens of the kept messages
report.tokens_budget  # the budget you set
report.fits           # False only if pinned messages alone exceed the budget

for d in report.dropped:
    print(d.reason, "->", d.message["role"])
```

## Token counting

By default, contexttrim estimates tokens with a deterministic ~4-characters-per-token heuristic — zero dependencies. Inject your own counter for exact counts (e.g. tiktoken):

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

ctx = ContextManager(
    token_budget=8_000,
    strategy=ImportanceWeighted(),
    token_counter=lambda m: len(enc.encode(m.get("content", "") or "")),
)
```

## Message format

Messages are plain dicts in the common OpenAI/Anthropic shape — `{"role": ..., "content": ...}`. contexttrim never mutates them; `fit()` returns a new list. Non-string `content` (e.g. Anthropic content blocks) is serialized for token counting.

## Writing your own strategy

Subclass `Strategy` and return `(kept, dropped)`:

```python
from contexttrim import Strategy, Dropped

class DropAssistant(Strategy):
    def fit(self, messages, budget, count):
        kept, dropped = [], []
        for msg in messages:
            if msg.get("role") == "assistant":
                dropped.append(Dropped(msg, "assistant messages disabled"))
            else:
                kept.append(msg)
        return kept, dropped
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

Part of the [aenealabs](https://github.com/aenealabs) AI agent toolkit.
