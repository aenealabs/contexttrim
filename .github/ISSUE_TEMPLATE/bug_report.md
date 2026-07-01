---
name: Bug report
about: Something isn't working correctly
labels: bug
---

## Description

A clear description of the bug.

## Reproduction

```python
from contexttrim import ContextManager
from contexttrim.strategies import RoleWeighted

ctx = ContextManager(token_budget=..., strategy=RoleWeighted())
ctx.add({"role": "...", "content": "..."})

trimmed = ctx.fit()

# What you expected:

# What you got:
```

## Environment

- contexttrim version:
- Python version:
- OS:

## Traceback

```
Paste the full traceback here, if any.
```
