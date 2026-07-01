# Contributing to contexttrim

Thank you for taking the time to contribute.

## Ground rules

- **Zero external dependencies.** contexttrim must remain pure Python stdlib. PRs that introduce any third-party import will not be merged.
- **Python 3.9+.** All code must run on Python 3.9 through the latest stable release.
- **Tests required.** Every new feature or bug fix must include a corresponding test. The CI matrix runs on 3 operating systems × 5 Python versions — please run tests locally before opening a PR.
- **Keep it focused.** contexttrim does one thing: fit a conversation to a token budget. Feature requests outside that scope belong in a separate package.

## Setting up a development environment

```bash
git clone https://github.com/aenealabs/contexttrim
cd contexttrim
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

To run against a specific Python version, use `hatch`:

```bash
pip install hatch
hatch run test
```

## Architecture

contexttrim is small and layered — read the modules in this order:

1. `src/contexttrim/_types.py` — the `Message` shape, `Dropped` record, and the `Strategy` base class.
2. `src/contexttrim/_tokens.py` — the default character-heuristic token counter.
3. `src/contexttrim/strategies.py` — the six strategies and the shared drop helper.
4. `src/contexttrim/_manager.py` — `ContextManager` and `FitReport`.
5. `src/contexttrim/__init__.py` — the public surface.

## Adding a strategy

1. Subclass `Strategy` in `strategies.py` and implement `fit(self, messages, budget, count) -> (kept, dropped)`.
2. Never mutate the input messages — return new lists (and new dicts if you rewrite content).
3. Add it to `strategies.__all__`.
4. Add tests under `tests/` covering selection order, pinning (if any), and the no-mutation guarantee.
5. Update `README.md` and `CHANGELOG.md`.

## Submitting a pull request

1. Fork the repository and create a branch: `git checkout -b fix/my-fix` or `feat/my-feature`.
2. Make your changes and add tests.
3. Run `pytest tests/ -v` — all tests must pass.
4. Open a pull request against `main` with a clear description of what changed and why.

## Reporting bugs

Open an issue using the **Bug report** template. Include the Python version, OS, the strategy and budget used, and a minimal set of messages.

## Suggesting features

Open an issue using the **Feature request** template. Explain the use case, not just the solution.

## Code style

contexttrim uses no formatter or linter by choice to keep contributor setup minimal. Please follow the style of the surrounding code: 4-space indentation, descriptive variable names, module-level docstrings on every file.
