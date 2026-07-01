# Changelog

All notable changes to contexttrim are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-01

### Added
- `ContextManager` — add/extend/clear messages, `fit()` to a token budget, `total_tokens()`, and `last_fit_report()`
- `FitReport` — `kept`, `dropped` (each a `Dropped(message, reason)`), `tokens_used`, `tokens_budget`, `fits`, `num_dropped`
- Six pluggable strategies in `contexttrim.strategies`:
  - `RecencyDrop` — drop oldest first
  - `MiddleDrop` — drop from the middle ("lost in the middle"), preserving head and tail
  - `RoleWeighted` — drop lowest-scored roles first; pin `system`
  - `ImportanceWeighted` — role weight × recency decay ÷ length penalty
  - `ToolResultMerge` — merge/dedup adjacent tool results, then truncate the largest if still over budget
  - `SemanticCluster` — drop least topically-relevant messages via pure-Python TF-IDF cosine similarity
- `Strategy` base class for writing custom strategies
- Character-heuristic token counter by default (`default_token_counter`), with an injectable `token_counter`
- Zero external dependencies — pure Python stdlib (3.9+)
- 24 unit tests covering the manager, all six strategies, token counting, and no-mutation guarantees

[Unreleased]: https://github.com/aenealabs/contexttrim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aenealabs/contexttrim/releases/tag/v0.1.0
