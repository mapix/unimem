# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break things).

Benchmark numbers are from the offline configuration (heuristic decider, hash embedder). E1 figures are on the
out-of-sample `heldout2` set unless noted otherwise. Details: `bench/results/`.

## [0.5.0] — 2026-10-21

### Added
- Open-source readiness: this README, the license, contributing and security docs, a CI workflow, issue and PR
  templates, and a commented `examples/config.toml`.
- v0.5 engine work (commit `7c8742a`): the relevance prior uses a z-scored cosine
  and an RRF term, with weights fitted per embedder family on dev data only (`PRIOR_HASH`, `PRIOR_DENSE`);
  `Depth::TopK` ranking mode for explicit `k` (agent tool, CLI `-k`, UI, `/api/recall`); the ONNX e5 embedder as a
  production option (`unimem models download e5-small`). Results: `bench/std/RESULTS.md` §7 and `bench/results/v0.5/`.
- Pre-release audit (`docs/AUDIT.md`): requests from web pages are refused (foreign `Origin`, or a non-local `Host`
  as used by DNS rebinding; override with `UNIMEM_ALLOW_ANY_ORIGIN=1`); decompressed request bodies are capped at
  256 MB; the MCP session map is bounded.

### Results (v0.5)
- LongMemEval_S test split / LoCoMo: the e5 ranker is level with or above BM25+e5 RRF (LoCoMo R@10 +0.035,
  95% CI [+0.022, +0.047]; LongMemEval turn R@10 +0.025, borderline).
- heldout2 (hash, adaptive): R@1 0.555, R@5 0.693, MRR 0.617, cross-harness R@5 0.671.

## [0.4.0] — 2026-10-21

Implements the P0/P1 findings of the design review (`docs/REVIEW.md`).

### Added
- **Token budgets** (`budget.rs`). They apply per message (min(1200 tok, 0.6% of the window)), to the preamble
  (min(1500, 1%)), per tool result (3000, k ≤ 8), to tool-error recall (≤ 2 notes / 400 tok), and to the share of a
  conversation's live context that unimem may hold (5%). Context windows come from a model table, which you can
  override with `inject.context_windows`.
- **Cross-turn de-dup ledger.** A note that is still visible in the conversation is not injected again.
- **Compaction-aware re-injection.** Compaction is detected by four signals: message-count drop, a changed first
  message, a summary marker, or zero anchor hits. When it fires, the ledger is reset and the preamble is sent again.
- **Tool-error recall (R4).** Error-looking tool output triggers a gated, rate-limited recall.
- **Pre-compaction flush.** At 70% of the window, the session is distilled in the background and the agent is
  nudged once to save durable facts.
- **Correction hints.** When the user corrects the agent, a short hint suggests saving it, and the episode gets an
  importance boost.
- New distiller triggers: session end (a new session in the same project and harness) and a rate-limit gate
  (`distill.min_rate_limit_remaining`). Episodes now have a TTL (`episode_ttl_days`).
- mem0-style ADD / UPDATE / DELETE / NOOP distiller operations when a model is configured.
- `Embedder::embed_query`: queries get e5's asymmetric `query:` / `passage:` prefixes.
- Optional local ONNX transformer embedder (`--features onnx`, `kind = "onnx"`).
- **Web UI** at `/ui`. It has per-kind memory views, project boards, the memory graph, a timeline, a recall
  playground and an activity log. The JSON API is under `/ui/api/`. New commands: `unimem ui` and
  `unimem demo-seed`.
- Standard benchmarks: LongMemEval_S and LoCoMo retrieval (`unimem-std-bench`, `bench/std/`), cross-checked against
  the official LongMemEval code.
- Bench E5: a long-session scenario covering budgets, de-dup, compaction, tool-error recall and flush.

### Changed
- **Usage reporting is now honest.** A merged response reports the *last* round's `input_tokens` / `cache_*` (the
  real context size) and the *summed* `output_tokens`. True totals across rounds are sent in the `x-unimem-usage`
  header or SSE comment.
- The passive path (preamble and passive recall) no longer injects notes that read as instructions or authorization
  aimed at the agent, nor superseded or stale notes.
- Rendered notes are capped (600 chars of content, 400 for `why`), with a pointer to the full text.
- Near-duplicate thresholds are now set per embedder (`near_dup_cos`, `cos_baseline`). This fixes e5 merging
  unrelated turns as duplicates.

### Fixed
- The distiller marked all pending episodes as distilled but read only the newest 24k characters. It now works in
  chunks.
- After a compaction, the preamble was never re-sent.

### Results
- E5: duplicate injections went from 128 to 0. With heavy notes, peak unimem context went from 7.0% to 3.5% of the
  window. The flush saved 5 of 6 planted facts before compaction (0 of 6 without it).
- E1 unchanged from v0.3. Passive "rejected-note returned" went from 6.6% to 1.2%.
- Gateway: +1.15 ms p50 streaming TTFB, +0.37 ms JSON.

## [0.3.0] — 2026-10-20

### Added
- Calibrated hybrid retrieval: BM25 (FTS5) and vector anchors scored by a logistic `relevance_prior`, with an
  adaptive cut-off.
- Subject-aware supersede and duplicate detection; generalized write screens (secrets, transient status,
  injection); a `sufficient` / `missing` signal on recall.
- `hash2-512` offline embedder: stems, compound-identifier parts and bilingual concept features (`lexicon.rs`).
- Out-of-sample evaluation set `heldout2` (a new developer, 6 new repos, a new harness), the final evaluation
  (`bench/results/FINAL.md`), and a technical report (then `docs/paper.md`, now `docs/paper/unimem.tex`).

### Changed
- Renamed the project from `uni-mem` to `unimem`.

### Results (heldout2)
- R@1 0.555, R@5 0.688, MRR 0.616, precision 0.423, stale returned 16.3%.
- R@5 is significantly better than v0.1 (+0.119) and v0.2 (+0.138). It is not significantly better than top-5
  cosine with the same embedder.
- Supersede recall rose from 0.24 to 0.64 with 0/12 distractor errors. The secret screen caught 10/10.
- Known gaps: injection screen 1/7 out-of-sample, cross-language R@5 0.33.

## [0.2.0] — 2026-10-20

### Fixed
- `TCP_NODELAY` on the gateway: added streaming TTFB went from +42.8 ms to +0.76 ms.
- Cross-session anchor collisions (the probe went from 10/10 to 0/10).
- Over-eager supersede (subject check), global-preference scoping, the CJK gate on passive recall, the write
  screens, the sufficiency signal, and project detection walking up to the git root (E3 went from 35/37 to 37/37).

### Added
- `heldout` evaluation set, and the v0.1 vs v0.2 comparison (`bench/results/COMPARISON.md`).

### Known regressions
- Chinese recall, false abstention (21%) and supersede recall all got worse. Fixed in v0.3.

## [0.1.0] — 2026-10-20

### Added
- Initial release: the memory engine (SQLite, FTS5, vectors, memory graph), the System One decider chain (Jev, LLM,
  heuristic), project detection and the `unimem_*` tool face.
- Gateway with three dialects. The Anthropic Messages dialect has overlay replay and streaming tool interception.
  The OpenAI Responses (Codex) and Chat Completions dialects were also added.
- CLI, MCP endpoint (`/mcp`), passive distiller, configuration.
- Evaluation harness (`unimem-bench`, E1–E4) and the `devmem` dataset.

[Unreleased]: https://github.com/mapix/unimem/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mapix/unimem/releases/tag/v0.4.0
[0.3.0]: https://github.com/mapix/unimem/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mapix/unimem/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mapix/unimem/releases/tag/v0.1.0
