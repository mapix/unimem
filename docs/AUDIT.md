# Pre-release audit (2026-10-21)

Independent review of unimem at commit `e0b1edf` (crate 0.4.0, engine/paper "v0.5"), done before the first public
push. Scope: build and tests, reproduction of the headline numbers, the paper (`docs/paper/unimem.tex`), a code
review of the gateway hot path, an end-to-end smoke test of the binary, and repository hygiene.

Severity: **[H]** must fix before release · **[M]** should fix or disclose · **[L]** minor / cosmetic.
Status: **FIXED** in the audit commit · **OPEN** remains.

## 1. Build and tests

| check | result |
|---|---|
| `cargo fmt --check` | pass |
| `cargo clippy --all-targets -- -D warnings` | pass (0 warnings) |
| `cargo build --release` | pass, 0 warnings |
| `cargo test --release --all-targets` | 128/128 before the audit, **130/130** after (2 new tests) |

## 2. Reproduced numbers

All re-runs: release build, the same 4-core aarch64 VM as the paper, heuristic decider, hash embedder.

| experiment | claimed | re-run | verdict |
|---|---|---|---|
| E1 heldout2, unimem-full R@1 / R@5 / MRR | 0.555 / 0.688 / 0.616 (v0.3/v0.4, README + abstract) | 0.555 / **0.693** / **0.617** | matches the committed **v0.5** run (`bench/results/v0.5/e1-heldout2-hash`) exactly; the README quoted v0.4 → README now shows v0.5 with v0.4 in parentheses |
| E1 cross-harness R@5, unified vs silo | 0.664 vs 0.000 | **0.671** vs 0.000 | same as above (v0.5) |
| E1 silo R@1 / MRR | 0.243 / 0.257 | 0.239 / 0.254 | v0.5 run; README updated |
| E1 write path (supersede P/R, scope acc., kind acc.) | 0.889 / 0.640, 0.944, 0.456 | identical | ok |
| E2 interception assertions | 1060/1060 | 1060/1060 (912 scripted + 148 conversation leak/framing) | ok |
| E2 prompt cache | 42/42 stable, cacheable 0.987 | 42/42, 0.987 | ok |
| E2 added TTFB, stream + recall p50 / p95 | +1.15 / +3.1 ms | +0.98 / +2.95 ms (and +1.03 / +2.75 on a second run) | ok (noise) |
| E2 added TTFB, JSON p50 | +0.37 ms | +0.32 / +0.36 ms | ok |
| E2 naive injectors | NaiveSystem 9/36 stable, prefix 0.825 | 10/36, 0.835 | ok (data-dependent, v0.5 notes) |
| E5 duplicates removed by the ledger | 128 → 0 | 130–132 → 0 | qualitative claim holds |
| E5 heavy-notes peak share | 7.0% → 3.5% | 9.2% → 4.2% | **differs**; see A3 |
| E5 flush saves planted facts | 5/6 vs 0/6 | 5/6 vs 0/6 (3 runs) | ok |
| Std: LoCoMo R@10, unimem-topk-e5 − hybrid | +0.035 [+0.022, +0.047], 121/53 | identical in `v0.5/std/significance.txt` | ok |
| Std: LME test turn R@10 unimem-prior-e5 | 0.852 vs hybrid 0.828 | identical in `tables.test.md` | ok |
| Std: LME full-set turn R@10 | 0.862 vs 0.838 | identical in `tables.md` | ok |

The standard benchmarks were not re-run: they need the ~300 MB datasets and an e5 cache. Only the committed JSON and
text outputs were checked against the paper.

**Paper numbers spot-checked against sources (34):** all budget constants and config defaults (1200/1500/3000/0.05, 5,
400, 0.7, 1500 ms, k=5, 8, 6; distill 600/30/0.1/24000); every decision threshold (0.4, 0.6, 0.8, 0.7, 0.6, 0.7, 0.35,
0.12, 0.4·max, 0.3, z ∈ [−3, 8], RRF k=60, anchors 30, score 14, nodes 40, near-dup 0.97/0.95/b=0.75/Jaccard 0.5);
both prior-weight rows of Table 3 (`engine.rs`); test totals (128 tests, 16,217 / 4,790 / 3,552 lines); Tables 4 and
5 (from `tables.test.md`, `significance.txt`, `FINAL.md`); the E5 table (from `v0.4/e5_long_session.json`); cache
bytes 2,251,877 / 1,533,536; E4 scaling (7.654/17.241/4.751/7.226 ms, 103.71 MB, 77.3 MB, 3.582 ms); write-path
rejection counts 1,658 / 612 / 600 / 408 / 25 / 13; e5 over-merge 22,763 / 2,662 → 8,137 / 433; error analysis 68 /
32 / 0.333 / 0.788; v0.2 overfitting (0.444 → 0.447, 5.6% → 21.3%, 5/26 → 1/26, 44.4% → 50.8%, 0.500 → 0.321,
0.486 → 0.425); matched-depth numbers (0.916/0.921, 0.826/0.850, 0.520/0.456); Jev/LLM timeouts 8 s / 15 s. All
match, except the items listed under A.

**Bibliography:** all 15 arXiv ids resolve, with matching titles and authors (checked via the arXiv API). All 5 GitHub
repositories cited (jev-mem, memsearch, OpenWolf, BladeX, NautGate) exist.

## 3. Findings

### A. Paper and documentation

- **A1 [H] FIXED: version inconsistency.** The README presented v0.4 numbers and v0.4 conclusions ("significantly
  behind dense retrieval"), while the paper and abstract claim the v0.5 result ("level with or above BM25+e5 RRF").
  The CHANGELOG had no v0.5 entry. `Cargo.toml` is still 0.4.0. README now shows v0.4 and v0.5 side by side and
  labels which rows include the dev half. CHANGELOG has a v0.5 entry under Unreleased. *Judgment call:* whether to
  bump the crate to 0.5.0 before tagging. `docs/PUBLISH.md` tags `v0.4.0`, while the paper says "software v0.5".
  Recommend bumping to 0.5.0 and tagging `v0.5.0`. Not changed here because it is a release decision.
- **A2 [M] FIXED: E1 numbers in README/abstract not reproducible from the shipped code.** The shipped code gives
  R@5 0.693 and cross-harness 0.671, not 0.688 and 0.664. Added the v0.5 row to Table 6 and to the abstract, and a
  note in RQ2.
- **A3 [M] FIXED (disclosed): E5 is v0.4-only and not deterministic.** Re-running on the v0.5 code gives different
  magnitudes (heavy notes 9.2% → 4.2% instead of 7.0% → 3.5%). Across runs, cumulative injections vary by up to
  ~10%, because the background flush distillation races the request loop. Added a "Re-run with the v0.5 code"
  paragraph to RQ4 and a sentence to the README. *OPEN:* make E5 deterministic (await the flush task in the bench),
  and regenerate `bench/results/v0.5/e5_long_session.json` plus Figure 4.
- **A4 [M] FIXED: "heldout2 predates the v0.3 commit".** Only the file *timestamp* predates it. In git, heldout2 was
  first committed together with the v0.3 evaluation (`9dcc43a`, after `8a679a8`). Reworded.
- **A5 [M] FIXED: concurrent work.** The paper called BladeX and NautGate "concurrent work". Their repositories were
  created in May 2026, five months before unimem, so they are prior art. Reworded to "Both predate this work". The
  paper still does not compare against them. A reviewer will ask for at least a qualitative mechanism comparison
  (judgment call, left open).
- **A6 [M] FIXED: `sufficient` misfire rate.** The paper and README said "misfires on 45% of answerable queries".
  That is v0.3/v0.4. In v0.5 it is 8.7% (hash) and 9.6% (e5) on heldout2, but **89.9%** with e5 on devmem and
  heldout, so the threshold is not calibrated across embedders. The paper's limitation paragraph now says this.
  The README limitation bullet is updated the same way. *OPEN:* re-calibrate `T_SUFFICIENT` per embedder family.
- **A7 [L] FIXED: abstract overclaim.** "the provider's prompt-cache prefix never breaks" became "every upstream
  request stays an exact byte-prefix extension". Only byte prefixes were measured, and the paper itself lists the
  20-block lookback as a risk.
- **A8 [L] FIXED.** Error analysis said "4 golds were destroyed" but listed 3 (2+1). Now "4 trace to the write path
  (2 + 1, plus a duplicate-labelled note)". The Table A.2 caption now explains that scope_isolation n=22 includes 4
  forbidden-only queries, since its R@5 values (x/18) do not fit n=22. The line-count statement now names the commit
  it was measured at.
- **A9 [L] FIXED.** CHANGELOG referenced the removed `docs/paper.md`. `Cargo.toml` excluded the non-existent
  `docs/paper.pdf` instead of `docs/paper/`. `docs/PUBLISH.md` said the `rustfmt` job is allowed to fail, but CI
  already enforces it.
- **A10 [M] OPEN, judgment call: is the cross-harness 0.664 vs 0.000 a result?** It is true by construction, as the
  paper says. It is the headline number of the abstract, though, and it measures the store layout, not the gateway.
  A reviewer will discount it. Consider leading with the 42/42 and 1060/1060 mechanics and the E5 bounds instead.
- **A11 [M] OPEN, judgment call: test-split framing.** The v0.5 standard-benchmark gain uses weights fitted on the
  LongMemEval dev half. The full-set numbers (0.979 / 0.862) that README and §RQ1 also quote include the fitting
  questions. The README now says so. The paper says so only implicitly ("On the full 419-question set").
- **A12 [L] OPEN.** The relevance-prior intercept was re-fitted on devmem + heldout (RESULTS.md §7.1). Those are
  in-sample suites, but they are also the E1 dev suites, so `devmem`/`heldout` E1 numbers for v0.5 are not
  generalisation evidence. The paper already treats them as in-sample. Fine as is.
- **A13 [L] OPEN.** English is good throughout: no typos found, no doubled words, and LaTeX builds with no
  overfull boxes or undefined references (16 pages). Two style nits remain: the abstract is one 300-word paragraph,
  and RQ1 "Published numbers" is dense. Optional.

### B. Code (gateway hot path)

- **B1 [H] FIXED: browser cross-site write and DNS rebinding.** The gateway has no auth and no Origin/Host
  checks. A web page in the user's browser could `POST` `text/plain` JSON-RPC to `http://127.0.0.1:4777/mcp`, which
  needs no CORS preflight, and store a memory. Verified in the smoke test: "Saved m_… as fact in
  project:github.com/acme/widget". Every agent would then see that note, which is a persistent prompt-injection
  vector. A DNS-rebinding page could also read `/ui/api/*` (all memories). Fix: a router-wide middleware
  (`browser_guard` in `src/gateway/mod.rs`) refuses requests with a non-loopback, non-same-origin `Origin`, or a
  `Host` that is a domain name other than `localhost`. Harness, CLI and UI traffic is unaffected: smoke test and
  130 tests pass. Opt-out: `UNIMEM_ALLOW_ANY_ORIGIN=1`. Unit test:
  `browser_guard_blocks_foreign_origins_and_rebinding_hosts`.
- **B2 [M] FIXED: unbounded decompression.** `decode_body` decompressed zstd/gzip/deflate bodies (up to 64 MB
  compressed) with no output limit, so a few KB could expand to many GB. Output is now capped at 256 MB. Above the
  cap decoding fails and the request is passed through raw (fail-open). Unit test:
  `decode_body_roundtrips_and_caps_decompression_bombs`.
- **B3 [L] FIXED: unbounded MCP session map.** `mcp::sessions()` gained an entry on every `initialize` and lost one
  only on `DELETE`. It is now capped at 4096 entries (cleared when full; the only effect is a fallback harness label).
- **B4 [M] OPEN: blocking SQLite and CPU work on async workers.** `Store` holds one `parking_lot::Mutex<Connection>`.
  Every engine call (recall, write, overlay get/put, session state) locks it synchronously inside async handlers, and
  brute-force cosine over all notes (~5 ms at 10k notes) also runs on the runtime threads. Under parallel subagents
  and the distiller, this stalls tokio workers and serialises requests, including SSE forwarding of unrelated
  streams. The 1.5 s `tokio::time::timeout` around recall cannot preempt a blocking section. Suggested: run engine
  calls through `spawn_blocking` (or a dedicated DB thread with a channel), and use a read connection pool in WAL
  mode. The paper lists "Concurrency" as a limitation, but not the head-of-line blocking of streaming.
- **B5 [M] OPEN: overlay replay does one SQLite lookup per message per request.** `replay` calls `get_overlay`
  for every user and assistant message, so a 500-message conversation costs about 1,000 point queries under the
  global mutex on every request (prefix-cached in the provider but not in unimem). Consider caching anchors per
  session in memory.
- **B6 [L] OPEN: `session_state` cache (`Store::sstate`) grows until the 6-hourly GC clears it.** It is bounded in
  practice (one entry per thread), but the GC drops every entry, including active ones, which only costs a DB
  reload. Acceptable.
- **B7 [L] OPEN: `passive_ok` cache clears at 50k entries.** Bounded, fine.
- **B8 [L] reviewed, no issue.** Untrusted-JSON access in `anthropic.rs`, `responses.rs`, `chat.rs` and `turn.rs`
  uses `Value` indexing, which yields `Null` on a type mismatch, plus `as_*().unwrap_or(..)`. No `unwrap()` on
  request data. Slices (`&first_user[..16]`, `&input[len-n..]`, `keys[i-1]`) are guarded or index ASCII hex. The
  smoke test fed 14 malformed bodies (`messages` as a string or null, content of mixed types, `tools` as a string,
  `model` as an integer, 9k-character CJK content, invalid zstd) to all three dialects. No panics; every request
  was forwarded (fail-open), and `/health` stayed up.
- **B9 [L] reviewed, no issue.** `budget.rs`: `Budget::new` caps are correct, and the floors (64/128) cannot exceed
  the session cap for any window ≥ 2.6k tokens. `truncate_tokens` respects char boundaries.
  `Turn::remaining` uses `saturating_sub`.
- **B10 [L] OPEN: heuristic scoping.** In the smoke test, the heuristic decider stored "Run database migrations
  with make migrate, never with raw sqlx", written from inside the repo, as **global**. This is the known kind/scope
  weakness of the heuristic decider, but it is visible on the first command a new user is likely to try.

### C. End-to-end smoke test (binary, `UNIMEM_LISTEN=127.0.0.1:4791`, temp data dir and config, Python mock Anthropic upstream)

| check | result |
|---|---|
| CLI `remember` / `recall` / `list` / `stats` | pass (project detected from the git remote: `github.com/acme/widget`) |
| streaming prompt turn with tools; the model calls `unimem_recall` | pass: one merged message, contiguous indices, a single `message_start` / `message_delta` / `message_stop`, no `unimem_*` or `<unimem` visible, `stop_reason` end_turn, usage = last round |
| mixed turn (`unimem_recall` + `Bash`) | pass: client sees text + `Bash`, `stop_reason` tool_use |
| tool-result turn, then a JSON turn | pass; `x-unimem-usage` header present |
| upstream requests | system prompt untouched; unimem tools appended after the harness's; 4/4 consecutive requests are byte-prefix extensions |
| `GET /ui`, `/ui/api/overview` | 200, self-contained HTML (no CDN), JSON overview |
| `/mcp` initialize / tools/list / tools/call | pass; empty batch and bad JSON give 400 |
| malformed bodies (14 cases, 3 dialects) | no panic, fail-open passthrough |
| cross-site `text/plain` POST to `/mcp`; rebinding `Host` | **wrote a memory / read all memories before B1**; 403 after |

### D. Repository hygiene

- No secrets. The only key-like strings are AWS's published dummy keys in bench JSON, and a synthetic `sk-…` in a
  `distill.rs` unit test.
- No VM paths in non-JSON tracked files (`git grep /sessions/ -- ':!*.json'` matches only the PUBLISH.md check
  itself). `bench/results/*/SUMMARY.md` regenerated by a local re-run embeds absolute paths; the committed ones are
  clean.
- LICENSE (MIT, 2026 mapix) is present. Cargo metadata (description, license, repository, keywords, categories,
  readme) is complete. `exclude` is fixed (A9).
- All relative README/CONTRIBUTING/docs links resolve.
- `.gitignore` covers target, db files, bench caches and LaTeX by-products. `bench/results/FINAL.md` and
  `COMPARISON.md` are listed in `.gitignore` but are tracked (harmless, since they are already tracked, but
  confusing). *OPEN [L]:* drop those two lines.
- *OPEN [L]:* the commit author is the placeholder `unimem <a@b>`. `docs/PUBLISH.md` already gives the rewrite
  command. Run it before pushing.
- *OPEN [L]:* `bench/results/*/env.json` and SUMMARY.md show `rustc` as empty in fresh runs (the bench does not
  find `rustc` on PATH under rustup). Cosmetic.

## 4. What remains (priority order)

1. Decide the version: bump the crate to 0.5.0 and tag `v0.5.0` (A1).
2. Move blocking SQLite and cosine work off the async workers (B4), and cache overlay anchors (B5).
3. Make E5 deterministic, and regenerate the v0.5 E5 results and figure (A3).
4. Calibrate `sufficient` per embedder (A6).
5. Consider re-framing the abstract around the mechanics rather than the by-construction 0.664 vs 0.000 (A10), and
   add a qualitative comparison with BladeX and NautGate (A5).
6. Rewrite the commit author before pushing (D).
