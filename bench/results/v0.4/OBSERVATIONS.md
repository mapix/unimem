## v0.4 notes (vs v0.3)

Same machine, same offline config (heuristic decider, hash embedder). E1 here is the out-of-sample `heldout2` set; compare with `bench/results/v0.3-heldout2`.

**E1 (heldout2).** `unimem-full` and `unimem-lite` are identical to v0.3 on every metric (R@1 0.555, R@5 0.688, MRR 0.616, scope leak 2.5%), and the write-path metrics are unchanged. The budgets do not bind at bench note sizes, and the `embed_query` change is a no-op for the symmetric hash embedder. `unimem-passive` changes by design, because the passive path now drops notes that read as instructions or authorization aimed at the agent, plus superseded and stale notes:
- rejected-note returned falls from 6.6% to 1.2% of queries;
- R@5 stays at 0.688, R@1 goes from 0.555 to 0.560, MRR from 0.616 to 0.618;
- abstention on unanswerable queries rises from 5% to 10%;
- false abstain rises from 2.8% to 3.2%.

**E2.** All 912 assertions pass. Cache stability for the unimem conversation is 42/42 fully prefix-stable transitions, with a cacheable byte ratio of 0.987 (v0.3: 0.984). Upstream bytes fall from 2.46 MB to 2.25 MB because long notes are capped at render time and recall-tool output is token-capped.

Usage semantics changed on purpose. Merged responses now report the *last* round's `input_tokens` / `cache_*`, which is the real context size, and the *summed* `output_tokens`. True totals across all rounds are sent in the `x-unimem-usage` header (JSON) and, for streams, in an SSE comment `: x-unimem-usage {…}`. They are also written to the `usage` trace. The gateway tests were updated for this. The E2 bench does not assert usage, so no bench expectation changed.

Added latency against v0.3 built and run on this VM in the same session:
- stream TTFB p50: +1.15 ms in v0.4 vs +1.27 ms in v0.3;
- JSON TTFB p50: +0.37 ms in v0.4 vs +0.23 ms in v0.3.

The extra ≈0.14 ms on JSON is the per-request session-state bookkeeping and the request scan for note ids. It holds only because the passive-path screen result is cached per note version. Without that cache the screen cost about 0.6 ms per preamble.

**E5 (new).** One 120-request session with a forced compaction at request 60, 55 tool turns, 11 tool errors and a 200k window.

- **Duplicate injections.** The de-dup ledger removes all of them: 128 → 0 at the corpus's normal note sizes, and 38 → 0 with heavy notes (40 extra ≈3 kchar notes).
- **Peak unimem footprint with heavy notes.** From 14.0k to 7.1k tokens of live context, i.e. 7.0% → 3.5% of the window. The ledger alone gives 10.2k; budgets plus ledger give 7.1k. The budgets bound each turn; the session cap (5%) never bound in this run.
- **Cumulative new injection with heavy notes.** From 17.3k to 9.1k tokens.
- **At normal sizes the budgets never bind**, so no-budget = v0.4.
- **Compaction.** The compaction request reached the upstream byte-identical. The preamble came back on the first request after the summary. Live unimem context dropped to about 1.6k at request 61 and grew again.
- **Tool-error recall.** It fired on 6 of the 11 error turns, each time with the root-cause note. On 3 more turns the note was already live, so 9/11 errors had their explanation in context. The two misses are the uvicorn "address already in use" turns: that error shares only the port number with the note (n014), and the rare-token gate ignores numbers on purpose. Benign tool turns got 0 injections out of 44. Without the ledger the same 9 are covered, but by re-injecting notes that are already in context.
- **Pre-compaction flush.** The nudge fired once, at request 54 (estimated context ≈ 140k tokens, 70% of 200k). The background heuristic distillation it triggered had stored 5 of the 6 facts the user stated before the compaction by the time of the compaction; without the flush the count was 0. The sixth fact ("Note that … uses the EU Stripe sandbox account") was stored too, but only after the compaction, by a later background pass.
- **Corrections.** All 3 were hinted. A question containing "应该用" is no longer taken for a correction.
