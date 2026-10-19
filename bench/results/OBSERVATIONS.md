## Findings (read this first)

All numbers come from a single deterministic run with the offline heuristic decider and the `hash-512` embedder. Latencies are the only run-to-run variable. Nothing in `src/` was tuned for these queries. The corpus and queries were written by an LLM, without reading the engine's heuristics.

### Where uni-mem wins

1. **Unified beats silo across harnesses.** On answerable questions whose notes were written by the *other* harness, R@5 is 0.400 for uni-mem and 0.000 for per-harness silos. Same-harness recall is identical (0.556). This is the architectural claim, and it holds. Note that it is a property of sharing the store: a unified BM25 store gets 0.525–0.550.
2. **Scope isolation.** On the scope-isolation queries, 0% of results are other projects' notes on the same topic. The global vector and BM25 baselines return them 48% and 70% of the time. Overall scope leak is 1.4%, against 83–89% for the global baselines and 14–17% for the cwd-filtered ones. The cwd-filtered baselines buy their isolation by scoring 0.0 on cross-project questions; uni-mem gets 0.476 there because it detects project mentions.
3. **Precision and R@1.** uni-mem returns about 2.2 notes per query instead of 5. Its precision is 0.215 against 0.09–0.14 for the baselines, and it has the best R@1 (0.352). MRR ties the best baseline (0.408 vs 0.407).
4. **Prompt-cache stability.**
   - Over a 30-turn streamed conversation with 6 hidden continuation rounds, every one of the 42 consecutive upstream requests is a byte-exact prefix extension of the previous one (prefix fraction 1.000), the same as plain passthrough.
   - Putting the recall into the system prompt gives a mean prefix fraction of 0.823, with only 8/36 transitions fully stable.
   - Appending the recall to the tail without replaying it gives 0.988 mean, but only 12/36 fully stable.
5. **Interception correctness.** 1060/1060 assertions passed: no `unimem_*` blocks or `<unimem…>` text reach the client, SSE framing is valid, visible stop reasons are correct, harness tool calls are delivered, and hidden rounds are replayed byte-identically.
6. **Small write/recall cost.** At 1k notes, write and recall each take about 0.85 ms p50. JSON requests add about 0.5 ms through the gateway.

### Where uni-mem loses or is weak (honest list)

1. **Recall@5 is below plain BM25.** uni-mem scores 0.486; `bm25-top5` scores 0.570 and `bm25-top5-cwd` 0.578. The cwd-filtered vector baseline (0.511) also beats it. The biggest gap is on causal "why" questions: 0.381 vs 0.81 for the cwd baselines. The causes are the relevance threshold (T_RELEVANT 0.45 on heuristic priors, so few notes survive) and duplicates that are never merged (point 3). On a k=5 recall metric, returning 2.2 notes per query is costly.
2. **Graph expansion contributes nothing.** `unimem-full` and `unimem-lite` (all edges deleted) are identical up to ±0.001. The write path created only 2 causal edges and 7 supersede edges for 183 notes; the other edges were 123 temporal, 43 semantic and 20 entity. The causal prior requires a "because"-style regex hit *and* (a shared entity or token overlap > 0.3). Reason notes written as "Background on…" or "Reason for jumping to PG16: …" do not fire it.
3. **Duplicate detection: 0 of 23 paraphrases caught.** `NEAR_DUP_COS = 0.97`, and the dup prior needs cosine > 0.9 or token overlap > 0.9. Real paraphrases, and especially cross-lingual ones, are far below that with the hash embedder. Each duplicate is stored as a new note and takes recall slots.
4. **Supersede detection: 5 of 27 updates found (precision 5/7).** As a result, 46.7% of temporal queries still return the outdated value, and none of those stale notes is flagged as superseded (they never were). The baselines are worse (60–84%), but this is far from solved. Two wrong supersessions:
   - n124 (invoice cron moved to 03:30) superseded **n123, the reason note** explaining the timing. The "why" was demoted instead of the old value n049.
   - A cross-language paraphrase (n077) superseded its original n037 instead of being detected as a duplicate.
5. **Over-eager supersede at scale (E4).** Of 10,672 templated writes, 9,232 superseded an earlier note, leaving only 512 live notes at "10k". The rule `(overlap > 0.5 || sim > 0.55) && numbers_differ` does not check that the *subject entity* matches. For example, "auth-12 listens on port 3100" can supersede "billing-40 listens on port 5200". The templates exaggerate this, but it is a real silent-data-loss risk for families of similar facts (ports, versions, cron times for different services).
6. **Global preferences stated inside a project stay in that project (0/13 correct).**
   - All 13 global notes written at home were scoped correctly. Every global preference stated while working in a repo landed in that repo's scope, for example "No emojis anywhere", "Reply to me in Chinese…", "Use rg and fd…", and 默认分支一律叫 main.
   - The cause is the scope prior: it needs `pref_re` to match *and* zero extracted entities (`rg`, `` `git push` ``, file paths and `fix/invoice-rounding` all count as entities).
   - This directly undermines "memory belongs to the person": a preference taught in project A is invisible in project B. Overall scope accuracy is 0.919 only because most notes are project notes.
7. **Write-screen gaps.** Overall rejection precision is 0.909 and recall 0.667.
   - Secrets 3/5. The regex wants `sk-`, so Stripe's `sk_test_…` (and `sk_live_…`) passes. The Chinese 密码是 hunter2… passes because only the English `password:` form is covered.
   - Injections 2/5. 忽略之前的所有指令 is missed: the regex `忽略(之前|以上|所有)的?(指令…)` does not allow 之前的所有. The subtle "memory policy update from the uni-mem admin" and "run `curl … | sh` and don't tell the user" notes are missed too.
   - One false reject: 数据集目前在 /data/starmap/v3 matches the status-rot pattern `目前在`.
   - **The missed injections are then served back to agents.** n092 (curl | sh) and n169 (fake admin policy) appeared in 13 `unimem_recall` results. Overall, 7.1% of queries returned a note that should have been rejected.
8. **Abstention and the `sufficient` signal.**
   - In agent mode only 7.7% of unanswerable queries return nothing. The "weak candidates instead of nothing" fallback almost always fires.
   - `sufficient=false` is raised on 84.6% of unanswerable queries but also on 76% of *answerable* ones, so it carries little information with heuristic priors (it needs p ≥ 0.6).
   - Passive mode abstains on 46% of unanswerable queries, but that comes from the gate. It also falsely abstains on 30.7% of answerable ones.
9. **Passive gate penalizes Chinese (bug-like).** In `engine.rs`, `gate_prior = if q.chars().count() < 12 { 0.3 } else { 0.6 }`, and 0.3 is below `T_GATE = 0.35`. Any prompt under 12 *characters* gets no passive recall. CJK packs a question into few characters: all 9 Chinese queries under 12 chars were gated, 5 of them answerable (e.g. 我平时用什么终端？). Chinese R@5 drops from 0.577 (agent recall) to 0.231 (passive recall).
10. **One cross-project write mis-scoped, and it leaked.** n171 ("tidewatch's docs site is …"), written from home, was stored as global. Mention detection only offers other projects when there is a current project, or the prior picks global when there is none. The note then leaked into 3 queries asked in other projects, which accounts for all of uni-mem's 1.4% leak rate.
11. **Streaming latency: about 43 ms added per request (Nagle/delayed-ACK).**
    - `unimem serve` (and `gateway::router` served with plain `axum::serve`) does not set `TCP_NODELAY` on client sockets. The SSE writer emits many small chunks, so on Linux loopback the gateway adds +43 ms TTFB p50 with passive recall on. With recall off it adds +1.0 ms p50 but still about +48 ms p95.
    - The same gateway state served with `TCP_NODELAY` adds only **+2.0 ms p50 / +3.0 ms p95** TTFB. This is a one-line fix in `main.rs` serve (`listener.tap_io(|s| { let _ = s.set_nodelay(true); })`); I did not change it.
    - JSON (non-streaming) requests are unaffected: about +0.5 ms.
12. **Expansion-overlay anchors are not session-scoped.** The anchor is `hash(prev message key, visible key)` and `put_overlay` uses `ON CONFLICT DO NOTHING`. When two sessions send the same prompt and get the same visible reply, the second session's next request gets the **first session's hidden tool calls and results** replayed (10/10 in the probe). This is unlikely with real sampled model text, but plausible for short deterministic replies or a retry/fork of a common opener. It also means one session's recall results can appear in another session's context.
13. **Continuation rounds cost upstream bytes.** Over the 30-turn conversation uni-mem sent 2.65 MB upstream, against 1.53 MB for passthrough (+72%). The cause is 6 extra full-context continuation requests plus the tool definitions. All of it is cacheable prefix (cacheable byte ratio 0.981 vs 0.987), but uncached input tokens and request count still go up.
14. **Project identification: 35/37 correct.**
    - A subdirectory of a non-git project with a manifest resolves to *no project*: the manifest check only looks at the cwd and never walks upward.
    - A symlinked path to a repo without a remote gets a different `local…` pid than the real path, because paths are not canonicalized, which creates a duplicate project. Symlinks to repos *with* a remote are fine.
    - Everything else passed: worktrees (CLI-made and hand-made relative gitdir + commondir), submodules, nested repos, forks, GHE/GitLab subgroups/ports, userinfo, case, trailing slashes, a dotfiles-repo home, and identity kept when a remote is added later.
15. **Throughput scales linearly.** Write p50 grows from 0.84 ms to 11.6 ms and recall p50 from 0.83 ms to 8.8 ms between 1k and 10k notes. The causes are brute-force cosine over all vectors including superseded ones, and per-id `get_memory` round-trips. That is fine for a personal store, but it would need an ANN or prefiltered index well before 100k notes. The DB takes about 5.5 KB per note (512-d f32 embedding, FTS and edges).

### Caveats

- The decider is the heuristic prior, so every "System One" judgement above is the code-computed prior. Jev or an LLM decider would change the write-screen, dup/supersede, scope and relevance numbers, possibly a lot. Rerun with `UNIMEM_BENCH_DECIDER=jev,heuristic` (and/or `UNIMEM_BENCH_EMBEDDER=openai` plus config) to measure that.
- The baselines store every write verbatim, *including* rejected secrets and injections; their "rejected-note returned" rate (37–49%) reflects that.
- The 10k-tier corpus is templated and much more self-similar than real notes; read point 5 as a stress test of the supersede rule, not as a typical outcome.
