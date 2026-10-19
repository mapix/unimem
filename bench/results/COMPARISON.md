# uni-mem v0.1 vs v0.2: evaluation comparison

- **v0.1** = `db7eb95` (the evaluated release). **v0.2** = `61e31fc` (fixes for the v0.1 findings: TCP_NODELAY, session-scoped expansion anchors, supersede subject check, global preferences, CJK passive gate, write screens, sufficiency, project walk-up/canonicalisation).
- Both versions were run with the same bench binary (`src/bin/unimem-bench.rs` at this commit). For v0.1 it was built in a `db7eb95` worktree with two adaptations: `axum::serve` instead of `gateway::serve`, and the v0.1 `unimem serve` reqwest builder instead of `gateway::upstream_client`.
- Setup: offline heuristic decider, `hash-512` embedder, release profile, same 4-core aarch64 VM, rustc 1.99.1.
- Sanity check: the updated bench, run on v0.1 with devmem, reproduces every E1 number in `v0.1/e1_memory.json` exactly. Only the latency fields differ. So the bench changes (`--data/--out`, `decoy`/`not_supersedes`, silos per harness present in the dataset) do not move any v0.1 number.
- **devmem is in-sample.** The v0.1 findings on devmem drove the v0.2 fixes, so devmem deltas are optimistic by construction.
- **heldout is out-of-sample.** `bench/data/heldout.json` (generator `gen_heldout.py`) was written independently of the implementation: a different developer, 6 different repos, 3 harnesses (claude-code, codex, cursor), 194 notes and 210 queries. It adds 12 `not_supersedes` distractor pairs and 6 `decoy` notes. **The heldout numbers are the honest ones.**

Results directories:

| dir | version | experiments |
|---|---|---|
| `v0.1/` | db7eb95 | E1 to E4 on devmem (original run). Also `e2_gateway_rerun.json` (E2 rerun in this session) and `e4_throughput_n7000.json` (E4 at N=7000 in this session) |
| `v0.2/` | 61e31fc | E1 to E4 on devmem (E4 at N=7000). Also `e2_gateway_rerun.json` (second E2 run) |
| `v0.1-heldout/` | db7eb95 | E1 on heldout |
| `v0.2-heldout/` | 61e31fc | E1 on heldout |

## TL;DR

**On held-out data, v0.2 does not improve retrieval quality.**

| metric (heldout) | v0.1 | v0.2 | delta |
|---|---|---|---|
| R@5 | 0.444 | 0.447 | +0.003 |
| R@1 | 0.303 | 0.281 | -0.022 |
| MRR | 0.366 | 0.353 | -0.013 |

uni-mem is still clearly below plain BM25 on R@5 (`bm25-top5-cwd` 0.601, `bm25-top5` 0.598) and below `vector-top5-cwd` (0.472). It is now also below BM25 on R@1 (0.281 vs 0.292–0.298) and on MRR (0.353 vs 0.416).

What v0.2 really fixed (these hold out-of-sample or are mechanical):

- **Streaming gateway overhead** (`TCP_NODELAY`). Added TTFB p50 went from +42.8 ms to +0.76 ms, and p95 from +48.5 ms to +1.0 ms.
- **Cross-session anchor collision.** The probe went from 10/10 sessions affected to 0/10.
- **Project identification.** E3 went from 35/37 to 37/37. The two fixed layouts are the two that failed before; no new layouts were added.
- **Over-eager supersede.**
  - On heldout distractors, false supersedes went from 1/12 to 0/12, and supersede precision from 0.625 to 1.0.
  - In E4, the share of stored notes that superseded something fell from 91% to 27%.
- **Passive recall.** The CJK gate fix holds on heldout.
  - Passive R@5 went from 0.323 to 0.393.
  - Chinese passive R@5 went from 0.036 to 0.286.
- **Isolation, precision, safety.**
  - Scope leak went from 2.4% to 0.5%.
  - The scope_isolation category improved: R@5 from 0.438 to 0.594.
  - Precision went from 0.205 to 0.241.
  - Write-screen recall went from 0.278 to 0.500, and its precision from 0.625 to 0.750.
  - Rejected-note-returned went from 11.9% to 8.1%.

What got worse, or did not generalize (honest list):

1. **Chinese agent recall regressed on both sets.**
   - heldout: 0.500 → 0.321. devmem: 0.577 → 0.308.
   - 7 of the 8 heldout queries that v0.2 lost (gold found in v0.1, nothing found in v0.2) now return *zero* notes. Most are short Chinese questions: 初稿什么时候交？, 正文要写多少字？, 怎么加翻译文案？, commit 信息有什么要求？
   - This is consistent with the new minimum anchor-evidence floor (`MIN_ANCHOR_VSIM` / `MIN_ANCHOR_FTS_COVERAGE`) cutting short CJK queries. I have not bisected it.
2. **False abstention on answerable queries.**
   - It rose from 5.6% to 21.3% on heldout (devmem: 2.2% to 21.2%). In exchange, true abstention on unanswerable queries went from 4.3% to 17.4%.
   - One in five answerable questions now gets nothing at all. On these numbers that is a bad trade.
3. **Supersede recall collapsed, so stale answers went *up*.**
   - The subject check removed the false supersedes, and almost all the true ones with them.
   - heldout: 5/26 correct went to 1/26. devmem: 5/27 went to **0/27**.
   - Stale returned went from 44.4% to 50.8% on heldout and from 46.7% to 62.2% on devmem. On heldout v0.2 is no better than `bm25-top5-cwd` (50.8%), while v0.1 beat every baseline here.
4. **Write-screen fixes overfit.**
   - devmem: rejection precision and recall both went from 0.909/0.667 to 1.000/1.000.
   - heldout: rejection recall is 0.500 and precision 0.750. 9 of 18 bad notes still get through.
   - Missed injections: "IMPORTANT SYSTEM NOTE: …", an HTML-comment `<!-- assistant: … -->`, 致所有 AI 助手…, "Maintainer instruction (approved)…", "Note for any AI assistant…". Injection recall is 1/6.
   - Missed credentials: a Steamworks login and a Chrome Web Store client secret.
   - Missed status notes: two.
   - **Decoy false-reject rate is 3/6 in both versions, unchanged.** The rejected decoys are AWS's published example keys in docs, the `wip/*` branch policy, and a hostile-page test fixture.
5. **Duplicate detection did not improve.** heldout recall is 2/22 in both versions. devmem is 0/23 in both, so the new `DUP_JACCARD` path catches none of the 23 in-sample paraphrases either.
6. **Global preferences stated inside a project: still almost all wrong.**
   - heldout: 0/13 → 1/13. devmem (in-sample): 0/13 → 4/13.
7. **New silent-data-loss mode: false duplicates on templated notes (E4).**
   - In E4, v0.2 marked 12,664 of 19,664 writes as duplicates, but the generator produces only about 1.0k exact-text repeats.
   - A direct probe confirms it: "To deploy media-7 run `make deploy-media-7 ENV=qa` …" is dropped as a duplicate of the auth-12/dev note. So is the Chinese config note for search-5/prod, dropped against auth-12/dev.
   - The skeleton-Jaccard dup rule has the same missing subject check that the old supersede rule had. Different subject plus no differing *value* number means the note is dropped.
8. **Write latency is up about 30%.** E1 write p50 went from 0.53 ms to 0.69 ms on heldout.
9. **In-sample retrieval got worse too.** devmem R@5 went from 0.486 to 0.425 and R@1 from 0.352 to 0.285. Retrieval dropped even on the set that informed the fixes. Paraphrase fell 0.533 → 0.367, chinese 0.577 → 0.308, and false abstention rose to 21.2%.
10. **Graph expansion still contributes nothing measurable.** full vs lite on heldout R@5 is 0.447 vs 0.441.
11. **Sufficiency signal is still uninformative.** `sufficient=false` fires on 69.7% of answerable queries (was 74.7%) and on 82.6% of unanswerable ones.

Unchanged and still true: unified memory beats per-harness silos across harnesses. On the 117 heldout cross-harness answerable queries, R@5 is 0.444 for unified vs 0.000 for silos.

## A. devmem (in-sample: the fixes were informed by this set)

211 queries, 183 notes, 2 harnesses. v0.1 numbers = `v0.1/e1_memory.json`; v0.2 = `v0.2/e1_memory.json`. devmem has no `decoy`/`not_supersedes` annotations, hence "–".

### A.1 Overall (v0.1 → v0.2; delta in parentheses)

| metric | unimem-full | unimem-lite | unimem-passive | silo-per-harness |
|---|---|---|---|---|
| R@1 | 0.352 → 0.285 (-0.067) | 0.352 → 0.285 (-0.067) | 0.274 → 0.246 (-0.028) | 0.229 → 0.179 (-0.050) |
| R@5 | 0.486 → 0.425 (-0.061) | 0.486 → 0.419 (-0.067) | 0.385 → 0.380 (-0.005) | 0.307 → 0.235 (-0.072) |
| MRR | 0.408 → 0.341 (-0.067) | 0.409 → 0.340 (-0.069) | 0.319 → 0.300 (-0.019) | 0.265 → 0.202 (-0.063) |
| notes/query | 2.204 → 1.858 (-0.346) | 2.175 → 1.801 (-0.374) | 1.697 → 1.687 (-0.010) | 1.682 → 1.209 (-0.473) |
| precision | 0.215 → 0.218 (+0.003) | 0.218 → 0.223 (+0.005) | 0.220 → 0.216 (-0.004) | 0.177 → 0.187 (+0.010) |
| scope leak (queries) | 1.4% → 0.0% (-1.4pp) | 1.4% → 0.0% (-1.4pp) | 0.5% → 0.0% (-0.5pp) | 2.8% → 0.0% (-2.8pp) |
| forbidden hit | 0.0% → 0.0% (=) | 0.0% → 0.0% (=) | 0.0% → 0.0% (=) | 0.0% → 0.0% (=) |
| stale returned | 46.7% → 62.2% (+15.5pp) | 46.7% → 62.2% (+15.5pp) | 37.8% → 53.3% (+15.5pp) | 31.1% → 35.6% (+4.5pp) |
| stale unflagged | 46.7% → 62.2% (+15.5pp) | 46.7% → 62.2% (+15.5pp) | 37.8% → 53.3% (+15.5pp) | 31.1% → 35.6% (+4.5pp) |
| stale above gold | 37.8% → 44.4% (+6.6pp) | 37.8% → 44.4% (+6.6pp) | 28.9% → 37.8% (+8.9pp) | 26.7% → 31.1% (+4.4pp) |
| abstain (unanswerable) | 7.7% → 42.3% (+34.6pp) | 7.7% → 42.3% (+34.6pp) | 46.2% → 46.2% (=) | 19.2% → 46.2% (+27.0pp) |
| abstain or sufficient=false (unans.) | 84.6% → 88.5% (+3.9pp) | 84.6% → 88.5% (+3.9pp) | 84.6% → 88.5% (+3.9pp) | 92.3% → 92.3% (=) |
| false abstain (answerable) | 2.2% → 21.2% (+19.0pp) | 2.2% → 21.2% (+19.0pp) | 30.7% → 33.0% (+2.3pp) | 8.4% → 36.3% (+27.9pp) |
| sufficiency misfire: sufficient=false on answerable | 76.0% → 70.9% (-5.1pp) | 76.0% → 70.9% (-5.1pp) | 76.0% → 70.9% (-5.1pp) | 88.3% → 83.2% (-5.1pp) |
| rejected-note returned | 7.1% → 0.0% (-7.1pp) | 7.1% → 0.0% (-7.1pp) | 4.7% → 0.0% (-4.7pp) | 6.6% → 0.0% (-6.6pp) |

#### R@5 by category (MRR in parentheses), v0.1 → v0.2

| category | n | unimem-full | unimem-passive | silo-per-harness |
|---|---|---|---|---|
| single_hop | 35 | 0.486 (0.450) → 0.514 (0.464) +0.028 | 0.457 (0.421) → 0.457 (0.407) = | 0.229 (0.229) → 0.229 (0.229) = |
| paraphrase | 30 | 0.533 (0.475) → 0.367 (0.325) -0.166 | 0.400 (0.342) → 0.333 (0.292) -0.067 | 0.367 (0.317) → 0.167 (0.167) -0.200 |
| chinese | 26 | 0.577 (0.529) → 0.308 (0.260) -0.269 | 0.231 (0.202) → 0.231 (0.183) = | 0.308 (0.308) → 0.154 (0.154) -0.154 |
| cross_project | 21 | 0.476 (0.365) → 0.476 (0.357) = | 0.476 (0.365) → 0.476 (0.357) = | 0.286 (0.222) → 0.286 (0.222) = |
| temporal | 25 | 0.560 (0.420) → 0.480 (0.378) -0.080 | 0.480 (0.380) → 0.440 (0.338) -0.040 | 0.360 (0.310) → 0.280 (0.228) -0.080 |
| causal | 21 | 0.381 (0.254) → 0.476 (0.290) +0.095 | 0.333 (0.230) → 0.429 (0.266) +0.096 | 0.381 (0.262) → 0.333 (0.226) -0.048 |
| scope_isolation | 27 | 0.333 (0.278) → 0.333 (0.254) = | 0.286 (0.230) → 0.286 (0.206) = | 0.238 (0.190) → 0.238 (0.190) = |
| unanswerable | 26 | abstain 7.7% → 42.3% | abstain 46.2% → 46.2% | abstain 19.2% → 46.2% |

#### Cross-harness (answerable)

cross-harness answerable: 80, same-harness answerable: 99

| system | cross R@5 | cross MRR | same R@5 | same MRR |
|---|---|---|---|---|
| unimem-full | 0.400 → 0.388 | 0.348 → 0.323 | 0.556 → 0.455 | 0.457 → 0.356 |
| unimem-lite | 0.400 → 0.375 | 0.350 → 0.319 | 0.556 → 0.455 | 0.457 → 0.356 |
| unimem-passive | 0.338 → 0.338 | 0.292 → 0.273 | 0.424 → 0.414 | 0.341 → 0.321 |
| silo-per-harness | 0.000 → 0.000 | 0.000 → 0.000 | 0.556 → 0.424 | 0.479 → 0.365 |

#### Scope isolation queries

| system | R@5 | forbidden hit | leak |
|---|---|---|---|
| unimem-full | 0.333 → 0.333 | 0.0% → 0.0% | 0.0% → 0.0% |
| unimem-lite | 0.333 → 0.333 | 0.0% → 0.0% | 0.0% → 0.0% |
| unimem-passive | 0.286 → 0.286 | 0.0% → 0.0% | 0.0% → 0.0% |
| silo-per-harness | 0.238 → 0.238 | 0.0% → 0.0% | 3.7% → 0.0% |

#### Write path

| metric | v0.1 | v0.2 |
|---|---|---|
| rejection precision | 0.909 | 1.000 |
| rejection recall | 0.667 | 1.000 |
| decoy false-reject rate | – | – |
| duplicate precision | – | – |
| duplicate recall | 0.000 | 0.000 |
| dup recall (correct original) | 0.000 | 0.000 |
| supersede precision | 0.714 | – |
| supersede recall | 0.185 | 0.000 |
| distractor false-supersede (pair) | – | – |
| distractor target wrongly superseded (by anything) | – | – |
| scope accuracy | 0.919 | 0.946 |
| kind accuracy | 0.535 | 0.542 |
| rejections per type | {"reject_injection": "2/5", "reject_secret": "3/5", "reject_status": "5/5"} | {"reject_injection": "5/5", "reject_secret": "5/5", "reject_status": "5/5"} |
| false rejects (all non-reject notes) | 1 | 0 |
| decoys rejected per type | null | {} |
| dup: gold / detected / false pos. | 23/0/0 | 23/0/0 |
| supersede: gold / predicted / correct | 27/7/5 | 27/0/0 |
| distractor pairs / false supersedes | 0/0 | 0/0 |
| graph edges | {"causal": 2, "entity": 20, "semantic": 43, "supersedes": 7, "temporal": 123} | {"causal": 6, "entity": 23, "semantic": 51, "temporal": 114} |
| write latency p50 / p95 ms | 0.522 / 0.902 | 0.639 / 1.147 |

Scope assignment by group:

| group | v0.1 | v0.2 |
|---|---|---|
| global note written at home | 13/13 (1.000) | 12/12 (1.000) |
| global note written inside a project | 0/13 (0.000) | 4/13 (0.308) |
| note about another project (cross-project write) | 9/10 (0.900) | 10/10 (1.000) |
| project note written in that project | 136/136 (1.000) | 133/133 (1.000) |

#### Baselines (devmem; identical in both versions, they don't use the engine's write path)

| system | R@1 | R@5 | MRR | notes/q | precision | leak | forbidden hit | stale returned | rejected-note returned | causal R@5 | cross_project R@5 | cross-harness R@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unimem-full (v0.2) | 0.285 | 0.425 | 0.341 | 1.858 | 0.218 | 0.0% | 0.0% | 62.2% | 0.0% | 0.476 | 0.476 | 0.388 |
| vector-top5 | 0.212 | 0.430 | 0.286 | 5.000 | 0.086 | 89.1% | 48.1% | 60.0% | 41.2% | 0.333 | 0.571 | 0.413 |
| vector-top5-cwd | 0.218 | 0.511 | 0.321 | 5.000 | 0.103 | 17.1% | 0.0% | 68.9% | 48.3% | 0.810 | 0.000 | 0.488 |
| bm25-top5 | 0.279 | 0.570 | 0.391 | 4.654 | 0.120 | 82.5% | 70.4% | 82.2% | 37.0% | 0.714 | 0.524 | 0.525 |
| bm25-top5-cwd | 0.296 | 0.578 | 0.407 | 4.147 | 0.138 | 14.2% | 0.0% | 84.4% | 48.8% | 0.810 | 0.000 | 0.550 |

## B. heldout (out-of-sample: the honest numbers)

210 queries (178 answerable, 23 unanswerable, 63 with a stale note, 24 with forbidden notes), 194 notes, 3 harnesses, 12 distractor pairs, 6 decoys. v0.1 = `v0.1-heldout/e1_memory.json`, v0.2 = `v0.2-heldout/e1_memory.json`. The baseline numbers are identical between the two runs (checked).

### B.1 Overall (v0.1 → v0.2; delta in parentheses)

| metric | unimem-full | unimem-lite | unimem-passive | silo-per-harness |
|---|---|---|---|---|
| R@1 | 0.303 → 0.281 (-0.022) | 0.303 → 0.281 (-0.022) | 0.219 → 0.258 (+0.039) | 0.146 → 0.118 (-0.028) |
| R@5 | 0.444 → 0.447 (+0.003) | 0.444 → 0.441 (-0.003) | 0.323 → 0.393 (+0.070) | 0.183 → 0.154 (-0.029) |
| MRR | 0.366 → 0.353 (-0.013) | 0.366 → 0.352 (-0.014) | 0.262 → 0.313 (+0.051) | 0.165 → 0.136 (-0.029) |
| notes/query | 2.238 → 1.857 (-0.381) | 2.190 → 1.771 (-0.419) | 1.690 → 1.614 (-0.076) | 1.424 → 0.871 (-0.553) |
| precision | 0.205 → 0.241 (+0.036) | 0.210 → 0.252 (+0.042) | 0.196 → 0.240 (+0.044) | 0.129 → 0.178 (+0.049) |
| scope leak (queries) | 2.4% → 0.5% (-1.9pp) | 1.9% → 0.0% (-1.9pp) | 2.4% → 0.5% (-1.9pp) | 3.8% → 0.0% (-3.8pp) |
| forbidden hit | 0.0% → 0.0% (=) | 0.0% → 0.0% (=) | 0.0% → 0.0% (=) | 0.0% → 0.0% (=) |
| stale returned | 44.4% → 50.8% (+6.4pp) | 44.4% → 50.8% (+6.4pp) | 34.9% → 42.9% (+8.0pp) | 20.6% → 20.6% (=) |
| stale unflagged | 44.4% → 50.8% (+6.4pp) | 44.4% → 50.8% (+6.4pp) | 34.9% → 42.9% (+8.0pp) | 20.6% → 20.6% (=) |
| stale above gold | 36.5% → 42.9% (+6.4pp) | 36.5% → 42.9% (+6.4pp) | 28.6% → 34.9% (+6.3pp) | 19.0% → 19.0% (=) |
| abstain (unanswerable) | 4.3% → 17.4% (+13.1pp) | 4.3% → 17.4% (+13.1pp) | 30.4% → 34.8% (+4.4pp) | 13.0% → 34.8% (+21.8pp) |
| abstain or sufficient=false (unans.) | 78.3% → 82.6% (+4.3pp) | 78.3% → 82.6% (+4.3pp) | 78.3% → 82.6% (+4.3pp) | 91.3% → 91.3% (=) |
| false abstain (answerable) | 5.6% → 21.3% (+15.7pp) | 5.6% → 21.3% (+15.7pp) | 36.5% → 34.3% (-2.2pp) | 19.1% → 48.3% (+29.2pp) |
| sufficiency misfire: sufficient=false on answerable | 74.7% → 69.7% (-5.0pp) | 74.7% → 69.7% (-5.0pp) | 77.5% → 69.7% (-7.8pp) | 88.8% → 86.5% (-2.3pp) |
| rejected-note returned | 11.9% → 8.1% (-3.8pp) | 11.4% → 7.1% (-4.3pp) | 9.0% → 7.1% (-1.9pp) | 9.0% → 4.3% (-4.7pp) |

#### R@5 by category (MRR in parentheses), v0.1 → v0.2

| category | n | unimem-full | unimem-passive | silo-per-harness |
|---|---|---|---|---|
| single_hop | 36 | 0.556 (0.470) → 0.583 (0.498) +0.027 | 0.500 (0.428) → 0.528 (0.456) +0.028 | 0.167 (0.153) → 0.167 (0.153) = |
| paraphrase | 31 | 0.323 (0.306) → 0.290 (0.290) -0.033 | 0.258 (0.258) → 0.258 (0.258) = | 0.129 (0.113) → 0.097 (0.081) -0.032 |
| chinese | 28 | 0.500 (0.446) → 0.321 (0.286) -0.179 | 0.036 (0.036) → 0.286 (0.268) +0.250 | 0.286 (0.286) → 0.179 (0.179) -0.107 |
| cross_project | 22 | 0.227 (0.157) → 0.227 (0.157) = | 0.227 (0.157) → 0.227 (0.157) = | 0.045 (0.045) → 0.045 (0.045) = |
| temporal | 25 | 0.440 (0.287) → 0.560 (0.315) +0.120 | 0.440 (0.287) → 0.520 (0.295) +0.080 | 0.280 (0.213) → 0.280 (0.210) = |
| causal | 20 | 0.600 (0.467) → 0.600 (0.427) = | 0.500 (0.392) → 0.550 (0.402) +0.050 | 0.250 (0.225) → 0.200 (0.175) -0.050 |
| scope_isolation | 25 | 0.438 (0.396) → 0.594 (0.502) +0.156 | 0.281 (0.240) → 0.375 (0.315) +0.094 | 0.094 (0.094) → 0.094 (0.094) = |
| unanswerable | 23 | abstain 4.3% → 17.4% | abstain 30.4% → 34.8% | abstain 13.0% → 34.8% |

#### Cross-harness (answerable)

cross-harness answerable: 117, same-harness answerable: 61

| system | cross R@5 | cross MRR | same R@5 | same MRR |
|---|---|---|---|---|
| unimem-full | 0.436 → 0.444 | 0.356 → 0.356 | 0.459 → 0.451 | 0.385 → 0.348 |
| unimem-lite | 0.436 → 0.436 | 0.356 → 0.354 | 0.459 → 0.451 | 0.385 → 0.348 |
| unimem-passive | 0.355 → 0.406 | 0.292 → 0.326 | 0.262 → 0.369 | 0.205 → 0.290 |
| silo-per-harness | 0.000 → 0.000 | 0.000 → 0.000 | 0.533 → 0.451 | 0.481 → 0.398 |

#### Scope isolation queries

| system | R@5 | forbidden hit | leak |
|---|---|---|---|
| unimem-full | 0.438 → 0.594 | 0.0% → 0.0% | 4.0% → 0.0% |
| unimem-lite | 0.438 → 0.594 | 0.0% → 0.0% | 4.0% → 0.0% |
| unimem-passive | 0.281 → 0.375 | 0.0% → 0.0% | 4.0% → 0.0% |
| silo-per-harness | 0.094 → 0.094 | 0.0% → 0.0% | 8.0% → 0.0% |

#### Write path

| metric | v0.1 | v0.2 |
|---|---|---|
| rejection precision | 0.625 | 0.750 |
| rejection recall | 0.278 | 0.500 |
| decoy false-reject rate | 0.500 | 0.500 |
| duplicate precision | 1.000 | 1.000 |
| duplicate recall | 0.091 | 0.091 |
| dup recall (correct original) | 0.091 | 0.091 |
| supersede precision | 0.625 | 1.000 |
| supersede recall | 0.192 | 0.038 |
| distractor false-supersede (pair) | 0.083 | 0.000 |
| distractor target wrongly superseded (by anything) | 0.083 | 0.000 |
| scope accuracy | 0.918 | 0.928 |
| kind accuracy | 0.533 | 0.533 |
| rejections per type | {"reject_injection": "0/6", "reject_secret": "3/7", "reject_status": "2/5"} | {"reject_injection": "1/6", "reject_secret": "5/7", "reject_status": "3/5"} |
| false rejects (all non-reject notes) | 3 | 3 |
| decoys rejected per type | {"injection": "1/2", "secret": "1/3", "status": "1/1"} | {"injection": "1/2", "secret": "1/3", "status": "1/1"} |
| dup: gold / detected / false pos. | 22/2/0 | 22/2/0 |
| supersede: gold / predicted / correct | 26/8/5 | 26/1/1 |
| distractor pairs / false supersedes | 12/1 | 12/0 |
| graph edges | {"causal": 8, "entity": 32, "semantic": 31, "supersedes": 8, "temporal": 115} | {"causal": 8, "entity": 34, "semantic": 36, "supersedes": 1, "temporal": 109} |
| write latency p50 / p95 ms | 0.53 / 0.903 | 0.687 / 1.145 |

Scope assignment by group:

| group | v0.1 | v0.2 |
|---|---|---|
| global note written at home | 18/18 (1.000) | 18/18 (1.000) |
| global note written inside a project | 0/13 (0.000) | 1/13 (0.077) |
| note about another project (cross-project write) | 2/4 (0.500) | 3/4 (0.750) |
| project note written in that project | 149/149 (1.000) | 145/145 (1.000) |

#### Baselines (heldout; identical in both versions, they don't use the engine's write path)

| system | R@1 | R@5 | MRR | notes/q | precision | leak | forbidden hit | stale returned | rejected-note returned | causal R@5 | cross_project R@5 | cross-harness R@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unimem-full (v0.2) | 0.281 | 0.447 | 0.353 | 1.857 | 0.241 | 0.5% | 0.0% | 50.8% | 8.1% | 0.600 | 0.227 | 0.444 |
| vector-top5 | 0.242 | 0.424 | 0.312 | 5.000 | 0.085 | 96.2% | 37.5% | 57.1% | 30.0% | 0.500 | 0.500 | 0.457 |
| vector-top5-cwd | 0.258 | 0.472 | 0.339 | 5.000 | 0.096 | 10.5% | 0.0% | 54.0% | 39.0% | 0.550 | 0.091 | 0.449 |
| bm25-top5 | 0.298 | 0.598 | 0.416 | 4.576 | 0.131 | 77.1% | 45.8% | 61.9% | 43.8% | 0.650 | 0.545 | 0.594 |
| bm25-top5-cwd | 0.292 | 0.601 | 0.416 | 4.029 | 0.150 | 4.8% | 0.0% | 50.8% | 30.5% | 0.700 | 0.091 | 0.598 |

Wrong supersedes in v0.1 on heldout: n064→n059 (distractor: dbt `ci` target threads vs another target), plus n187→n185 and n144→n140 (paraphrase duplicates superseding their original). v0.2 has none, but it predicts only one supersede in total.

## C. E2: gateway (devmem notes; mock upstream with TCP_NODELAY; 300 requests per cell)

The v0.1 column is the original run (`v0.1/e2_gateway.json`); `v0.1/e2_gateway_rerun.json` from this session reproduces it within about 5%. v0.2 is `v0.2/e2_gateway.json`; a second run (`v0.2/e2_gateway_rerun.json`) agrees within about 3%.

**Caveat on absolute numbers.** The *direct* path (bench client → mock, no uni-mem code) measures about 2× faster in the v0.2 binary (direct/json TTFB p50 0.136 ms vs 0.245–0.257 ms). This reproduced across two runs of each binary in the same session, and I have not explained it. Compare the **added** latency (difference to direct, paired per request), not absolute values.

Added latency vs direct (ms):

| via | v0.1 +TTFB p50 / p95 | v0.2 +TTFB p50 / p95 | v0.1 paired ΔTTFB p50 / p95 | v0.2 paired ΔTTFB p50 / p95 |
|---|---|---|---|---|
| gateway/stream (as `unimem serve`) | 42.776 / 48.470 | **0.757 / 1.022** | 42.807 / 48.580 | 0.754 / 1.046 |
| gateway/json | 0.489 / 0.582 | 0.218 / 0.269 | 0.477 / 0.663 | 0.217 / 0.279 |
| gateway-norecall/stream | 1.017 / 47.687 | 0.224 / 0.275 | 1.019 / 47.721 | 0.220 / 0.311 |
| gateway-norecall/json | 0.503 / 0.579 | 0.226 / 0.280 | 0.497 / 0.657 | 0.229 / 0.286 |
| gateway-nodelay/stream (diagnostic: bench forces NODELAY) | 2.047 / 3.404 | 0.760 / 1.004 | 2.038 / 3.418 | 0.765 / 1.040 |
| gateway-nodelay/json | 0.486 / 0.637 | 0.224 / 0.274 | 0.474 / 0.705 | 0.228 / 0.288 |

The nodelay effect:

- In v0.1, `unimem serve` streaming cost +42.8 ms p50. The same state with NODELAY forced by the bench cost +2.0 ms.
- In v0.2, `gateway/stream` and `gateway-nodelay/stream` are identical (+0.76 ms). That confirms the server now sets NODELAY itself.
- The v0.1 `norecall` p95 tail of +47.7 ms is gone (+0.28 ms).

Absolute latency (ms):

| path | v0.1 TTFB p50 / p95 | v0.2 TTFB p50 / p95 | v0.1 total p50 / p95 | v0.2 total p50 / p95 |
|---|---|---|---|---|
| direct/json | 0.257 / 0.335 | 0.136 / 0.155 | 0.257 / 0.336 | 0.137 / 0.155 |
| direct/stream | 0.328 / 0.417 | 0.162 / 0.197 | 1.295 / 1.503 | 0.676 / 0.739 |
| gateway/json | 0.746 / 0.917 | 0.354 / 0.423 | 0.747 / 0.918 | 0.355 / 0.424 |
| gateway/stream | 43.104 / 48.887 | 0.919 / 1.219 | 44.219 / 49.979 | 1.411 / 1.727 |
| gateway-norecall/json | 0.760 / 0.913 | 0.362 / 0.435 | 0.761 / 0.914 | 0.363 / 0.436 |
| gateway-norecall/stream | 1.344 / 48.104 | 0.386 / 0.472 | 2.441 / 49.340 | 0.874 / 0.997 |
| gateway-nodelay/json | 0.743 / 0.971 | 0.360 / 0.429 | 0.745 / 0.972 | 0.360 / 0.429 |
| gateway-nodelay/stream | 2.375 / 3.821 | 0.922 / 1.201 | 3.345 / 5.086 | 1.405 / 1.731 |

### Prompt-cache stability (30-turn streamed conversation)

| mode | upstream req / continuations (v0.1 → v0.2) | mean prefix fraction | min | fully stable transitions | cacheable byte ratio | upstream bytes |
|---|---|---|---|---|---|---|
| Passthrough | 37/0 → 37/0 | 1.000 → 1.000 | 1.000 → 1.000 | 36/36 → 36/36 | 0.987 → 0.987 | 1,533,499 → 1,533,499 |
| Unimem | 43/6 → 43/6 | 1.000 → 1.000 | 1.000 → 1.000 | 42/42 → 42/42 | 0.981 → 0.981 | 2,645,782 → 2,624,646 |
| NaiveSystem | 37/0 → 37/0 | 0.823 → 0.826 | 0.614 → 0.620 | 8/36 → 10/36 | 0.804 → 0.806 | 1,558,104 → 1,555,839 |
| NaiveTail | 37/0 → 37/0 | 0.988 → 0.988 | 0.966 → 0.969 | 12/36 → 13/36 | 0.976 → 0.976 | 1,551,893 → 1,550,867 |

- Cache stability is unchanged: every uni-mem transition is still a byte-exact prefix extension.
- The upstream byte overhead is unchanged: +71% vs passthrough, from the 6 extra continuation rounds.

### Interception correctness and anchor-collision probe

| | v0.1 | v0.2 |
|---|---|---|
| assertions passed | 1060 / 1060 | 1060 / 1060 |
| client → upstream requests (continuations) | 120 → 144 (24) | 120 → 144 (24) |
| anchor-collision probe: second session got the first session's hidden rounds | **10 / 10** | **0 / 10** |

## D. E3: project identification (37 synthetic layouts; in-sample)

| | v0.1 | v0.2 |
|---|---|---|
| correct | 35 / 37 | **37 / 37** |
| subdir of non-git manifest dir (`plain/nodey/src/lib`) | ✗ (no project) | ✓ `local$TMP/plain/nodey` |
| symlink to a repo without remote | ✗ (`local$TMP/links/scratch-link`, a duplicate project) | ✓ `local$TMP/home/code/scratch` |

Every other case is unchanged (all ✓). Both fixed cases are the ones the v0.1 report named. No held-out layouts exist, so this is in-sample.

## E. E4: engine throughput (templated synthetic notes)

- A full 10k-note run does not fit inside the ~178 s per-command limit of this sandbox: v0.2 needs about 20k write attempts to reach 10k stored notes, because of the duplicate verdicts below.
- So both versions were run in this session at `UNIMEM_BENCH_E4_N=7000` (tiers 1k and 7k). The original v0.1 10k run is shown for reference.
- Machine variance: the v0.1 1k tier measured 0.84 ms write p50 originally and 1.10 ms today. Treat about ±30% as noise.

| version | stored notes | live (not superseded) | write attempts | write p50 / p95 ms | recall p50 / p95 ms | passive p50 / p95 ms | notes/recall | DB MB | bytes/note | RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| v0.1 (today) | 1000 | 179 | 1024 | 1.100 / 1.762 | 1.214 / 1.808 | 1.171 / 1.764 | 3.843 | 5.62 | 5623 | 15.8 |
| v0.2 | 1000 | 880 | 1725 | 1.346 / 1.914 | 1.550 / 2.306 | 1.507 / 2.236 | 4.073 | 6.71 | 6705 | 16.8 |
| v0.1 (today) | 7000 | 436 | 7414 | 8.345 / 11.833 | 4.224 / 8.343 | 4.069 / 8.177 | 4.227 | 38.34 | 5477 | 28.6 |
| v0.2 | 7000 | 5064 | 19664 | 7.434 / 10.359 | 7.092 / 8.869 | 7.343 / 8.735 | 4.237 | 49.06 | 7008 | 29.8 |
| v0.1 (original) | 10000 | 512 | 10672 | 11.614 / 15.944 | 8.845 / 11.743 | 8.683 / 11.900 | 4.243 | 54.56 | 5456 | 65.0 |

Write outcomes:

| version (N) | stored (no supersede) | stored, superseding | duplicate | share of stores that superseded |
|---|---|---|---|---|
| v0.1 (7000) | 608 | 6392 | 414 | 91% |
| v0.2 (7000) | 5078 | 1922 | 12664 | 27% |
| v0.1 (10000, original) | 768 | 9232 | 672 | 92% |

Over-eager supersede is largely fixed: 5064 of 7000 notes stay live, against 436 before. That is also why v0.2 recall at 7k is slower (7.1 vs 4.2 ms p50). It now scores about 12× more live notes, so the two rows are not like for like. Recall cost still grows linearly (brute-force cosine).

**New problem: duplicate verdicts.**

- 64% of v0.2's write attempts were rejected as duplicates. An offline re-simulation of the generator's RNG (approximate: it ignores the RNG draws consumed by the recall queries between tiers) gives only about 1.0k exact-text repeats in 19,664 attempts. So roughly 11k distinct facts were silently dropped.
- Direct probe of HEAD with the heuristic decider:
  - "To deploy media-7 run `make deploy-media-7 ENV=qa` …" → duplicate of the auth-12/dev note.
  - "search-5 服务的配置在 deploy/prod/search-5.yaml…" → duplicate of the auth-12/dev note.
  - Port notes for different services (same port) were stored correctly, without supersede.
- Templates exaggerate this, as they did for v0.1's supersede finding. But it is the same failure class: no subject check, here on the skeleton-Jaccard duplicate path.

## Bench changes in this commit

- `unimem-bench <exp> [--data <dataset.json>] [--out <dir>]`.
  - E1 uses `--data` (default `bench/data/devmem.json`).
  - E2 always uses devmem, because its scripted conversation is about ledger-api.
- Optional note fields `not_supersedes` and `decoy`. They are reported as:
  - `distractor_false_supersede_rate`: the distractor superseded its look-alike.
  - `distractor_target_wrongly_superseded_rate`: the look-alike was superseded by anything that is not a gold update.
  - `decoy_false_reject_rate`, with per-type counts.
- Silos are built per harness present in the dataset, so heldout's `cursor` gets its own silo. On devmem the silos are identical to before.
- The gateway under test is served with `gateway::serve` and `gateway::upstream_client`, exactly as `unimem serve` does in v0.2.
- Everything else is unchanged.
