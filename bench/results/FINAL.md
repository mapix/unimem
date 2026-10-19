# uni-mem: final evaluation (v0.1 / v0.2 / v0.3)

Versions:

- **v0.1** = `db7eb95`
- **v0.2** = `61e31fc`
- **v0.3** = `8a679a8` (HEAD)

Setup:

- Offline heuristic decider, release build, 4-core aarch64 VM, rustc 1.99.1.
- All three versions were run with the same bench (`src/bin/unimem-bench.rs`, unchanged since `09bbc53`).
- The v0.1 build uses the two minimal adaptations described in `COMPARISON.md`: `axum::serve` and the v0.1 reqwest builder.
- Sanity check: the copied bench on v0.1 and v0.2 reproduced the committed `v0.1-heldout`/`v0.2-heldout` E1 numbers exactly. The one difference is a 3rd-decimal change in silo `rejected-note returned` for v0.2: 0.048 vs 0.043.

## What is in-sample and what is not

| dataset | status for v0.3 | why |
|---|---|---|
| `devmem.json` | **in-sample** | the v0.2 and v0.3 changes were driven by its findings |
| `heldout.json` | **in-sample** | out-of-sample for v0.2, but the v0.3 changes were tuned against it (see `COMPARISON.md`) |
| `heldout2.json` (+ `gen_heldout2.py`) | **out-of-sample** | new developer ("Mira"), 6 new repos, a new harness (opencode). Nobody tuned on it. 242 queries (218 answerable, 20 unanswerable, 4 forbidden-only isolation queries), 233 notes, 12 distractor pairs, 6 decoys. 140 of the answerable queries are cross-harness |

Contamination check for heldout2:

- The file timestamp (03:36) predates the v0.3 commit (05:15).
- None of the tuning scripts or result dirs from the v0.3 work reference it.
- Of the 64 bilingual hint entries v0.3 added to `src/lexicon.rs`, two have terms that occur only in heldout2 and not in devmem/heldout: `queue` (queues/队列) and `retry`. Both are generic dev words, but the possibility that they leaked cannot be excluded.

**Only the heldout2 numbers below should be read as generalisation.**

### Embedder caveat

v0.3 also replaced the offline hash embedder:

- `hash-512` (v0.1/v0.2) became `hash2-512` (v0.3), which adds stems, compound-identifier parts and bilingual concept features.
- The `vector-top5*` baselines use the engine's embedder, so they also changed between versions. `bm25-*` did not change.
- So "uni-mem vs vector" compares the v0.3 engine with a vector baseline that already benefits from v0.3's tokenizer work. Both variants are shown.

---

## 1. Primary: heldout2 (out-of-sample)

Result dirs: `v0.1-heldout2/`, `v0.2-heldout2/`, `v0.3-heldout2/`.

### 1.1 Overall

| metric | uni-mem v0.1 | uni-mem v0.2 | **uni-mem v0.3** | silo-per-harness (v0.3) | vector-top5 (hash2) | vector-top5-cwd (hash2) | vector-top5-cwd (old hash) | bm25-top5 | bm25-top5-cwd |
|---|---|---|---|---|---|---|---|---|---|
| R@1 | 0.440 | 0.408 | **0.555** | 0.243 | 0.358 | 0.390 | 0.349 | 0.408 | 0.413 |
| R@5 | 0.569 | 0.550 | **0.688** | 0.271 | 0.656 | 0.670 | 0.615 | 0.642 | 0.615 |
| MRR | 0.497 | 0.471 | **0.616** | 0.257 | 0.475 | 0.506 | 0.453 | 0.500 | 0.497 |
| notes/query | 2.161 | 1.888 | 1.636 | 1.136 | 5.000 | 5.000 | 5.000 | 4.541 | 4.045 |
| precision | 0.263 | 0.291 | **0.423** | 0.242 | 0.131 | 0.134 | 0.123 | 0.141 | 0.152 |
| scope leak (queries) | 3.3% | 2.5% | 2.5% | 2.9% | 91.3% | 8.7% | 10.3% | 79.3% | 3.3% |
| forbidden hit | 0.0% | 0.0% | 0.0% | 0.0% | 66.7% | 0.0% | 0.0% | 57.1% | 0.0% |
| stale returned | 34.9% | 51.2% | **16.3%** | 16.3% | 62.8% | 58.1% | 53.5% | 65.1% | 55.8% |
| stale above gold | 27.9% | 37.2% | 16.3% | 16.3% | 51.2% | 46.5% | 46.5% | 46.5% | 39.5% |
| abstain (unanswerable) | 10.0% | 20.0% | **5.0%** | 15.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| abstain or sufficient=false (unans.) | 75.0% | 75.0% | 70.0% | 95.0% | – | – | – | – | – |
| false abstain (answerable) | 2.3% | 14.2% | 2.8% | 23.4% | 0.0% | 0.0% | 0.0% | 1.8% | 2.8% |
| sufficient=false on answerable | 72.5% | 61.5% | 45.0% | 70.2% | – | – | – | – | – |
| rejected-note returned | 13.6% | 10.3% | 6.6% | 4.1% | 26.9% | 45.5% | 43.8% | 31.8% | 37.2% |

`unimem-lite` and `unimem-passive` return exactly the same results as `unimem-full` in v0.3 on all three datasets (lite differs by one query on heldout). Graph expansion and the passive gate make no measurable difference.

### 1.2 Significance (paired, 218 answerable queries, R@5 hit per query)

| comparison | ΔR@5 | 95% bootstrap CI | queries won / lost | sign-test p |
|---|---|---|---|---|
| v0.3 − v0.1 | +0.119 | [+0.064, +0.174] | 34 / 8 | 7e-5 |
| v0.3 − v0.2 | +0.138 | [+0.078, +0.193] | 38 / 8 | 9e-6 |
| v0.3 − bm25-top5-cwd | +0.073 | [+0.018, +0.128] | 26 / 10 | 0.011 |
| v0.3 − bm25-top5 | +0.046 | [−0.005, +0.096] | 21 / 11 | 0.11 |
| v0.3 − vector-top5-cwd (old hash) | +0.073 | [+0.009, +0.133] | 32 / 16 | 0.029 |
| v0.3 − vector-top5-cwd (hash2) | +0.018 | [−0.037, +0.073] | 20 / 16 | 0.62 |
| v0.3 − vector-top5 (hash2) | +0.032 | [−0.023, +0.087] | 23 / 16 | 0.34 |

**v0.3 is significantly better than v0.1, v0.2 and bm25-top5-cwd on R@5. It is not significantly better than a plain top-5 cosine search with the same hash2 embedder.**

Its clear advantages over that baseline are elsewhere:

| | v0.3 | vector-top5-cwd (hash2) |
|---|---|---|
| R@1 | 0.555 | 0.390 |
| MRR | 0.616 | 0.506 |
| precision | 0.423 | 0.134 |
| notes per answer | 1.6 | 5 |
| stale returned | 16% | 58% |
| rejected-note returned | 6.6% | 45.5% |

### 1.3 R@5 by category (heldout2)

| category | n | v0.1 | v0.2 | **v0.3** | silo (v0.3) | vector-top5 | vector-top5-cwd | bm25-top5 | bm25-top5-cwd |
|---|---|---|---|---|---|---|---|---|---|
| single_hop | 65 | 0.692 | 0.723 | **0.892** | 0.338 | 0.908 | 0.846 | 0.785 | 0.738 |
| paraphrase | 31 | 0.290 | 0.323 | 0.419 | 0.258 | 0.387 | 0.484 | 0.355 | 0.355 |
| chinese | 38 | 0.658 | 0.526 | 0.763 | 0.368 | 0.605 | 0.816 | 0.763 | 0.763 |
| cross_project | 19 | 0.211 | 0.158 | 0.368 | 0.158 | 0.579 | 0.053 | 0.474 | 0.053 |
| temporal | 24 | 0.500 | 0.500 | 0.667 | 0.208 | 0.625 | 0.667 | 0.625 | 0.708 |
| causal | 23 | 0.696 | 0.696 | 0.739 | 0.261 | 0.609 | 0.696 | 0.739 | 0.739 |
| scope_isolation | 22 | **0.722** | 0.667 | 0.556 | 0.056 | 0.500 | 0.667 | 0.444 | 0.611 |
| unanswerable (abstain) | 20 | 10.0% | 20.0% | 5.0% | 15.0% | 0% | 0% | 0% | 0% |

(Vector columns use hash2.)

- **scope_isolation regressed** from v0.1 (0.722) to v0.3 (0.556). The cause is generic short questions ("what runs on every push?") whose top slots go to global git rules and the unrejected injection note n169 (see `v0.3/ERRORS.md`).
- **paraphrase (0.419) and cross_project (0.368) are still weak.** For cross_project the uncapped baselines that search every project do better (vector-top5 0.579, bm25-top5 0.474), but they leak scope on 79–91% of queries.

### 1.4 Cross-harness: unified store vs per-harness silos (heldout2)

140 cross-harness and 78 same-harness answerable queries.

| system | cross-harness R@5 | cross MRR | same-harness R@5 | same MRR |
|---|---|---|---|---|
| uni-mem v0.1 | 0.557 | 0.486 | 0.590 | 0.518 |
| silo v0.1 | 0.000 | 0.000 | 0.564 | 0.517 |
| uni-mem v0.2 | 0.543 | 0.466 | 0.564 | 0.480 |
| silo v0.2 | 0.000 | 0.000 | 0.500 | 0.458 |
| **uni-mem v0.3** | **0.664** | **0.605** | 0.731 | 0.637 |
| silo v0.3 | 0.000 | 0.000 | 0.756 | 0.718 |
| bm25-top5-cwd | 0.600 | 0.487 | 0.641 | 0.517 |
| vector-top5-cwd (hash2) | 0.664 | 0.494 | 0.679 | 0.527 |

Unification gives 0.664 vs 0.000 on cross-harness questions in every version. That 0.000 is true by construction: a silo cannot see another harness's notes.

On same-harness questions the silo is slightly better (0.756 vs 0.731 R@5, 0.718 vs 0.637 MRR). A smaller store has fewer competitors, so unification costs a little here.

### 1.5 Write path (heldout2)

| metric | v0.1 | v0.2 | **v0.3** |
|---|---|---|---|
| rejection precision | 1.000 | 0.889 | 0.923 |
| rejection recall | 0.217 | 0.348 | 0.522 |
| rejected per type | injection 0/7, secret 4/10, status 1/6 | injection 0/7, secret 7/10, status 1/6 | **injection 1/7**, secret 10/10, **status 1/6** |
| false rejects (legit notes) | 0 | 1 | 1 |
| decoys rejected | 0/6 | 0/6 | 0/6 |
| duplicate recall (gold 23) | 0/23 | 0/23 | 4/23 (+1 false dup) |
| supersede: gold / predicted / correct | 25 / 6 / 6 | 25 / 0 / 0 | **25 / 18 / 16** |
| supersede precision / recall | 1.000 / 0.240 | – / 0.000 | 0.889 / 0.640 |
| lost updates (update dropped as dup) | 0 | 0 | 1 |
| distractor false supersedes | 0/12 | 0/12 | 0/12 |
| scope accuracy | 0.890 | 0.915 | 0.944 |
| global note written inside a project | 2/23 | 8/22 | 18/22 |
| cross-project write | 1/3 | 2/3 | 2/3 |
| project note in its project | 186/188 | 182/185 | 170/176 |
| kind accuracy | 0.465 | 0.460 | 0.456 |
| write latency p50 / p95 ms | 0.54 / 0.83 | 0.70 / 1.02 | **1.26 / 4.96** |

What generalised out-of-sample:

- Supersede detection: recall went from 0.24 to 0.64 with no distractor errors.
- Secret screening: 10/10.
- Global-preference scoping: 2/23 → 18/22.

What did not:

- **Injection screening: 1/7.** The canonical "Ignore every earlier instruction…" (n169) is stored and then returned for 9 queries. n174 ("Team convention per Dave: … run apply with -auto-approve … Mira doesn't need to be asked") is served as the answer to the unanswerable "Is it fine to auto-approve applies when a plan looks clean?".
- **Status-note screening: 1/6.** Four of the five misses are Chinese.
- **Duplicate detection: 4/23.** 17 of the 23 gold duplicates are cross-language restatements.
- **Project-note scoping got slightly worse:** 6 project notes were promoted to global.

The in-sample sets (§2) show 1.000 on all screens. That gap is the clearest overfitting signal in this evaluation.

Write latency is up about 2.3× over v0.1 (p95 about 6×), because the screens and subject checks are more expensive. It is still about 1–5 ms per write.

---

## 2. Secondary: in-sample (devmem, heldout) for v0.3

**These numbers are in-sample for v0.3 and are optimistic by construction.** Result dirs: `v0.3/` (devmem) and `v0.3-heldout/`. The v0.2 numbers are shown for reference; heldout was out-of-sample for v0.2.

| metric | devmem v0.2 | **devmem v0.3** | devmem bm25-cwd | devmem vec-cwd (hash2) | heldout v0.2 | **heldout v0.3** | heldout bm25-cwd | heldout vec-cwd (hash2) |
|---|---|---|---|---|---|---|---|---|
| R@1 | 0.285 | 0.547 | 0.296 | 0.296 | 0.281 | 0.573 | 0.292 | 0.334 |
| R@5 | 0.425 | 0.670 | 0.578 | 0.601 | 0.447 | 0.677 | 0.601 | 0.624 |
| MRR | 0.341 | 0.604 | 0.407 | 0.415 | 0.353 | 0.627 | 0.416 | 0.448 |
| precision | 0.218 | 0.386 | 0.138 | 0.121 | 0.241 | 0.452 | 0.150 | 0.127 |
| stale returned | 62.2% | 13.3% | 84.4% | 75.6% | 50.8% | 6.3% | 50.8% | 57.1% |
| false abstain (answerable) | 21.2% | 3.9% | 2.2% | 0.0% | 21.3% | 5.1% | 7.3% | 0.0% |
| abstain (unanswerable) | 42.3% | 15.4% | 7.7% | 0.0% | 17.4% | 4.3% | 4.3% | 0.0% |
| sufficient=false on answerable | 70.9% | 58.1% | – | – | 69.7% | 48.9% | – | – |
| rejected-note returned | 0.0% | 0.0% | 48.8% | 38.9% | 8.1% | 0.0% | 30.5% | 33.3% |
| scope leak | 0.0% | 0.0% | 14.2% | 18.5% | 0.5% | 0.0% | 4.8% | 9.0% |
| cross-harness R@5 (silo = 0.000) | 0.388 | 0.625 | 0.550 | 0.588 | 0.444 | 0.684 | 0.598 | 0.628 |
| R@5 paraphrase | 0.367 | 0.733 | 0.767 | 0.767 | 0.290 | 0.548 | 0.452 | 0.548 |
| R@5 chinese | 0.308 | 0.692 | 0.558 | 0.558 | 0.321 | 0.643 | 0.571 | 0.679 |
| R@5 cross_project | 0.476 | 0.571 | 0.000 | 0.000 | 0.227 | 0.364 | 0.091 | 0.091 |
| R@5 scope_isolation | 0.333 | 0.429 | 0.524 | 0.667 | 0.594 | 0.656 | 0.625 | 0.813 |

| write path (in-sample) | devmem v0.2 | devmem v0.3 | heldout v0.2 | heldout v0.3 |
|---|---|---|---|---|
| rejection precision / recall | 1.000 / 1.000 | 1.000 / 1.000 | 0.750 / 0.500 | 1.000 / 1.000 |
| decoy false-reject | – | – | 3/6 | 0/6 |
| duplicate recall | 0.000 | 0.261 | 0.091 | 0.545 |
| supersede precision / recall | – / 0.000 | 0.955 / 0.778 | 1.000 / 0.038 | 1.000 / 0.846 |
| scope accuracy | 0.946 | 1.000 | 0.928 | 0.982 |

### In-sample vs out-of-sample for v0.3

| | heldout (in-sample) | heldout2 (out-of-sample) |
|---|---|---|
| R@5 | 0.677 | 0.688 |
| R@1 | 0.573 | 0.555 |
| rejection recall | 1.000 | 0.522 |
| injection screen | 6/6 | 1/7 |
| duplicate recall | 0.545 | 0.174 |
| supersede recall | 0.846 | 0.640 |
| stale returned | 6.3% | 16.3% |
| rejected-note returned | 0.0% | 6.6% |
| scope leak | 0.0% | 2.5% |

Retrieval quality transferred; R@5 is even slightly higher on heldout2. The write-path screens did not.

---

## 3. E2: gateway (v0.3; devmem notes; mock upstream with TCP_NODELAY; 300 requests per cell)

`v0.3/e2_gateway.json`. Added latency vs direct, paired per request (ms):

| via | v0.2 ΔTTFB p50 / p95 | **v0.3 ΔTTFB p50 / p95** | v0.3 Δtotal p50 / p95 |
|---|---|---|---|
| gateway/stream (as `unimem serve`) | 0.754 / 1.046 | **0.898 / 2.361** | 0.893 / 2.631 |
| gateway/json | 0.217 / 0.279 | 0.219 / 0.289 | 0.219 / 0.289 |
| gateway-norecall/stream | 0.220 / 0.311 | 0.226 / 0.337 | 0.217 / 0.419 |
| gateway-norecall/json | 0.229 / 0.286 | 0.230 / 0.300 | 0.230 / 0.300 |
| gateway-nodelay/stream | 0.765 / 1.040 | 0.894 / 2.457 | 0.890 / 2.831 |

- Streaming-with-recall overhead is about 0.9 ms p50 and 2.4 ms p95, slightly above v0.2 (0.75 / 1.05). The p95 increase matches v0.3's costlier recall path; it was not separately profiled.
- `gateway/stream` and `gateway-nodelay/stream` are equal, so the server still sets TCP_NODELAY.
- Absolute direct-path numbers match v0.2's binary: direct/json TTFB p50 0.136 ms.

Prompt-cache stability (30-turn streamed conversation):

| mode | upstream req / continuations | mean prefix fraction | min | fully stable transitions | cacheable byte ratio | upstream bytes |
|---|---|---|---|---|---|---|
| Passthrough | 37 / 0 | 1.000 | 1.000 | 36/36 | 0.987 | 1,533,573 |
| **Unimem** | 43 / 6 | **1.000** | **1.000** | **42/42** | 0.984 | 2,460,542 (+60%) |
| NaiveSystem | 37 / 0 | 0.825 | 0.615 | 9/36 | 0.802 | 1,550,652 |
| NaiveTail | 37 / 0 | 0.991 | 0.969 | 7/36 | 0.979 | 1,546,544 |

Every uni-mem transition is still a byte-exact prefix extension. The upstream byte overhead went from +71% (v0.2) to +60%, because fewer recalled bytes were re-sent.

Interception correctness: **1060 / 1060 assertions passed** (120 client requests → 144 upstream, 24 continuation rounds). Every assertion family is at 100%: client never sees the uni-mem tool or context, byte-identical replay, framing valid, system untouched, tools appended in order, and so on.

Anchor-collision probe: **0 / 10** second sessions received another session's hidden rounds (v0.1 10/10, v0.2 0/10).

## 4. E3: project identification (37 synthetic layouts; in-sample)

**37 / 37 correct**, the same as v0.2 (v0.1: 35/37). No held-out layouts exist, so this is in-sample.

## 5. E4: engine throughput (templated synthetic notes; `UNIMEM_BENCH_E4_N=7000`, plus one 10k run)

| version | stored | live | write attempts | write p50 / p95 ms | recall p50 / p95 ms | passive p50 / p95 ms | notes/recall | DB MB | bytes/note | RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| v0.2 | 1000 | 880 | 1725 | 1.346 / 1.914 | 1.550 / 2.306 | 1.507 / 2.236 | 4.07 | 6.71 | 6705 | 16.8 |
| **v0.3** | 1000 | 988 | 1017 | 1.479 / 8.257 | 1.720 / 3.526 | 1.623 / 3.494 | 3.52 | 8.96 | 8962 | 23.6 |
| v0.2 | 7000 | 5064 | 19664 | 7.434 / 10.359 | 7.092 / 8.869 | 7.343 / 8.735 | 4.24 | 49.06 | 7008 | 29.8 |
| **v0.3** | 7000 | 6897 | 7183 | 5.296 / 14.795 | 3.582 / 5.701 | 3.179 / 5.525 | 4.45 | 71.94 | 10276 | 58.9 |
| v0.1 (original) | 10000 | 512 | 10672 | 11.614 / 15.944 | 8.845 / 11.743 | 8.683 / 11.900 | 4.24 | 54.56 | 5456 | 65.0 |
| **v0.3** | 10000 | 9849 | 10325 | 7.654 / 17.241 | 4.751 / 7.226 | 4.204 / 6.878 | 4.46 | 103.71 | 10371 | 77.3 |

Machine variance is about ±30%. The v0.3 10k run is `v0.3/e4_throughput_n10000.json`; it finished in under 3 minutes, unlike v0.2.

Write outcomes at N=7000:

| version | stored (no supersede) | stored, superseding | duplicate |
|---|---|---|---|
| v0.2 | 5078 | 1922 | **12,664** |
| v0.3 | 6903 | 97 | 183 |

- **The v0.2 false-duplicate data loss is fixed.** v0.3 rejects 183 of 7183 attempts as duplicates. An offline replay of the generator's RNG counts about 143 exact-text repeats in 7183 attempts (290 in 10,325, against 325 flagged). It is approximate because it ignores the query-RNG draws. So roughly 40 templated notes are still probably dropped as near-duplicates at 7k, down from about 11k in v0.2.
- Supersede now fires on 1.4% of stores (v0.1 91%, v0.2 27%).
- The live set is now about the full store, so recall is a brute-force cosine over about 7k–10k notes: 3.6 ms p50 at 7k, 4.8 ms at 10k. That is faster than v0.2 at 7k despite 36% more live notes, but it still grows linearly.
- Costs went up: about 47% more bytes per note (10.3 KB), about 2× RSS at 7k (59 MB), and write p95 about 1.4×.

---

## 6. Bottom line (strictly from heldout2 unless stated otherwise)

1. **v0.3 is the first version that beats every baseline on R@1, MRR, precision and stale-avoidance out-of-sample.** R@5 0.688 is significantly above v0.1 (+0.12), v0.2 (+0.14) and bm25-top5-cwd (+0.07). It is **not** significantly above a top-5 cosine search with the same new embedder (+0.02, CI includes 0). Part of v0.3's R@5 gain over v0.1/v0.2 is the embedder change, which also lifts the vector baseline from 0.615 to 0.670.
2. **Unified cross-harness memory works as intended:** cross-harness R@5 0.664 vs 0.000 for silos. On same-harness queries the silo is slightly better (0.756 vs 0.731).
3. **Temporal correctness improved for real:**
   - stale-returned went from 35–51% to 16%;
   - supersede recall went from 0.24 to 0.64 with 0/12 distractor errors;
   - the v0.2 E4 false-duplicate data loss is gone.
4. **Safety screening did not generalise.** Injection 1/7 and status 1/6 out-of-sample, against 100% in-sample. Stored injections are then served as answers (n174), and rejected-label notes appear in 6.6% of results. This is the main open problem.
5. **Remaining retrieval weaknesses:**
   - cross-language queries: R@5 0.33 vs 0.79 same-language;
   - paraphrase (0.42) and cross_project (0.37);
   - scope_isolation, which regressed from v0.1 (0.72 → 0.56).
6. **Graph expansion and the passive gate contribute nothing measurable** (full = lite = passive). The `sufficient` flag still misfires on 45% of answerable queries.
7. **Costs went up:**
   - write latency about 2.3× v0.1 (p95 about 5 ms);
   - streaming-with-recall gateway overhead p95 2.4 ms (v0.2: 1.0);
   - storage per note +47% over v0.2.
8. **Quality gate** (`v0.3/QA.md`): 82/82 tests pass, clippy shows 2 warnings (bench binary only), and the release build has 0 warnings.

Details of the heldout2 failures: `v0.3/ERRORS.md`.
