# unimem on standard long-term-memory benchmarks: LongMemEval_S and LoCoMo (retrieval)

Everything here runs offline: no API keys, CPU only. Files:

- `bench/std/download.sh`: datasets, with size and sha256 checks.
- `bench/std/embed_e5.py`: embedding cache.
- `src/bin/unimem-std-bench.rs`: runs, metrics, report.
- `bench/std/crosscheck_official.py`: compares against LongMemEval's own code.
- `bench/std/significance.py`, `bench/std/matched_budget.py`: extra analyses.
- `bench/std/qa_eval.py`: end-to-end QA. Written but not run (it needs a key).
- `bench/std/run_all.sh`: reproduces everything.

Raw per-system aggregates are in `results/longmemeval_s_retrieval.json` and `results/locomo_retrieval.json`. All tables, including NDCG@k, R_any and per-type breakdowns, are in `results/tables.md`.

## 1. Setup

### Datasets

| | file | size / sha256 | used |
|---|---|---|---|
| LongMemEval_S (Wu et al., ICLR 2025, arXiv 2410.10813) | `longmemeval_s_cleaned.json`, HF `xiaowu0162/longmemeval-cleaned` | 277,383,467 B / `d6f21ea9…a3c442` | 500 questions, 47.7 sessions and 245 user turns per question |
| LongMemEval oracle | `longmemeval_oracle.json` | 15,388,478 B / `821a2034…20c` | downloaded for QA; not needed for retrieval |
| LoCoMo (Maharana et al., ACL 2024) | `locomo10.json`, github `snap-research/locomo` | 2,805,274 B / `79fa87e9…ff4` | 10 conversations, 5,882 turns, 1,986 QA |

### Systems

All systems are evaluated on the same units and the same queries.

| system | what it is |
|---|---|
| `bm25-okapi-ref` | Reproduces LongMemEval's own `flat-bm25` baseline exactly: `rank_bm25.BM25Okapi` (k1 = 1.5, b = 0.75, idf floor 0.25·mean) over `str.split(" ")` tokens. This is the anchor for comparing with the paper. |
| `bm25` | A standard BM25: k1 = 1.2, b = 0.75, Lucene idf, lowercased, split on non-alphanumerics, Lucene English stop set, Snowball English stemmer (`rust-stemmers`). |
| `hash-dense` | Cosine over unimem's offline `hash2-512` embedder. |
| `e5-small` | Dense retrieval with **intfloat/multilingual-e5-small** (118M parameters, 384-d, multilingual, so it covers Chinese). Official ONNX export; inputs prefixed `"query: "` / `"passage: "`; max 512 tokens; masked mean pooling; L2 norm. |
| `hybrid-rrf` | Reciprocal rank fusion of `bm25` and `e5-small` (k = 60). |
| `unimem-{hash,e5}` | The real `Engine::recall` (heuristic System-One decider, `k = 10`, `Ctx::default()` so project scoping is off, `passive = false`). Every unit is written **directly into the engine's store** with `Store::insert_memory` (scope `global`, `created_at` = session timestamp), **bypassing the write screens, dedup and edge creation**. This keeps all systems on the same corpus, but it also means there are no graph edges, so this is the "without graph" configuration. |
| `unimem-prior-{hash,e5}` | The engine's own candidate scorer on the same store: anchors = BM25 top-30 ∪ cosine top-30, ranked by the calibrated `relevance_prior`. It drops the adaptive cut-off, the decider round-trip and the stale-value demotion, which gives a full ranking so Recall@10 means the same thing as for the baselines. |
| `unimem-write-{hash,e5}` | The full product path: every unit goes through `Engine::write`, i.e. the screens, dedup/supersede, and causal/entity/semantic/temporal edges, so this is the "with graph" configuration. Then `Engine::recall`. Units that are rejected or merged are lost. Recall maps a merged note back to all of its units. |

How the e5 embedder was run:

- e5 vectors were precomputed by `embed_e5.py` (Python onnxruntime 1.23.2) and cached.
- The **Rust `onnx` embedder** in `src/embed.rs` (cargo feature `onnx`, using the `ort` 2.0.0-rc.10 crate with `load-dynamic` plus the `tokenizers` crate) loads the same model through the Python wheel's `libonnxruntime.so`.
- It reproduces the cache (`results/onnx_parity.json`): minimum cosine with the Python vectors is 0.99999988 over 203 passages and 51 queries.
- The Rust path runs at about 100–150 short passages/s on 4 cores. It uses `"passage: "` for notes and `"query: "` for recall queries via the new `Embedder::embed_query`. `src/engine.rs` (not changed here) still embeds recall queries with `embed_one`. The cached embedder used in this bench routes them to the query cache, but in production with `kind = "onnx"` recall queries would get the passage prefix. **TODO for the engine owner:** call `embed_query` in `Engine::recall`.
- `ort`'s own runtime download (`cdn.pyke.io`) is not needed and was not used.

### Protocol

**LongMemEval** follows `src/retrieval/run_retrieval.py` and `eval_utils.py` from the official repo, reproduced exactly:

- Each question gets its own index over its own haystack.
- Keys and values are **user-side utterances only**, as in the paper (§5.1: "When sessions or rounds are used as the key, we only keep the user-side utterances").
- Turn granularity: one unit per user turn. Gold = turns with `has_answer` inside `answer_*` sessions.
- Session granularity: one unit per session, made of its user turns joined by `" "`. Gold = evidence sessions that contain at least one `has_answer` user turn.
- Skipped, as in the official code: the 30 abstention questions (`_abs`), plus 51 questions whose evidence is only on the assistant side. Since only user turns are indexed, they have no target. **419 of 500 questions are evaluated.** For the same reason, `single-session-assistant` has only 5 evaluated questions.
- Metrics:
  - `recall_all@k` is the paper's "Recall" (`print_retrieval_metrics.py` prints `recall_all`). All gold units must be in the top k.
  - `ndcg_any@k` is the official DCG.
  - `recall_any@k` is also reported in `tables.md`.
- Turn-level runs also report session metrics through the official `evaluate_retrieval_turn2session`. Its quirk is preserved: the non-evidence turns of an evidence session are renamed `noans_…`, so they count as a different session. `tables.md` additionally has a `session_true` variant that uses the real session ids.
- **The implementation matches the official code exactly.** `crosscheck_official.py` re-runs `rank_bm25` and the official metric functions on all 419 questions; every mean is identical and the maximum per-question difference is 1e-16 (`results/crosscheck_official.txt`).
- Time-aware retrieval: the question date is not used. All haystack sessions fall on or before the question's day, so a date filter would remove nothing at day resolution. 75 gold sessions are timestamped later on the same day, so a filter at minute resolution would remove gold. The paper's time-aware *query expansion* needs an LLM to extract time ranges, so it is out of scope here, as is the "K = V + fact" key expansion.

**LoCoMo**:

- One index per conversation. One unit per dialog turn (`dia_id`), with text `"<speaker>: <text>"` plus `" [shares <blip_caption>]"` when the turn shares an image.
- Gold = the QA's `evidence` ids. Malformed ids were normalised: `"D8:6; D9:17"` is split, `"D:11:26"` becomes `D11:26`, `"D30:05"` becomes `D30:5`. 3 ids could not be parsed; 4 questions with no valid evidence are skipped. **1,982 questions remain: 1,536 in categories 1–4 and 446 in category 5.**
- Metric: "recall accuracy" = fraction of evidence turns found in the top k, as in LoCoMo's `evaluation.py`, per category. Category ids in `locomo10.json`: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
- Headline numbers exclude category 5, which is reported separately. The "all 1–5" column is the one comparable to the paper's "Overall".

### Environment

- 4-core aarch64 VM, 3.8 GB RAM, Linux 6.8.
- rustc 1.99.1, release build.
- Python 3.10, onnxruntime 1.23.2, tokenizers 0.23.

Runtime:

| step | time |
|---|---|
| e5 embedding: 93,931 unique LongMemEval user turns | ≈ 17 min |
| e5 embedding: 18,363 unique sessions | ≈ 17 min |
| e5 embedding: 5,880 LoCoMo turns + 2,474 queries | ≈ 2 min |
| Overall embedding throughput | about 3.9k tokens/s |
| All baselines + `unimem` / `unimem-prior` on both datasets | about 4 min in total |
| `unimem-write-*` on LongMemEval turn granularity | about 3.2 min per embedder (0.45 s per question for about 245 writes) |

Per-query search latency: BM25 ≤ 0.03 ms, e5 ≤ 0.2 ms, engine recall 2–6 ms (`search_ms_mean` in the json).

## 2. LongMemEval_S results (419 questions)

### Session-granularity index

Metrics are session-level.

| system | R@5 | NDCG@5 | R@10 | NDCG@10 | notes/query |
|---|---|---|---|---|---|
| bm25-okapi-ref (= official flat-bm25) | 0.742 | 0.771 | 0.823 | 0.794 | ranking |
| bm25 (stemmed, k1 = 1.2) | 0.859 | 0.883 | 0.924 | 0.896 | ranking |
| hash-dense | 0.632 | 0.692 | 0.776 | 0.729 | ranking |
| e5-small | 0.912 | 0.903 | 0.969 | 0.915 | ranking |
| **hybrid-rrf** | **0.921** | **0.924** | **0.971** | **0.933** | ranking |
| unimem-prior-hash | 0.850 | 0.881 | 0.924 | 0.894 | ranking |
| unimem-prior-e5 | 0.847 | 0.879 | 0.938 | 0.896 | ranking |
| unimem-hash (recall) | 0.561 | 0.727 | 0.561 | 0.727 | 1.71 |
| unimem-e5 (recall) | 0.618 | 0.760 | 0.618 | 0.760 | 1.89 |
| unimem-write-hash (write + recall) | 0.458 | 0.656 | 0.458 | 0.656 | 1.64 |
| unimem-write-e5 (write + recall) | 0.494 | 0.677 | 0.494 | 0.676 | 1.84 |

### Turn-granularity index

T = turn-level metric; S = official turn→session metric.

| system | T R@5 | T NDCG@5 | T R@10 | T NDCG@10 | S R@5 | S R@10 | S NDCG@10 | notes/query |
|---|---|---|---|---|---|---|---|---|
| bm25-okapi-ref | 0.489 | 0.531 | 0.592 | 0.564 | 0.527 | 0.644 | 0.572 | ranking |
| bm25 | 0.654 | 0.686 | 0.754 | 0.717 | 0.716 | 0.799 | 0.723 | ranking |
| hash-dense | 0.499 | 0.570 | 0.609 | 0.602 | 0.532 | 0.659 | 0.609 | ranking |
| e5-small | 0.663 | 0.695 | 0.835 | 0.737 | 0.788 | 0.885 | 0.746 | ranking |
| **hybrid-rrf** | **0.721** | **0.742** | **0.838** | **0.773** | **0.811** | **0.900** | **0.783** | ranking |
| unimem-prior-hash | 0.678 | 0.695 | 0.764 | 0.720 | 0.740 | 0.821 | 0.729 | ranking |
| unimem-prior-e5 | 0.671 | 0.693 | 0.783 | 0.727 | 0.752 | 0.838 | 0.735 | ranking |
| unimem-hash | 0.415 | 0.577 | 0.415 | 0.577 | 0.427 | 0.427 | 0.577 | 2.35 |
| unimem-e5 | 0.458 | 0.602 | 0.465 | 0.604 | 0.473 | 0.480 | 0.604 | 2.83 |
| unimem-write-hash | 0.382 | 0.548 | 0.382 | 0.548 | 0.389 | 0.389 | 0.548 | 2.20 |
| unimem-write-e5 | 0.410 | 0.557 | 0.418 | 0.561 | 0.420 | 0.425 | 0.562 | 2.42 |

### Per question type, turn-level R@10

Number of evaluated questions per type in parentheses.

| system | single-session-user (64) | single-session-assistant (5) | single-session-preference (30) | multi-session (121) | temporal-reasoning (127) | knowledge-update (72) |
|---|---|---|---|---|---|---|
| bm25-okapi-ref | 0.844 | 0.600 | 0.400 | 0.372 | 0.614 | 0.778 |
| bm25 | 0.984 | 0.800 | 0.500 | 0.587 | 0.756 | 0.931 |
| e5-small | 0.984 | 1.000 | 0.700 | 0.736 | 0.795 | 0.986 |
| hybrid-rrf | 0.969 | 1.000 | 0.733 | 0.711 | 0.827 | 0.986 |
| unimem-prior-e5 | 1.000 | 0.800 | 0.533 | 0.628 | 0.787 | 0.944 |
| unimem-e5 (recall) | 0.922 | 0.600 | 0.333 | 0.265 | 0.409 | 0.542 |
| unimem-write-e5 | 0.891 | 0.600 | 0.300 | 0.231 | 0.370 | 0.431 |

The session-level breakdown for all systems is in `tables.md`. Same ordering there; for example, multi-session R@10 is 0.975 for e5-small, 0.901 for unimem-prior-e5 and 0.438 for unimem-e5.

### Comparison with published numbers (LongMemEval paper, arXiv 2410.10813 v2, HTML)

Values for the "K = V" key design, i.e. plain value-as-key retrieval:

| paper, Table 9 ("A comparison between three retrievers", K = V) | session R@5 | NDCG@5 | R@10 | NDCG@10 | round R@5 | NDCG@5 | R@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.634 | 0.516 | 0.710 | 0.540 | 0.472 | 0.352 | 0.538 | 0.372 |
| Contriever | 0.723 | 0.634 | 0.823 | 0.663 | 0.589 | 0.454 | 0.747 | 0.495 |
| Stella V5 1.5B | 0.720 | 0.594 | 0.794 | 0.615 | 0.660 | 0.498 | 0.784 | 0.528 |
| *paper, Table 3, LongMemEval_M, Stella, K = V* | 0.706 | 0.617 | 0.783 | 0.638 | 0.582 | 0.481 | 0.692 | 0.512 |
| **ours on S**: bm25-okapi-ref (same algorithm as the paper's BM25) | 0.742 | 0.771 | 0.823 | 0.794 | 0.489 | 0.531 | 0.592 | 0.564 |
| ours on S: e5-small | 0.912 | 0.903 | 0.969 | 0.915 | 0.663 | 0.695 | 0.835 | 0.737 |
| ours on S: hybrid-rrf | 0.921 | 0.924 | 0.971 | 0.933 | 0.721 | 0.742 | 0.838 | 0.773 |
| ours on S: unimem-prior-e5 | 0.847 | 0.879 | 0.938 | 0.896 | 0.671 | 0.693 | 0.783 | 0.727 |

How to read this comparison:

- **The two are not directly comparable. The paper's retrieval tables are most likely on LongMemEval_M, which has about 500 sessions per question; we ran S, which has about 50.**
  - §5 says "Using LongMemEval M". Table 9 does not name its dataset, but its Stella "K = V + summary" and "K = V + keyphrase" session rows are identical to Table 3, which is on M.
  - The paper reports no retrieval table for S.
  - Our exact re-implementation of their BM25, run on S, scores much higher on NDCG than their BM25 row (0.771 vs 0.516). That gap is what ten times fewer distractors would produce.
  - Our "round" is the paper's round key: user-side utterances only, one per round, ids `<session>_<turn#>`. The paper's round values also add the assistant reply for reading, which does not affect retrieval.
- Gains of each system over the paper's BM25 (re-implemented on S):

  | | session R@5 | session R@10 | round R@5 | round R@10 |
  |---|---|---|---|---|
  | e5-small (118M) | +0.170 | +0.146 | +0.174 | +0.243 |
  | paper: Contriever | +0.089 | | | |
  | paper: Stella 1.5B | +0.086 | | +0.188 | +0.246 |
  | standard stemmed BM25 | +0.117 | +0.101 | | |

  In relative terms, e5-small is on par with the paper's Stella 1.5B at round level and above it at session level. The standard stemmed BM25 alone gains +0.117 / +0.101 at session level. The paper's BM25 baseline (no lowercasing, whitespace tokens) is weak.

### Significance

Paired bootstrap, 2000 resamples, 95% CI; win/loss counts are per question (`results/significance.txt`).

| comparison | metric | Δ | 95% CI | win / loss |
|---|---|---|---|---|
| unimem-prior-e5 − hybrid-rrf | session-index R@10 | −0.033 | [−0.053, −0.014] | 2 / 16 |
| unimem-prior-e5 − hybrid-rrf | turn-index R@10 | −0.055 | [−0.079, −0.033] | 3 / 26 |
| unimem-prior-e5 − e5-small | turn-index R@10 | −0.053 | [−0.084, −0.024] | 14 / 36 |
| unimem-prior-hash − bm25 | session R@10 | +0.000 | [−0.017, +0.017] | 7 / 7 |
| unimem-prior-hash − bm25 | turn R@10 | +0.010 | [−0.017, +0.033] | 16 / 12 |
| unimem-prior-e5 − unimem-prior-hash | session R@10 | +0.014 | [+0.002, +0.029] | 7 / 1 |
| unimem-prior-e5 − unimem-prior-hash | turn R@10 | +0.019 | [+0.005, +0.036] | 10 / 2 |
| e5-small − bm25 | session R@10 | +0.045 | [+0.021, +0.069] | 23 / 4 |
| e5-small − bm25 | turn R@10 | +0.081 | [+0.045, +0.117] | 50 / 16 |

## 3. LoCoMo results (locomo10, dialog-turn units)

Evidence recall per category. R@k = fraction of evidence turns in the top k.

| system | R@5 (1–4) | R@10 (1–4) | R@25 (1–4) | R@50 (1–4) | multi-hop (1) R@10 | temporal (2) R@10 | open-dom. (3) R@10 | single-hop (4) R@10 | adversarial (5) R@10 | all 1–5: R@5 / 10 / 25 / 50 | notes/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25-okapi-ref | 0.295 | 0.377 | 0.480 | 0.554 | 0.123 | 0.384 | 0.161 | 0.483 | 0.488 | 0.315 / 0.402 / 0.502 / 0.575 | ranking |
| bm25 | 0.470 | 0.553 | 0.653 | 0.720 | 0.278 | 0.654 | 0.278 | 0.636 | 0.639 | 0.486 / 0.572 / 0.679 / 0.741 | ranking |
| hash-dense | 0.341 | 0.422 | 0.516 | 0.610 | 0.162 | 0.490 | 0.151 | 0.513 | 0.433 | 0.344 / 0.424 / 0.517 / 0.610 | ranking |
| e5-small | 0.482 | 0.569 | 0.682 | 0.777 | 0.319 | 0.670 | 0.297 | 0.643 | 0.461 | 0.453 / 0.544 / 0.655 / 0.754 | ranking |
| hybrid-rrf | 0.551 | 0.629 | **0.734** | 0.807 | **0.379** | **0.737** | **0.328** | 0.704 | 0.612 | 0.540 / 0.625 / **0.739** / 0.815 | ranking |
| unimem-prior-hash | 0.543 | 0.620 | 0.695 | 0.723 | 0.346 | 0.728 | 0.296 | **0.707** | 0.701 | **0.560** / 0.638 / 0.714 / 0.741 | ranking |
| unimem-prior-e5 | 0.542 | 0.619 | 0.697 | **0.811** | 0.345 | 0.723 | 0.297 | **0.707** | **0.709** | 0.559 / **0.639** / 0.717 / **0.822** | ranking |
| unimem-hash (recall) | 0.471 | 0.483 | 0.483 | 0.483 | 0.225 | 0.601 | 0.220 | 0.552 | 0.549 | 0.487 / 0.498 / 0.498 / 0.498 | 2.88 |
| unimem-e5 (recall) | 0.483 | 0.504 | 0.504 | 0.504 | 0.251 | 0.624 | 0.226 | 0.574 | 0.577 | 0.499 / 0.521 / 0.521 / 0.521 | 3.55 |
| unimem-write-hash | 0.458 | 0.467 | 0.467 | 0.467 | 0.216 | 0.575 | 0.198 | 0.539 | 0.528 | 0.471 / 0.481 / 0.481 / 0.481 | 2.87 |
| unimem-write-e5 | 0.409 | 0.438 | 0.452 | 0.453 | 0.221 | 0.524 | 0.204 | 0.504 | 0.502 | 0.425 / 0.453 / 0.467 / 0.468 | 3.42 |

Significance (R@10, categories 1–4):

| comparison | Δ | 95% CI | win / loss |
|---|---|---|---|
| unimem-prior-e5 − hybrid-rrf | +0.014 | [−0.001, +0.030] | 199 / 177 |
| unimem-prior-hash − bm25 | +0.066 | [+0.052, +0.080] | 230 / 68 |
| unimem-prior-e5 − e5-small | +0.095 | [+0.073, +0.116] | 417 / 216 |

**Published numbers.** LoCoMo paper, Table 3 ("Question answering performance of RAG-based gpt-3.5-turbo-16k"), retriever DRAGON, retrieval unit = dialog:

| | k = 5 | k = 10 | k = 25 | k = 50 |
|---|---|---|---|---|
| Overall recall accuracy | 58.8 | 67.5 | 79.9 | 84.8 |
| Per category (single-hop / multi-hop / temporal / open-domain / adversarial) | 66.2 / 34.4 / 89.2 / 38.5 / 45.7 | 72.8 / "247.4" (sic, a typo in the paper) / 97.3 / 53.8 / 54.3 | | |

Our best "all 1–5" numbers:

| | R@5 | R@10 | R@25 | R@50 |
|---|---|---|---|---|
| unimem-prior-e5 | 0.559 | 0.639 | 0.717 | 0.822 |
| hybrid-rrf | 0.540 | 0.625 | 0.739 | 0.815 |

Our numbers are a few points below DRAGON. Differences in protocol:

1. The paper used the full 50-conversation LoCoMo (7,512 questions). Only `locomo10` (10 conversations, 1,986 questions) is public.
2. LoCoMo's retriever embeds each turn with its session date prefixed, `"(<date>) <speaker> said, "…""`. That explains its very high temporal recall: "when did …" questions match the date text. Our units carry no date in the text; dates live in metadata only.
3. DRAGON is a retriever trained specifically for dialogue and QA.

How the per-category numbers line up with the paper's columns is not documented in the dataset. We used the mapping given above.

## 4. Findings

1. **unimem's ranking function is a strong lexical-hybrid ranker. Its cut-off, not its ranking, limits Recall@k.**
   - `unimem-prior-*` (the engine's anchors plus the calibrated `relevance_prior`) ties a well-tuned stemmed BM25 on LongMemEval: Δ R@10 = +0.000 at session level and +0.010 at turn level, both n.s.
   - It is the best system on LoCoMo at R@5 and R@10: +0.066 over BM25 (significant) and +0.014 over BM25+e5 RRF (n.s.).
   - On LongMemEval it is significantly **below** the dense baselines: −0.033 session R@10 and −0.055 turn R@10 vs hybrid-rrf.
   - The weakest types are multi-session (0.628 vs 0.736 for e5 at turn R@10) and single-session-preference (0.533 vs 0.733). Both are paraphrase-heavy, which is exactly where a real embedding model helps.
2. **The real `engine.recall` returns 1.7–3.5 notes per query** (adaptive threshold: `T_RELEVANT`, `KEEP_RATIO`). R@10 is therefore effectively R@2–3:
   - LongMemEval turn R@10 is 0.465 for `unimem-e5` vs 0.838 for hybrid.
   - Questions that need several evidence turns suffer most: multi-session R@10 is 0.265 vs 0.628 for its own ranking.
   - This is a design choice (precision for prompt injection), but it means unimem is not a top-k retriever. At **matched depth**, cutting every baseline at the number of notes unimem returned for that question (`results/matched_budget.txt`), unimem keeps up:

     | setting | unimem-e5 | hybrid | bm25 | e5 |
     |---|---|---|---|---|
     | LongMemEval session, hit rate | 0.916 | 0.921 | 0.919 | 0.897 |
     | LongMemEval turn, hit rate | 0.826 | 0.850 | 0.821 | 0.804 |
     | LoCoMo evidence recall | 0.520 | 0.456 | 0.420 | 0.369 |

     Precision at that depth is the best or near-best of all systems.
   - Recommendation: when the caller asks for k notes (the `unimem_recall` tool, or this bench), relax the cut-off. Also add a "multi-evidence" mode for aggregation questions such as "how many …" or "in total".
3. **A real embedder helps unimem's ranking only a little: +0.014 / +0.019 R@10, significant but small.**
   - Plain e5 beats plain BM25 by +0.045 / +0.081. The logistic prior was fitted on hash-embedder features: cosine weights are 0.25 / 0.6, against 1.2 / 1.5 / 1.7 for coverage, BM25 and relative coverage. It therefore under-uses a good embedder.
   - Recommendation: refit `relevance_prior` per embedder, or fuse ranks the way RRF does.
4. **The write path loses evidence on chat-style data** (`unimem-write-*` vs the direct-insert `unimem-*`).
   - LongMemEval turns:
     - 1,658 of 102k writes are rejected: 612 are "too long" (over 2,400 chars), 600 "transient status", 408 "too short", 25 "injection", 13 "credential".
     - Sessions: 1,206 are rejected as too long.
   - **With e5, the near-duplicate check merges distinct turns:**

     | | turns merged as "duplicates" |
     |---|---|
     | LongMemEval, e5 | 22,763 of 102k (22%) |
     | LongMemEval, hash | 3,978 |
     | LoCoMo, e5 | 2,662 of 5,882 (45%) |
     | LoCoMo, hash | 106 |

     `NEAR_DUP_COS = 0.97` and `dup_prior` were calibrated on hash cosines. E5 cosines are compressed: on LoCoMo turns, random pairs have median cosine 0.823 (p5 0.773, p95 0.873), and 9.6% of turns have some other turn at ≥ 0.97. This is a known anisotropy of the E5 family, so 0.97 does not mean near-duplicate for e5. **This is a real bug for anyone who enables a transformer embedder.** The dedup thresholds must be per-embedder, or computed on a calibrated similarity.
   - Note: these benchmarks feed raw chat turns. In real use, notes are distilled facts written by the agent, which is what the screens were designed for. Losing raw turns here is expected; the e5 dedup problem is not.
5. **The graph contributes nothing measurable.**
   - In the direct-insert runs there are no edges, so they are the "without graph" ablation. In the write-path runs, graph expansion contributed 0 (LongMemEval turn), 1 (session) and 28–38 (LoCoMo) notes out of about 5.7k returned.
   - The write path's losses come from screens and dedup, not from graph effects.
6. **Baselines.** A properly tokenized BM25 (lowercase, stemming, stop words) gains 0.10–0.18 over the whitespace-split `BM25Okapi` that LongMemEval used. Any comparison with the paper's BM25 row should account for that.

## 5. Not done, and why

- **The LLM-dependent parts of LongMemEval's best design** (fact key expansion, "K = V + fact"; time-aware query expansion; chain-of-note reading) need an LLM, so they are out of scope offline. Table 3 of the paper reports +0.06 to +0.09 R@k for fact expansion on M.
- **End-to-end QA was not run.** `bench/std/qa_eval.py` is ready:
  - Get retrieval output with `unimem-std-bench dump --system <any> …`.
  - Reader: any OpenAI-compatible endpoint, using LongMemEval's official `flat-turn` / `flat-session` JSON history prompt.
  - Judge: the official GPT-4o judge prompts from `src/evaluation/evaluate_qa.py`, reproduced verbatim (per question type and for abstention; temperature 0, max_tokens 10, label = "yes" in the reply).
  - LoCoMo: the official RAG prompt and QA prompts (two-option category-5 format; date hint for category 2) and the official token-F1 scorer with Porter stemming.
  - `--dry-run` builds all 500 and 1,986 prompts offline. That was tested; no scores were produced.
- LongMemEval_M (2.7 GB) was not run because of the disk limit (3 GB free).
- Larger embedders (bge-m3 at 568M parameters) were not tried within 3.8 GB of RAM and the time budget. The `onnx` backend loads any mean-pooled encoder from a folder with `model.onnx` and `tokenizer.json`.

## 6. Reproduce

```bash
CACHE=/path/to/bench-cache bench/std/run_all.sh
```

This script downloads the data and model, embeds (resumable), runs all systems, writes the report, and runs the significance and official cross-check scripts.

To use the Rust ONNX embedder:

```bash
cargo build --release --features onnx
ORT_DYLIB_PATH=<site-packages>/onnxruntime/capi/libonnxruntime.so.1.23.2
```

then set `[embedder] kind = "onnx"` and `model = "<dir with model.onnx + tokenizer.json>"`.
