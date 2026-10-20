#!/usr/bin/env bash
# Reproduce bench/std/results from scratch (offline after the downloads).
#   CACHE=/path/to/bench-cache bench/std/run_all.sh
# Every step is resumable and can be re-invoked after an interruption.
set -euo pipefail
CACHE="${CACHE:-./bench-cache}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$CACHE/models/me5s"
bash "$ROOT/bench/std/download.sh" "$CACHE"
if [ ! -f "$MODEL/model.onnx" ]; then
  mkdir -p "$MODEL"
  for f in model.onnx tokenizer.json config.json special_tokens_map.json tokenizer_config.json; do
    curl -sSfL --retry 3 -o "$MODEL/$f" "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/$f"
  done
fi
echo "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665  $MODEL/model.onnx" | sha256sum -c -
# 1. embeddings (python onnxruntime; ~35 min on 4 CPU cores). Each call stops after --budget seconds; loop until done.
for set in lme_query locomo_query locomo_turn lme_turn lme_session; do
  until python3 "$ROOT/bench/std/embed_e5.py" --cache "$CACHE" --model "$MODEL" --set "$set" --budget 600 | tee /dev/stderr | grep -q '^done'; do :; done
done
# 2. retrieval runs
(cd "$ROOT" && cargo build --release --bin unimem-std-bench)
B="${CARGO_TARGET_DIR:-$ROOT/target}/release/unimem-std-bench"
SYS=bm25-okapi-ref,bm25,hash-dense,e5-small,hybrid-rrf,unimem-hash,unimem-prior-hash,unimem-e5,unimem-prior-e5,unimem-write-hash,unimem-write-e5
"$B" run --cache "$CACHE" --dataset locomo --gran turn    --systems "$SYS"
"$B" run --cache "$CACHE" --dataset lme    --gran session --systems "$SYS"
"$B" run --cache "$CACHE" --dataset lme    --gran turn    --systems "$SYS"
"$B" report --cache "$CACHE" --out "$ROOT/bench/std/results"
# 3. optional checks: official LongMemEval code (needs `pip install rank_bm25`), significance, Rust ONNX embedder parity
python3 "$ROOT/bench/std/crosscheck_official.py" --cache "$CACHE" | tee "$ROOT/bench/std/results/crosscheck_official.txt" || true
python3 "$ROOT/bench/std/significance.py" "$CACHE" | tee "$ROOT/bench/std/results/significance.txt"
# ORT_DYLIB_PATH=.../onnxruntime/capi/libonnxruntime.so.1.23.2 cargo run --release --features onnx --bin unimem-std-bench -- verify-onnx --cache "$CACHE" --model "$MODEL"
# 4. v0.5: dev-split evidence -> prior fit, unimem re-runs into runs-v05, test-split + full reports, significance
for g in session turn; do for e in hash e5; do
  "$B" evidence --cache "$CACHE" --dataset lme --gran $g --embedder $e --out "$CACHE/ev-lme-$g-$e.jsonl"
done; done
mkdir -p "$CACHE/runs-v05"
for f in "$CACHE"/runs/*-{bm25-okapi-ref,bm25,hash-dense,e5-small,hybrid-rrf}.jsonl; do cp -n "$f" "$CACHE/runs-v05/"; done
V5=unimem-prior-hash,unimem-prior-e5,unimem-topk-hash,unimem-topk-e5,unimem-hash,unimem-e5,unimem-write-hash,unimem-write-e5
UNIMEM_STD_RUNS=runs-v05 "$B" run --cache "$CACHE" --dataset locomo --gran turn    --systems "$V5"
UNIMEM_STD_RUNS=runs-v05 "$B" run --cache "$CACHE" --dataset lme    --gran session --systems "$V5"
UNIMEM_STD_RUNS=runs-v05 "$B" run --cache "$CACHE" --dataset lme    --gran turn    --systems "$V5"
for s in test all; do UNIMEM_STD_RUNS=runs-v05 "$B" report --cache "$CACHE" --out "$ROOT/bench/results/v0.5/std" --split $s; done
python3 -I "$ROOT/bench/std/significance_v05.py" "$CACHE" runs-v05 "$ROOT/bench/results/v0.5/std/significance.json" | tee "$ROOT/bench/results/v0.5/std/significance.txt"
# (prior fit: python3 -I bench/std/fit_prior.py bench/results/v0.5/prior_fit.json $CACHE/ev-*.jsonl <E1 evidence via UNIMEM_BENCH_EVIDENCE>)
