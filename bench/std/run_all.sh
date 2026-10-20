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
