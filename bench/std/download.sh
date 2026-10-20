#!/usr/bin/env bash
# Download standard long-term-memory benchmark datasets (NOT committed to the repo).
#   LongMemEval (Wu et al., ICLR 2025, arXiv:2410.10813) -- HF xiaowu0162/longmemeval-cleaned
#   LoCoMo      (Maharana et al., ACL 2024)              -- github snap-research/locomo
# Usage: bench/std/download.sh [CACHE_DIR]   (default: $UNIMEM_BENCH_CACHE or ./bench-cache)
set -euo pipefail
CACHE="${1:-${UNIMEM_BENCH_CACHE:-./bench-cache}}"
mkdir -p "$CACHE/data"
cd "$CACHE/data"

HF=https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main
# name | url | sha256 (HF LFS oid / computed for LoCoMo) | bytes
FILES=(
  "longmemeval_s_cleaned.json|$HF/longmemeval_s_cleaned.json|d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442|277383467"
  "longmemeval_oracle.json|$HF/longmemeval_oracle.json|821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c|15388478"
  "locomo10.json|https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json|79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4|2805274"
)
fail=0
for row in "${FILES[@]}"; do
  IFS='|' read -r name url sha size <<<"$row"
  if [ ! -f "$name" ]; then
    echo "downloading $name"
    curl -sSfL --retry 3 -o "$name.part" "$url" && mv "$name.part" "$name"
  fi
  got_size=$(stat -c %s "$name" 2>/dev/null || stat -f %z "$name")
  got_sha=$( (sha256sum "$name" 2>/dev/null || shasum -a 256 "$name") | cut -d' ' -f1)
  if [ "$got_size" != "$size" ] || [ "$got_sha" != "$sha" ]; then
    echo "MISMATCH $name size=$got_size sha=$got_sha (expected $size $sha)"; fail=1
  else
    echo "ok $name ($size bytes)"
  fi
done
exit $fail
