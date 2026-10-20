# Contributing to unimem

Thanks for your interest! unimem is young and experimental, so these contributions help the most:

- **Reports from real harness traffic.** Examples: a Claude Code or Codex version that breaks the anchors, a
  compaction that wasn't detected, a stream that got mangled. Redact secrets first. `RUST_LOG=unimem=debug` and
  `unimem trace` are your friends.
- **Benchmark-backed improvements** to retrieval, the write screens or the deciders.
- **Docs**: in particular, an English translation of `docs/DESIGN.md`.

Please open an issue before you start a large change, so we can agree on the approach.

## Development setup

You need a recent stable Rust toolchain, installed via [rustup](https://rustup.rs). SQLite is bundled through
`rusqlite`, so no system libraries are required.

```sh
git clone https://github.com/mapix/unimem && cd unimem
cargo build                 # debug build
cargo build --release       # what you'll actually run
```

Run a scratch instance that cannot touch your real memory:

```sh
export UNIMEM_DATA_DIR=/tmp/unimem-dev UNIMEM_LISTEN=127.0.0.1:4799
cargo run --release -- demo-seed
cargo run --release -- serve        # UI at http://127.0.0.1:4799/ui
```

### Optional: the ONNX embedder

`--features onnx` builds a local transformer embedder (`ort` with `load-dynamic`, plus `tokenizers`). ONNX Runtime
is loaded at runtime from `ORT_DYLIB_PATH`; the Python `onnxruntime` wheel's `libonnxruntime.so`/`.dylib` works. CI
builds this feature, but a failure there doesn't block a merge.

```sh
cargo build --release --features onnx
```

## Tests

```sh
cargo test                       # unit + integration tests (mock upstreams, no network, no keys)
cargo test --test v04            # one integration suite
cargo clippy --all-targets
```

The integration tests in `tests/` start the real gateway against mock Anthropic / OpenAI upstreams and check
framing, interception, overlay replay, budgets and the UI API. **Every behaviour change needs a test.** A gateway
change needs one for each dialect it touches: `gateway_anthropic.rs`, `gateway_responses.rs`, `gateway_chat.rs`.

## Benchmarks

If a change affects retrieval, the write path or the gateway, run the relevant suite and put the before/after
numbers in the PR.

```sh
# Own suites: E1 memory quality, E2 gateway overhead/correctness, E3 project detection, E4 throughput, E5 long session
cargo run --release --bin unimem-bench -- e1 --data bench/data/heldout2.json --out /tmp/bench-out
cargo run --release --bin unimem-bench -- all --data bench/data/heldout2.json --out /tmp/bench-out

# Standard benchmarks (LongMemEval_S + LoCoMo retrieval). Downloads about 300 MB; the e5 embedding step takes about 35 min on 4 cores
CACHE=/path/to/bench-cache bench/std/run_all.sh
```

Ground rules for benchmark claims:

- **`heldout2.json` is the out-of-sample set.** Don't tune on it. If your change was informed by its failures, say
  so, because the numbers then become in-sample. A new held-out set is the better contribution.
- Report significance (paired bootstrap / sign test), not just point estimates. See `bench/std/significance.py`.
- Committed result files should not contain machine-specific absolute paths.

## Style

- `cargo fmt --check` (config in `rustfmt.toml`) and `cargo clippy --all-targets` must be clean; CI enforces both.
- Keep the gateway **fail-open**. Any error in the memory layer must degrade to forwarding the original request.
- Keep changes **cache-safe**. Never modify the system prompt or reorder existing content. Anything injected must be
  replayed byte-for-byte on later requests (overlays).
- Thresholds belong in code or config, with a comment saying where the number comes from.
- No new network calls by default. Optional backends must be opt-in, with their key read from an env var named by
  `api_key_env`.
- Commit messages: a short imperative summary line. Mention the bench delta when relevant.

## Pull requests

1. Fork and branch from `main`.
2. Make sure `cargo test` and `cargo clippy --all-targets` pass.
3. Update `README.md` / `CHANGELOG.md` (under "Unreleased") if user-visible behaviour changed.
4. Fill in the PR template.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
