# v0.3 quality gate (HEAD 8a679a8)

Toolchain: rustc 1.99.1 (58cf8f9c3 2026-10-13), cargo 1.99.1, clippy 0.1.99; aarch64 Ubuntu 22.04 VM, 4 cores.

## `cargo test --all-targets`: pass, 82/82

| target | passed | failed | ignored |
|---|---|---|---|
| unittests `src/lib.rs` | 36 | 0 | 0 |
| unittests `src/main.rs` | 0 | 0 | 0 |
| unittests `src/bin/unimem-bench.rs` | 0 | 0 | 0 |
| `tests/engine.rs` | 2 | 0 | 0 |
| `tests/gateway_anthropic.rs` | 9 | 0 | 0 |
| `tests/gateway_chat.rs` | 7 | 0 | 0 |
| `tests/gateway_responses.rs` | 9 | 0 | 0 |
| `tests/mcp.rs` | 2 | 0 | 0 |
| `tests/regressions.rs` | 7 | 0 | 0 |
| `tests/v03.rs` | 10 | 0 | 0 |
| **total** | **82** | **0** | **0** |

Exit code 0. The test build itself produced no compiler warnings.

## `cargo clippy --all-targets`: exit 0, 2 warnings (both in the bench binary)

- `src/bin/unimem-bench.rs:1232`: `clippy::too_many_arguments` (8 of 7).
- `src/bin/unimem-bench.rs:1917`: `clippy::single_char_add_str` (`s.push_str("\n")`).

Both are reported once for the bin target and once for the bin test target. The library, `unimem` main, and `tests/*` have no clippy warnings.

## `cargo build --release`: 0 warnings

`Finished release profile [optimized]` with no warnings. The sources were touched first to force a rebuild of the crate.

## Notes

- These checks say nothing about retrieval quality. That is measured in E1 (`../FINAL.md`).
- `tests/v03.rs` was written alongside the v0.3 changes. It encodes the behaviours v0.3 targets, so a pass there is expected by construction and is not independent evidence.
