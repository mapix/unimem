## What and why

<!-- What does this change, and why? Link the issue if there is one (Fixes #123). -->

## How it was tested

- [ ] `cargo test` passes
- [ ] `cargo clippy --all-targets` is clean for the touched code
- [ ] New or changed behaviour has a test (gateway changes: every affected dialect)

## Benchmarks (if retrieval, write path or gateway changed)

<!-- Before/after numbers, e.g. `unimem-bench e1 --data bench/data/heldout2.json`, and whether the change was tuned
on that set (that makes it in-sample). -->

## Checklist

- [ ] Gateway stays fail-open and cache-safe (system prompt untouched, injections replayed byte-for-byte)
- [ ] No new default network calls; keys only via `api_key_env`
- [ ] README / CHANGELOG ("Unreleased") updated for user-visible changes
