# Publishing unimem to GitHub

These are the commands to run on your Mac, from the repository root (for example `~/workspace/uni-mem`).

## 0. Pre-flight

```sh
cd ~/workspace/uni-mem
git status                                   # should be clean
git log --oneline | head                     # check history
git grep -n "/sessions/" -- ':!*.json'       # should print nothing (VM paths)
cargo test && cargo build --release          # optional, takes a few minutes

# Optional: rewrite the placeholder author (commits were made as "unimem <a@b>")
git config user.name  "mapix"
git config user.email "<your GitHub email or ID+mapix@users.noreply.github.com>"
# Rewrite existing commits (safe here because nothing has been pushed yet):
git rebase -r --root --exec 'git commit --amend --no-edit --reset-author'
```

## 1. Create the repository and push (GitHub CLI)

```sh
gh auth status || gh auth login
git branch -M main
gh repo create mapix/unimem --public --source=. --remote=origin \
  --description "One shared, project-aware long-term memory for every coding agent: a local LLM gateway for Claude Code, Codex and any OpenAI/Anthropic-compatible client." \
  --homepage "https://github.com/mapix/unimem" \
  --push
gh repo edit mapix/unimem \
  --add-topic llm --add-topic memory --add-topic ai-agents --add-topic claude-code \
  --add-topic codex --add-topic mcp --add-topic rust --add-topic long-term-memory \
  --enable-discussions
```

### Fallback without `gh`

Create an **empty** repository named `unimem` at https://github.com/new, with no README, license or .gitignore. Then:

```sh
git branch -M main
git remote add origin git@github.com:mapix/unimem.git    # or https://github.com/mapix/unimem.git
git push -u origin main
```

## 2. Tag v0.5.0

```sh
git tag -a v0.5.0 -m "unimem v0.5.0"
git push origin v0.5.0
```

## 3. GitHub release (optional)

```sh
gh release create v0.5.0 --title "unimem v0.5.0" --notes-file - <<'EOF'
First public release. **Experimental**: tested against mock upstreams and synthetic benchmarks, not yet against
long-term real Claude Code / Codex traffic. See "Status & known limitations" in the README.

Highlights
- Local gateway for Anthropic Messages, OpenAI Responses (Codex) and Chat Completions, plus an MCP endpoint
- Cache-safe overlay replay and streaming interception of unimem_* tools (1060/1060 correctness assertions,
  42/42 prefix-stable transitions)
- v0.4: token budgets, cross-turn de-dup ledger, compaction-aware re-injection, tool-error recall,
  pre-compaction flush, honest usage reporting
- Project detection from git root, remote and manifest; global vs project scopes
- Local web UI at /ui
- Benchmarks: own suites (out-of-sample heldout2) plus LongMemEval_S / LoCoMo retrieval

Full changelog: https://github.com/mapix/unimem/blob/main/CHANGELOG.md
EOF
```

To attach a macOS binary:

```sh
cargo build --release
tar -C target/release -czf unimem-v0.5.0-$(uname -m)-apple-darwin.tar.gz unimem
gh release upload v0.5.0 unimem-v0.5.0-$(uname -m)-apple-darwin.tar.gz
```

## 4. After pushing

- Check that the CI run on the Actions tab is green. Only the `onnx` job is allowed to fail (`continue-on-error`);
  `rustfmt` is enforced.
- Settings → Code security: enable **private vulnerability reporting** (`SECURITY.md` links to it), Dependabot
  alerts, and secret scanning.
- Settings → Branches: protect `main` and require the `clippy` and `test` checks.
- Optional: `cargo publish --dry-run`, before publishing to crates.io.
