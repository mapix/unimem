# unimem

**One shared, project-aware long-term memory for every coding agent. It runs as a local gateway, so no harness needs a plugin.**

[![CI](https://github.com/mapix/unimem/actions/workflows/ci.yml/badge.svg)](https://github.com/mapix/unimem/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)

unimem is a single Rust binary that sits between your coding harnesses (Claude Code, Codex CLI, or any
Anthropic- or OpenAI-compatible client) and the model API. You point each harness's base URL at it, and from then on
they all share one memory. A correction you gave Claude Code in the morning is known to Codex in the afternoon, and it
is filed under the right repository.

> [!WARNING]
> **unimem is research-grade software (v0.4).** It has been tested against mock upstreams and synthetic benchmarks,
> **not against real Claude Code / Codex traffic over long periods.** Read
> [Status & known limitations](#status--known-limitations) before routing your daily work through it.

![unimem web UI: overview](docs/images/ui-overview.png)

---

## Contents

- [Why](#why)
- [What it does](#what-it-does)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Web UI](#web-ui)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [Data & privacy](#data--privacy)
- [Status & known limitations](#status--known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing) · [Security](#security) · [License](#license) · [Citation](#citation)

## Why

Every harness now ships its own memory. Claude Code has `CLAUDE.md` and auto memory, Codex has `AGENTS.md` and its own
memories, and other tools keep their own rules files. Each of these is a silo: the formats and storage differ, none of
them can see the others, and most don't separate "about me" from "about this repo". If you use more than one agent,
you end up repeating yourself.

**Memory belongs to the user, not to the tool.** The one integration point that every harness shares is the model
API, and every harness lets you change its base URL. unimem sits at that point, so it works with each harness without
a per-harness integration.

## What it does

- **One memory across harnesses.** Claude Code (`/v1/messages`), Codex CLI (`/v1/responses`, API key or ChatGPT
  login) and Chat Completions clients all read and write the same store.
- **Project-aware by construction.** Memories are scoped as *global* (about you) or *project* (about one repo).
  Projects are detected from the agent's working directory: git root, normalized remote, manifest, README. All
  clones and worktrees of a repo share one project.
- **The model in the loop writes memories.** unimem adds four tools (`unimem_recall`, `unimem_remember`,
  `unimem_forget`, `unimem_project`) and runs them *inside the gateway*. The harness never sees a tool it doesn't
  know about.
- **Passive recall and a session preamble.** On each new user prompt, unimem appends the few notes that matter. The
  first message of a session gets the project card and your standing rules.
- **Prompt-cache safe.** Every change is replayed byte-for-byte on later requests. The system prompt is never touched.
  In the benchmark, 42/42 upstream transitions are exact prefix extensions.
- **Bounded footprint (v0.4).** Token budgets per turn and per session, a cross-turn de-dup ledger,
  compaction-aware re-injection, and honest `usage` reporting.
- **Temporal correctness.** New decisions *supersede* old ones instead of piling up beside them. Stale notes are
  demoted, or flagged when they are returned.
- **Fail-open.** If anything in the memory layer fails or is slow, the request is passed through unchanged.
- **Local-first.** One SQLite file. It runs fully offline by default (heuristic decider, hash embedder); smarter
  backends are opt-in.
- **Also an MCP server** (`/mcp`), a CLI, and a **local web UI** (`/ui`).

## Quickstart

Requirements: a recent stable Rust toolchain (edition 2021). SQLite is bundled, so nothing else is needed.

```sh
git clone https://github.com/mapix/unimem && cd unimem
cargo build --release
./target/release/unimem serve            # listens on 127.0.0.1:4777
```

Optionally put it on your `PATH` with `cargo install --path .`.

Leave `unimem serve` running, then connect a harness. `unimem setup <harness>` prints the exact snippet for your
configured listen address.

### Claude Code

```sh
unimem setup claude-code
export ANTHROPIC_BASE_URL=http://127.0.0.1:4777     # or put it under "env" in ~/.claude/settings.json
claude
```

API keys and subscription OAuth headers are both forwarded to the upstream unchanged.

### Codex CLI

```sh
unimem setup codex
```

```toml
# ~/.codex/config.toml
model_provider = "unimem"

[model_providers.unimem]
name = "unimem"
base_url = "http://127.0.0.1:4777/v1"
wire_api = "responses"
requires_openai_auth = true   # keeps your ChatGPT login / OPENAI_API_KEY
```

### Any other client

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:4777
export OPENAI_BASE_URL=http://127.0.0.1:4777/v1
```

### MCP only (no gateway)

If you don't want model traffic going through unimem, use the MCP endpoint instead. You get the four tools, but no
passive recall, no preamble and no episode capture, so the agent has to call the tools itself.

```sh
unimem setup mcp
claude mcp add --transport http unimem http://127.0.0.1:4777/mcp
```

```toml
# ~/.codex/config.toml
[mcp_servers.unimem]
url = "http://127.0.0.1:4777/mcp"
```

MCP tools take an extra `cwd` argument (or an `x-unimem-cwd` header) so the right project is used.

### Look inside

```sh
unimem ui                 # opens http://127.0.0.1:4777/ui
unimem recall "how do we run migrations?"
unimem list --scope project
```

To try everything on throwaway data first:

```sh
export UNIMEM_DATA_DIR=/tmp/unimem-demo UNIMEM_LISTEN=127.0.0.1:4799
unimem demo-seed && unimem serve     # then open http://127.0.0.1:4799/ui
```

## How it works

```mermaid
flowchart LR
    CC[Claude Code] -- /v1/messages --> GW
    CX[Codex CLI] -- /v1/responses --> GW
    OT[other clients] -- /v1/chat/completions --> GW
    MCP[MCP client] -- /mcp --> GW
    subgraph GW[unimem gateway · 127.0.0.1:4777]
        direction TB
        T[turn classifier] --> O[overlay replay]
        O --> B[budgets · de-dup ledger · compaction detector]
        B --> I[tool + recall injection]
        I --> S[streaming rewriter<br/>intercepts unimem_* calls]
        S <--> E[(Jev-Mem engine<br/>SQLite · FTS5 · vectors · graph)]
        P[project detector<br/>git root / remote / manifest] --> E
    end
    GW -- original auth headers --> UP[api.anthropic.com<br/>api.openai.com<br/>chatgpt.com/backend-api/codex]
    UI[web UI /ui] --- E
```

```
Claude Code ─┐ /v1/messages
Codex ───────┼ /v1/responses        ──►  unimem (127.0.0.1:4777)  ──►  Anthropic / OpenAI / ChatGPT
other ───────┘ /v1/chat/completions        │
MCP client ──── /mcp                       └─ SQLite ~/.unimem/unimem.db
```

**Gateway and turn classification.** Not every request should get memory. Requests that carry tools are *agent turns*.
Among them, a new user message is a *prompt turn* (passive recall) and a tool result is a *tool turn* (recall only if
the output looks like an error). Helper requests pass through untouched: titles, token-prefix checks, quota probes,
model lists, and the harness's own **compaction** request. `count_tokens` gets the same rewrite as the real request,
so the harness's estimate stays accurate.

**Overlay replay (cache-friendly injection).** Every change unimem makes is stored as an *overlay*, keyed by a content
anchor that the harness sends back verbatim. Changes include the preamble, passive recall blocks, and hidden tool
calls with their results. Each later request in the session re-applies those changes byte-for-byte. This has two
effects: prompt-cache prefixes stay stable, and the model still "remembers" what it looked up several turns ago. The
system prompt is never modified, and tools are appended after the harness's own.

**Streaming interception.** When the model calls a `unimem_*` tool, the streaming rewriter holds back that call, runs
it inside the gateway, and continues the generation upstream (up to `max_continuations` rounds). The harness receives
one continuous, well-formed response. Mixed turns, where unimem tools appear next to harness tools like `Bash`, are
split: unimem's calls are hidden and replayed later, and the harness's calls are delivered. This works for all three
dialects (Anthropic Messages, OpenAI Responses, Chat Completions), streaming and JSON.

**Budgets and compaction awareness (v0.4).** Injected memory competes with your task for context, so it is capped:

| injection | cap |
|---|---|
| passive recall + hints, per message | min(1200 tokens, 0.6% of the window) |
| session preamble | min(1500 tokens, 1% of the window) |
| one `unimem_recall` tool result | 3000 tokens, k ≤ 8 |
| tool-error recall | ≤ 2 notes, ≤ 400 tokens, at most once per 5 tool turns |
| everything unimem keeps live in one conversation | 5% of the window; passive recall stops beyond it |

A **de-dup ledger** prevents re-injecting a note that is still visible in the conversation. **Compaction** is detected
by four signals: the message count drops, the first user message changes, a summary marker appears, or the session had
overlays and none of them matched. When it happens, the ledger is reset and the preamble is sent again. At 70% of the
window, a **pre-compaction flush** distills the session in the background and nudges the agent once to save durable
facts. Merged responses report the *last* round's `input_tokens` (the real context size), so the harness's compaction
heuristics are not fooled. True totals are in the `x-unimem-usage` header.

**Jev-Mem engine.** The engine follows the System One / System Two split of
[Jev-Mem](https://arxiv.org/abs/2609.23986). The in-loop model (System Two) writes and reasons. Many small typed
decisions go to a fast *System One* decider in batched questions, with thresholds in code. Examples: global or
project? durable or transient? duplicate? supersedes an older note? relevant? is the evidence sufficient? Decider
backends form a fallback chain: TypeSafe Jev, any small OpenAI-compatible LLM (for example via Ollama), and an offline
heuristic. Memories form a graph with semantic, temporal, causal, entity and supersedes edges. Search uses BM25 (FTS5)
plus vectors, fused by RRF, then a calibrated relevance prior and an adaptive cut-off. Writes are screened for
secrets, transient status and prompt injection (see the limitations below).

**Project detection.** The gateway runs on your machine, so it can read the cwd the harness mentions in its prompt. It
resolves cwd → git root → remote → manifest (`Cargo.toml`, `package.json`, `pyproject.toml`) → README. Project ids
come from the normalized remote (`github.com/me/repo`), so all clones and worktrees share one project. The agent can
rewrite the project card with `unimem_project`.

**Passive capture (optional).** Prompt turns are logged as episodes, with secrets redacted and a 30-day TTL. A
distiller pulls out user corrections, preferences, decisions with their reasons, and gotchas. It runs when a session
goes idle, when a new session starts, or at the pre-compaction flush, and it skips its run when the rate limit is low.
Its output goes through the normal write path, and it is **off by default**.

Full design rationale, in Chinese: [docs/DESIGN.md](docs/DESIGN.md). Design review against frontier memory systems:
[docs/REVIEW.md](docs/REVIEW.md).

## Web UI

`unimem serve` also serves a local web UI at `http://127.0.0.1:4777/ui`. It is one self-contained page embedded in the
binary, with no CDN, so it works offline. It is light by default and has a dark mode.

| | |
|---|---|
| ![Memories](docs/images/ui-memories.png) | ![Projects](docs/images/ui-projects.png) |
| **Memories.** Filter by scope, project, kind and agent, or search the way an agent would ask. Each kind has its own view: preferences as rule chips, decisions as ADR-style records with their `v1 → v2 → current` chain, procedures as copyable terminal blocks, gotchas as warnings. | **Projects.** One board per repository: the editable project card that every agent reads (summary, stack, aliases, root, remote), and the repo's notes grouped by kind. |
| ![Graph](docs/images/ui-graph.png) | ![Recall playground](docs/images/ui-recall.png) |
| **Graph.** The memory graph, with edges coloured by type (supersedes, causal, same entity, same session, similar). | **Recall playground.** Type a prompt and see exactly what the gateway would inject: scores, how each note was found, whether the answer looks sufficient, and the literal `<unimem-recall>` block. |

There are also **Overview**, **Timeline** (notes by day, with the points where a rule was replaced) and **Activity**
(every write, recall and edit, plus the raw turns the distiller reads). Click any note for its detail, where you can
pin it, edit it (it is re-embedded), move its scope or delete it. The JSON API behind the UI lives under `/ui/api/`.
**It has no authentication** (see the limitations).

## Benchmarks

Every number below was produced **offline**: heuristic decider, CPU only, no LLM in the loop and no real harness
traffic. Treat them as engineering measurements, not as a claim about end-to-end agent quality.

### Standard benchmarks: LongMemEval_S and LoCoMo (retrieval only)

Full report: [bench/std/RESULTS.md](bench/std/RESULTS.md). Our BM25 re-implementation matches LongMemEval's official
code exactly (max per-question difference 1e-16).

| LongMemEval_S, 419 questions | session R@10 | turn R@10 |
|---|---|---|
| BM25 as in the LongMemEval paper (`rank_bm25`, whitespace tokens) | 0.823 | 0.592 |
| stemmed BM25 | 0.924 | 0.754 |
| multilingual-e5-small (118M) | 0.969 | 0.835 |
| BM25 + e5, RRF | **0.971** | **0.838** |
| unimem ranking (`relevance_prior`, e5) | 0.938 | 0.783 |
| unimem `recall` as shipped (returns ~2–3 notes) | 0.618 | 0.465 |

| LoCoMo (locomo10), evidence recall, categories 1–4 | R@5 | R@10 |
|---|---|---|
| stemmed BM25 | 0.470 | 0.553 |
| BM25 + e5, RRF | 0.551 | **0.629** |
| unimem ranking (e5) | 0.542 | 0.619 |

What the numbers say:

- **unimem's ranker is a solid lexical-hybrid ranker.** On LongMemEval it ties a well-tuned BM25, and on LoCoMo it is
  roughly level with BM25+e5 RRF.
- **It is significantly behind dense retrieval on paraphrase-heavy questions.** On LongMemEval it trails hybrid-rrf
  by −0.055 turn R@10 (95% CI [−0.079, −0.033]).
- **`recall` is not a top-k retriever.** It deliberately returns 1.7–3.5 notes for prompt injection, so its R@10
  is effectively an R@3. Cut every baseline at the same depth and it keeps up.
- Not done yet: end-to-end QA, which needs an LLM reader and judge (the script is ready), and LongMemEval_M.

### Own suites (`unimem-bench`)

Full reports: [bench/results/FINAL.md](bench/results/FINAL.md) (v0.1–v0.3) and
[bench/results/v0.4/SUMMARY.md](bench/results/v0.4/SUMMARY.md). E1 numbers are from the **out-of-sample** `heldout2`
set (242 queries, 233 notes, 6 repos, 3 harnesses).

| E1, memory quality | unimem v0.4 | top-5 cosine, same embedder, cwd-scoped | BM25 top-5, cwd-scoped | per-harness silo |
|---|---|---|---|---|
| R@1 / R@5 / MRR | **0.555** / **0.688** / **0.616** | 0.390 / 0.670 / 0.506 | 0.413 / 0.615 / 0.497 | 0.243 / 0.271 / 0.257 |
| precision (notes returned per query) | **0.423** (1.6) | 0.134 (5) | 0.152 (4.0) | 0.242 (1.1) |
| stale note returned | **16.3%** | 58.1% | 55.8% | 16.3% |
| cross-harness R@5 | **0.664** | 0.664 | 0.600 | 0.000 |

- The R@5 gain over a plain top-5 cosine search is **not significant** (+0.018, CI [−0.037, +0.073]). The real wins
  are in precision, R@1 and avoiding stale notes.
- **E2, gateway.** Added latency over a direct connection: +1.15 ms p50 / +3.1 ms p95 TTFB when streaming with recall,
  and +0.37 ms for JSON. All 1060/1060 interception assertions pass (the harness never sees a unimem tool or context,
  framing is valid, replay is byte-identical, the system prompt is untouched). Prompt cache: 42/42 fully
  prefix-stable transitions.
- **E5, long session (v0.4).** 120 requests with a forced compaction. The de-dup ledger cuts duplicate injections
  from 128 to 0. With heavy notes, peak unimem context goes from 7.0% to 3.5% of the window. The preamble is
  re-injected after compaction, and the pre-compaction flush saved 5 of 6 planted facts (0 of 6 without it).

Reproduce: `cargo run --release --bin unimem-bench -- all --data bench/data/heldout2.json --out /tmp/out` and
`CACHE=/path/to/cache bench/std/run_all.sh`.

## Configuration

The config file is `~/.unimem/config.toml` (or `--config` / `$UNIMEM_CONFIG`). **Every field is optional.** With no
file, unimem runs fully offline (heuristic decider, hash embedder). A fully commented example covering every option is
in [examples/config.toml](examples/config.toml), and `unimem config` prints the short version.

| section | key settings |
|---|---|
| top level | `listen` (`127.0.0.1:4777`), `data_dir` (`~/.unimem`) |
| `[upstream]` | `anthropic`, `openai`, `chatgpt` base URLs, `timeout_secs` |
| `[inject]` | `enabled` (false = pure pass-through), `tools`, `recall`, `preamble`, `recall_budget_ms`, `max_recall_notes`, `max_continuations`, `record_episodes`, budgets (`max_inject_tokens`, `max_preamble_tokens`, `max_tool_tokens`, `max_session_share`, `context_windows`), `dedup`, `tool_recall*`, `flush_at`, `correction_hints` |
| `[decider]` | `chain = ["jev", "llm", "heuristic"]`; `[decider.jev]` and `[decider.llm]` endpoints |
| `[embedder]` | `kind = "hash"` (offline), `"openai"` (any `/v1/embeddings`), or `"onnx"` (local, needs `--features onnx`); `model`, `dim`, `near_dup_cos`, `cos_baseline` |
| `[distill]` | `enabled` (off), `base_url`/`model` (empty = heuristic), `idle_secs`, `episode_ttl_days`, `min_rate_limit_remaining`, `chunk_chars` |

API keys are never written to the config. Each backend reads its key from the environment variable named in its
`api_key_env` (`TYPESAFE_API_KEY`, `UNIMEM_LLM_API_KEY`, `UNIMEM_EMBED_API_KEY`, `UNIMEM_DISTILL_API_KEY`).

Environment overrides: `UNIMEM_CONFIG`, `UNIMEM_DATA_DIR`, `UNIMEM_LISTEN`. Logging: `RUST_LOG=unimem=debug`.
After changing the embedder, run `unimem reembed`.

## CLI reference

Every command except `serve` works directly on the database. The project is taken from the current directory or
from `--cwd`.

| command | what it does |
|---|---|
| `unimem serve [--listen ADDR]` | run the gateway (Anthropic, Responses, Chat, MCP, UI) |
| `unimem setup claude-code\|codex\|shell\|mcp` | print integration snippets |
| `unimem ui [--no-open]` | print and open the web UI URL |
| `unimem remember "<note>" [--why …] [--scope auto\|global\|project\|<name>] [--kind …] [--pin]` | store a memory |
| `unimem recall "<query>" [--project <name>\|all] [-k N] [--json]` | search the current project + global |
| `unimem forget <id>` | delete a memory |
| `unimem list [--scope global\|project\|<name>] [--all] [-n N] [--json]` | list memories (`--all` includes superseded) |
| `unimem projects [--json]` | list known projects |
| `unimem project [--summary "…"] [--name …]` | show or edit the current project card |
| `unimem stats [--json]` | counts per scope, episodes, decider chain, embedder |
| `unimem trace [-n N]` | recent engine decisions |
| `unimem reembed` | recompute embeddings with the configured embedder |
| `unimem distill [--now]` | run the distiller once |
| `unimem config` | print the config path and a commented sample |
| `unimem demo-seed [--dir PATH]` | write ~45 demo memories across 3 throwaway projects + global |

Global option: `--config PATH`. Also built: `unimem-bench` and `unimem-std-bench`, the benchmark drivers.

## Data & privacy

- Everything lives in **one SQLite file**, `~/.unimem/unimem.db`. It holds memories, edges, the FTS index,
  embeddings, projects, sessions, the episode log, overlays and decision traces. Overlays, session state and
  episodes are garbage-collected after 30 days.
- **Nothing leaves your machine** except the traffic you were already sending to your model provider, plus the
  decider, embedder and distiller endpoints if you configure them. The defaults make no extra network calls. The
  `jev` decider is only used if `TYPESAFE_API_KEY` is set.
- Memories are screened for secrets before they are written, and episodes are redacted. The screens are regex- and
  heuristic-based, so don't rely on them as your only protection.
- unimem has **no telemetry**.
- To wipe everything, stop the server and delete `~/.unimem/`.

## Status & known limitations

unimem is **v0.4 and immature.** The architecture is well tested against mocks. Its behaviour in real daily use is
not yet known. Please read this list before you use it.

**Validation**

- **Not validated against real Claude Code / Codex traffic.** All gateway tests and benchmarks use mock upstreams
  and scripted, harness-like requests. Real harness versions may send shapes the anchors don't expect. One example is
  thinking blocks trimmed from history: an overlay is then skipped. The request stays valid, but that memory context
  is lost.
- Codex with a **ChatGPT login** through a custom provider, and user messages inserted after `function_call_output`
  (tool-error recall on Responses), have not been verified against the real backend.
- All quality numbers come from the **offline heuristic decider and the hash embedder**, on synthetic datasets
  written by the author. The Jev and LLM deciders have not been evaluated end to end.

**Context window and cost**

- **Context growth risk.** Injected memory, plus hidden tool rounds replayed for the life of a conversation, add up.
  v0.4 caps this with per-turn budgets, a 5% per-session share and a de-dup ledger. Before v0.4 the worst case could
  overflow a 200k window. Upstream bytes are still about 50–60% higher than pass-through in the 30-turn benchmark,
  because of continuation rounds.
- **Compaction detection is heuristic.** Two of its signals (summary markers and the compaction-request wording)
  depend on **the current text Claude Code and Codex use**. If a harness changes that wording, unimem may inject
  memory into compaction requests or miss a compaction. The fallback signals, message-count drop and zero anchor
  hits, don't depend on the harness.
- Context windows come from a built-in model table, and unknown models fall back to 128k. Override it with
  `inject.context_windows`.

**Safety**

- **The prompt-injection and status screens are weak out-of-sample.** They caught 1/7 injections and 1/6 status
  notes on held-out data, against 100% in-sample. v0.4 keeps instruction-like notes out of *passive* injection, but a
  poisoned note can still be stored and returned through `unimem_recall`. Treat shared memory as untrusted input.
- **There is no authentication** on the gateway, the MCP endpoint or the web UI. Anything that can reach the listen
  address can read and edit your memory. Keep it on `127.0.0.1` and never bind it to a public interface.
- **Single point of failure.** All model traffic goes through the gateway. If `unimem serve` is not running, the
  harness cannot reach its model. Memory-layer errors fail open, but a dead process does not, so unset the base URL
  to bypass it.
- **Terms of service.** Rewriting requests on a *subscription* plan (Claude Pro/Max OAuth, ChatGPT login) may be
  subject to your provider's terms. Check them yourself. The API-key mode is the conservative choice.

**Quality**

- **The default decider is a heuristic.** Its thresholds were tuned on the author's synthetic data. Kind
  classification is about 46% accurate, and the `sufficient` signal misfires on 45% of answerable queries.
- **Cross-language recall is weak.** A Chinese query against an English note (or the reverse) has R@5 ≈ 0.33,
  against 0.79 within one language. Cross-language duplicates are mostly not detected.
- Paraphrase (R@5 0.42) and cross-project questions (0.37) are weak. Graph expansion has no measurable effect yet.
- `recall` returns few notes on purpose, which hurts multi-evidence questions ("how many…", "in total").
- Recall is a brute-force cosine over all live notes, so it scales linearly: about 5 ms at 10k notes.
- The MCP endpoint only answers with plain JSON. It has no server-initiated SSE stream (`GET /mcp` returns 405).

## Roadmap

- Regression suite recorded from real Claude Code and Codex traffic, with anchor and compaction checks per harness
  version.
- Evaluate the Jev and small-LLM deciders, and refit `relevance_prior` per embedder.
- Stronger, model-backed injection and status screening, with a review queue for borderline notes.
- Cross-language recall (multilingual embedder by default when available) and cross-language dedup.
- A `k`-honouring mode for explicit recall, and a multi-evidence mode for aggregation questions.
- End-to-end QA on LongMemEval and LoCoMo with an LLM reader and judge.
- Optional auth token for the UI and API, and a `doctor` command that checks harness wiring.
- Consolidation pass (sleep-time merge and cleanup), and an ANN index for large stores.
- Prebuilt binaries and Homebrew formula.

## Contributing

Contributions are very welcome, especially bug reports from real harness traffic (with secrets redacted). See
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests, benchmarks and style. Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## Security

Please report vulnerabilities privately. See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 mapix

## Citation

The paper (English, covering v0.5) is at [docs/paper/unimem.pdf](docs/paper/unimem.pdf); the LaTeX source is
[docs/paper/unimem.tex](docs/paper/unimem.tex).

```bibtex
@techreport{unimem2026,
  title       = {unimem: A Harness-Agnostic Memory Gateway for Coding Agents},
  author      = {mapix},
  year        = {2026},
  institution = {GitHub},
  url         = {https://github.com/mapix/unimem},
  note        = {Technical report, v0.5}
}
```

unimem builds on the System One / System Two design of Jev-Mem (arXiv:2609.23986).
