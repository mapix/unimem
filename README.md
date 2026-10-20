# unimem

**One shared, project-aware long-term memory for every coding agent.**

unimem is a single local binary that sits between your coding harnesses (Claude Code, Codex CLI, anything
OpenAI- or Anthropic-compatible) and the model API. Point each harness's base URL at it, and they all share the same
memory: the correction you gave Claude Code this morning is known to Codex this afternoon, filed under the right
project.

Design rationale and internals: [docs/DESIGN.md](docs/DESIGN.md) (in Chinese).

## Why

Every harness now ships its own memory: Claude Code has `CLAUDE.md` files and auto-memory, Codex has `AGENTS.md`
and its own memories, other tools have their own rules files. Each one is a silo. They use different formats and
storage, they don't know about each other, and they mostly don't separate "about me" from "about this repo".
If you use more than one agent, you end up repeating yourself.

Memory belongs to the user, not to the tool. The only integration point every harness shares is the model API, and
every harness lets you change its base URL. So unimem lives there and needs no plugin for each harness.

## How it works

```
Claude Code ─┐ /v1/messages
Codex ───────┼ /v1/responses        ──►  unimem (127.0.0.1:4777)  ──►  api.anthropic.com / api.openai.com / chatgpt.com
other ───────┘ /v1/chat/completions        │
MCP client ──── /mcp                       └─ SQLite ~/.unimem/unimem.db
```

- **Gateway, transparent by default.** Requests are forwarded to the real upstream with the original auth headers
  (API keys and subscription OAuth). Helper requests, such as title generation, token counting and model lists,
  pass through untouched. If anything in the memory layer fails, the request goes through unmodified.
- **Tool interception.** On agent turns, unimem appends four tools to the request: `unimem_recall`,
  `unimem_remember`, `unimem_forget` and `unimem_project`. When the model calls one, the streaming rewriter holds
  back that call, runs it inside the gateway, and continues the generation upstream. The harness only sees one
  continuous response and never sees a tool it doesn't know. The model in the loop (System Two) writes the
  memories, because it understands what matters.
- **Cache-friendly overlay replay.** Every change unimem makes is stored as an *overlay*, keyed by content anchors
  that the harness sends back verbatim. Changes include the session preamble, passive recall blocks, and hidden tool
  calls with their results. Each later request in the session re-applies those changes byte-for-byte, so prompt-cache
  prefixes stay stable and the model remembers what it looked up. The system prompt is never modified.
- **Jev-Mem engine.** A memory graph with semantic, temporal, causal, entity and supersedes edges, searched with
  BM25 (FTS5) plus vectors fused by RRF. Many small decisions are made by a fast *System One* decider in batched
  typed questions, with thresholds in code. Examples: global or project scope? durable or transient? duplicate?
  supersedes an older note? relevant to this query? enough evidence? Backends: TypeSafe Jev, any small
  OpenAI-compatible LLM, or an offline heuristic. They form a fallback chain.
- **Project detection.** The gateway runs on your machine, so it reads the cwd the harness puts in the prompt and
  resolves git root → remote → manifest → README. Project ids come from the normalized remote
  (`github.com/me/repo`), so all clones and worktrees share one project. The agent can rewrite the project card with
  `unimem_project`.
- **Passive capture (optional).** Prompt turns are logged as episodes. Once a session goes idle, a distiller pulls
  out user corrections, preferences, decisions with reasons and gotchas, and runs them through the normal write
  path.

## Quickstart

```sh
cargo build --release
./target/release/unimem serve            # listens on 127.0.0.1:4777
```

### Claude Code

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:4777
claude
```

### Codex CLI

Add this to `~/.codex/config.toml`:

```toml
model_provider = "unimem"

[model_providers.unimem]
name = "unimem"
base_url = "http://127.0.0.1:4777/v1"
wire_api = "responses"
requires_openai_auth = true
```

### Other clients

```sh
export OPENAI_BASE_URL=http://127.0.0.1:4777/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:4777
```

`unimem setup claude-code|codex|shell|mcp` prints these snippets for your configured listen address.

## MCP fallback

If you'd rather not route model traffic through unimem, the same tools are served over MCP Streamable HTTP at
`/mcp`. In this mode you get no passive recall, preamble or episode capture. The agent has to call the tools itself.

```sh
claude mcp add --transport http unimem http://127.0.0.1:4777/mcp
```

```toml
# ~/.codex/config.toml
[mcp_servers.unimem]
url = "http://127.0.0.1:4777/mcp"
```

MCP tools take an extra `cwd` argument so the right project is used. You can also send it in the `x-unimem-cwd`
header.

## CLI

All commands except `serve` work directly on the database. They act as harness `cli`, with the project taken from
the current directory or `--cwd`.

| Command | |
|---|---|
| `unimem serve [--listen ADDR]` | run the gateway |
| `unimem remember "<note>" [--why] [--scope auto\|global\|project\|<name>] [--kind] [--pin]` | store a memory |
| `unimem recall "<query>" [--project <name>\|all] [-k N] [--json]` | search the current project + global memory |
| `unimem forget <id>` | delete a memory |
| `unimem list [--scope global\|project\|<name>] [--all] [-n N] [--json]` | list memories |
| `unimem projects` / `unimem project [--summary "…"] [--name …]` | known projects / show or edit the current card |
| `unimem stats` | counts per scope, episodes, decider chain, embedder |
| `unimem trace [-n N]` | recent engine decisions |
| `unimem reembed` | recompute embeddings after changing the embedder |
| `unimem distill [--now]` | run the distiller once |
| `unimem config` | config path + commented sample |
| `unimem setup <harness>` | integration snippets |
| `unimem ui [--no-open]` | print (and open) the web UI URL |
| `unimem demo-seed [--dir PATH]` | write ~45 demo memories across 3 throwaway projects + global |

The global option `--config PATH` (or `$UNIMEM_CONFIG`) selects the config file.

## Web UI

`unimem serve` also serves a local web UI at `http://127.0.0.1:4777/ui` (`unimem ui` prints and opens the URL).
It is one self-contained page embedded in the binary: no CDN, works offline, light by default with a dark mode.

- **Overview**: what is in memory, by kind, project and agent; the standing rules every session starts with; recent writes and engine activity.
- **Memories**: filter by scope, project, kind and agent, or search the way an agent would ask (the real recall ranking, plus exact text matches). Each kind is shown in its own form: preferences as rule chips grouped *everywhere* vs per project, decisions as ADR-style records (decision, because, date) with their version chain `v1 → v2 → current`, procedures as terminal blocks with copyable commands, gotchas as amber warnings, facts as notes with entity tags, episodes on a date rail. Superseded notes can be shown, struck through and linked to their successor.
- **Projects**: one board per repository with the editable project card (summary every agent reads, stack, aliases, root, remote) and its notes grouped by kind.
- **Graph**: the memory graph in a small force layout, edges coloured by type (supersedes, causal, same thing, same session, similar), click a node to open it.
- **Timeline**: notes by day, with the moments a decision or rule was replaced.
- **Recall playground**: type a prompt, pick the project (or a cwd) and see exactly what the gateway would inject: scores, how each note was found, whether the answer looks sufficient, the literal `<unimem-recall>` block and the session preamble.
- **Activity**: every write, recall and edit, plus the raw turns the distiller reads.

Click any note for its detail: content, reason, source (agent, session, cwd, branch, commit), entities, connected notes and versions, with pin / edit (re-embeds) / move scope / delete.

The JSON API behind it lives under `/ui/api/` (`overview`, `memories`, `memories/:id`, `projects`, `projects/:pid`,
`graph`, `timeline`, `traces`, `recall`). Like the rest of the gateway it is meant for `127.0.0.1` only and has no
authentication.

To try it on sample data without touching your memory:

```sh
export UNIMEM_DATA_DIR=/tmp/unimem-demo UNIMEM_LISTEN=127.0.0.1:4799
unimem demo-seed && unimem serve     # then open http://127.0.0.1:4799/ui
```

## Configuration

`~/.unimem/config.toml`. Every field is optional, and with no file unimem runs fully offline (heuristic decider,
hash embedder). `unimem config` prints this sample:

```toml
listen   = "127.0.0.1:4777"
data_dir = "~/.unimem"              # unimem.db lives here

[upstream]
anthropic    = "https://api.anthropic.com"               # /v1/messages
openai       = "https://api.openai.com"                  # Responses / Chat with an API key
chatgpt      = "https://chatgpt.com/backend-api/codex"   # Codex signed in with ChatGPT
timeout_secs = 600

[inject]
enabled           = true    # false = pure pass-through recorder
tools             = true    # expose unimem_recall / remember / forget / project
recall            = true    # passive recall appended to each new user prompt
preamble          = true    # project card + standing notes on the first user message
recall_budget_ms  = 1500    # passive recall is skipped if slower than this
max_recall_notes  = 5
preamble_max_notes = 8
max_continuations = 6       # internal tool rounds per visible response
record_episodes   = true    # keep a raw episode log (input for the distiller)

# System One: the fast typed-decision model. Backends are tried in order;
# unavailable ones fall through. "heuristic" is offline and always appended.
[decider]
chain = ["jev", "llm", "heuristic"]

[decider.jev]                      # TypeSafe Jev /v1/systemone (or a compatible local server)
base_url    = "https://api.typesafe.ai"
api_key_env = "TYPESAFE_API_KEY"   # used only if set
timeout_ms  = 8000

[decider.llm]                      # any OpenAI-compatible chat model, e.g. Ollama
base_url    = ""                   # "http://127.0.0.1:11434/v1"
model       = ""                   # "qwen2.5:7b-instruct"
api_key_env = "UNIMEM_LLM_API_KEY"
timeout_ms  = 15000

[embedder]
kind  = "hash"                     # "hash" (offline) or "openai" (any /v1/embeddings)
# kind     = "openai"
# base_url = "http://127.0.0.1:11434/v1"   # Ollama
# model    = "bge-m3"
# dim      = 1024
api_key_env = "UNIMEM_EMBED_API_KEY"
dim   = 512
# after changing the embedder run: unimem reembed

[distill]                          # passive capture: extract memories from idle sessions
enabled     = false
base_url    = ""                   # OpenAI-compatible; empty model + enabled = heuristic extractor
model       = ""
api_key_env = "UNIMEM_DISTILL_API_KEY"
idle_secs   = 600
```

Environment overrides: `UNIMEM_CONFIG`, `UNIMEM_DATA_DIR`, `UNIMEM_LISTEN`. Logging: `RUST_LOG=unimem=debug`.

## Data

Everything lives in one SQLite file, `~/.unimem/unimem.db`. It holds memories, edges, FTS index, embeddings,
projects, sessions, the episode log, overlays (garbage-collected after 30 days) and decision traces. Nothing leaves
your machine except:

- the traffic you were already sending to your model provider;
- the decider, embedder and distiller endpoints, and only if you configure them.

Memories are screened for secrets before they are written.

## Limitations

- Pass-through of Codex's custom-provider auth with a ChatGPT login still needs testing on more Codex versions.
- Rewriting subscription traffic may be subject to your provider's terms. Check them yourself.
- Mixed turns (our tools plus the harness's tools in one response) depend on anchor matching. If the harness
  compacts or edits history, an overlay may be skipped. The request is still valid, but that context is lost.
- The heuristic decider is a zero-config baseline. Decisions get much better with Jev or a small LLM, and its
  thresholds still need tuning on real data.
- The MCP endpoint replies with plain JSON only. It has no server-initiated SSE stream (`GET /mcp` returns 405).

## License

MIT
