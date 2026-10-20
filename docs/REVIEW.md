# unimem design review: when to write memory, when to read it

Reviewer's scope: the timing of memory writes (extraction) and reads (injection) in unimem v0.3 (`8a679a8`), compared with frontier systems, with an adversarial look at context-window, cache, concurrency and safety consequences. Code read: `src/gateway/{mod,anthropic,responses}.rs`, `src/engine.rs`, `src/distill.rs`, `src/tools.rs`, `src/config.rs`, `src/store.rs` (parts). Docs read: `docs/DESIGN.md`, `docs/paper.md` (v0.3), `bench/results/FINAL.md`, `bench/results/v0.3/ERRORS.md`. No code was changed.

Citation policy: every external claim is followed by the URL I actually fetched for this review (October 2026). Anything I state from background knowledge that I could not fetch is marked **[unverified]**.

---

## 0. Summary

unimem's retrieval engine and its cache-stable overlay mechanism are well built and well measured. Its **timing policy**, though, is the simplest possible one, and most of its risks come from there:

- **Read:** a session preamble, injected exactly once per session key, plus passive recall on every *prompt* turn. Nothing is injected on tool-result turns, nothing after compaction, and nothing is de-duplicated across turns. No token budget applies to any of it.
- **Write:** the in-loop model calls `unimem_remember` when the tool description persuades it to. That is the only write path on by default. The idle distiller is off by default and has not been evaluated. Episodes are recorded by default but nothing uses them. The system has no pre-compaction flush, no session-end trigger, no consolidation pass and no correction detection.

The highest-impact problems, in order:

1. **Injection is unbounded and grows monotonically.** Nothing caps augmentation tokens per turn or per session, and overlays are replayed for the life of the conversation. In a typical session the cost is small (about 110 tokens per prompt turn). In the worst case allowed by the code constants it exceeds a 200k window within about 60 prompt turns (§3.1).
2. **Compaction silently disables memory.** After a harness compacts, the anchors no longer match, so the replayed context disappears. But the `pre:<session>` meta flag says the preamble was already sent, so it is never sent again. Post-compaction sessions run with no project card and no standing preferences (§3.1.3).
3. **Usage is summed across hidden continuation rounds.** `input_tokens` is added up across rounds, so the harness can be told the context is 2 to 7 times larger than it is. This can trigger compaction early or confuse its context display (§3.1.4).
4. **Passive injection amplifies memory poisoning.** ERRORS.md shows 6 of 7 injection notes stored on the out-of-sample set, and n174 served as the answer to the exact question it targets. Passive recall puts such notes in front of the model even when it never asked (§3.4).
5. **The distiller loses data.** It marks *all* undistilled episodes as distilled, but it only reads the newest 24,000 characters. Everything older in a long session is silently discarded (§3.3).

The prioritized fixes are in §4. P0 items are about safety and correctness: budget, de-dup ledger, compaction re-injection, usage accounting, distiller data loss, and passive-path poisoning. P1 items add the missing triggers: tool-result recall, pre-compaction flush, session-end detection, mem0-style distiller ops and correction capture. P2 items add consolidation and longer-term quality work.

---

## 1. Survey: how frontier systems decide when to write and when to read

### 1.1 Harness-native memory

**Claude Code: CLAUDE.md plus auto memory.** Both are loaded at the start of every conversation. CLAUDE.md is written by the user. Auto memory is written by Claude itself "based on your corrections and preferences". Claude writes notes typed `user`, `feedback`, `project` and `reference`, and it skips anything derivable from the code or already in CLAUDE.md. "Claude doesn't save something every session." Only the first 200 lines or 25KB of `MEMORY.md` load at session start. Topic files are read on demand with the ordinary file tools. When the index nears its limit, Claude Code reminds Claude to shorten it, which acts as a write-side budget. Writes happen **mid-session through the agent's normal file tools** ("Claude reads and writes memory files during your session"), and the UI shows "Saved N memories" / "Recalled N memories". An explicit "remember X" goes to auto memory. Source: https://code.claude.com/docs/en/memory.md

**Claude Code compaction and hooks.** `PreCompact` (matchers `manual`/`auto`) can block compaction. `PostCompact` receives the `compact_summary`. `SessionStart` fires with source `compact` after compaction, and "Claude Code runs them and adds their output to the compacted context". After compaction, project-root CLAUDE.md and auto memory are **re-injected from disk**, the five most recently modified files are re-read, and "context that hooks added earlier" is *summarized with the rest of the conversation*. In other words, injected context does not survive compaction unless it is re-injected. `UserPromptSubmit` can add `additionalContext` (capped at 10,000 characters) alongside the prompt. `PostToolUse` can add context alongside a tool result. `SessionEnd` exists, with a shared 1.5 s budget. Sources: https://code.claude.com/docs/en/hooks.md, https://code.claude.com/docs/en/context-window.md. Through a gateway, Claude Code assumes the Anthropic API context window for recognized models, and auto-compaction sits at the 200K boundary or about 967K on 1M models: https://code.claude.com/docs/en/model-config.md

**OpenAI Codex local memories.** Off by default. Extraction runs **in the background on idle chats**: "Codex skips active or short-lived sessions … updates memories in the background instead of immediately at the end of every chat … waits until a chat has been idle long enough to avoid summarizing work that's still in progress." A background pass is skipped when the rate-limit remaining percentage falls below `memories.min_rate_limit_remaining_percent`. There are separate `extract_model` (per chat) and `consolidation_model` (global) settings. `disable_on_external_context` keeps chats that used MCP or web search out of memory generation. Per-chat `/memories` controls choose whether a chat *uses* memories and whether it *contributes* to them. Source: https://learn.chatgpt.com/docs/customization/memories.md

**ChatGPT memory.** The product page confirms that memories "carry useful context from earlier chats into future work" and should not replace `AGENTS.md`: https://learn.chatgpt.com/docs/personalize.md. The two-part mechanism is **[unverified]**: "saved memories" written by a model-side tool when the user states something durable or asks it to remember, plus "reference chat history", which retrieves from past chats implicitly on each turn. help.openai.com returned 403.

### 1.2 Research and libraries

**mem0 (arXiv 2504.19413).** Extraction is **per message pair** (m_{t-1}, m_t). The prompt also contains a rolling conversation summary S and the last m messages. The update phase compares each candidate fact with similar existing memories, and the LLM chooses **ADD / UPDATE / DELETE / NOOP** through a tool call. Source: https://arxiv.org/html/2504.19413v1

**MemGPT (arXiv 2310.08560).** The model manages its own memory through function calls. A queue manager inserts a **"memory pressure" warning** when prompt tokens exceed a warning threshold ("e.g. 70% of the context window"), so the model can save important content before eviction. At the flush threshold (100%), it evicts about 50% of the window and updates a **recursive summary** stored at the head of the FIFO queue. Evicted messages stay retrievable from recall storage. Source: https://arxiv.org/html/2310.08560

**Sleep-time compute (arXiv 2504.13171) and Letta "dreaming".** The paper has the model "think offline about contexts before queries are presented" and reports about 5x less test-time compute at equal accuracy, plus a case study on agentic SWE: https://arxiv.org/abs/2504.13171. Letta's current docs describe "dreaming": background subagents review recent conversations and update memory, triggered "**after a set number of completed agent steps or when the context window is compacted**", with an optional agent review before changes apply. Explicit `/remember` also exists. Source: https://docs.letta.com/guides/agents/sleep-time-agents (redirects to docs.letta.com/configuration/memory)

**Zep / Graphiti (arXiv 2501.13956).** Data is ingested as **episodes**. Edges are **bi-temporal**: t_valid/t_invalid on the event timeline, plus created/expired on the transaction timeline. When a new edge contradicts an overlapping old one, the system **invalidates** the old edge by setting its t_invalid instead of deleting it. Source: https://arxiv.org/html/2501.13956

**LangMem.** LangMem explicitly separates **"conscious" (hot-path) formation**, where the agent saves during the conversation (immediate, but "adds perceptible latency … one more obstacle to the agent's ability to satisfy the user"), from **"subconscious" formation**, where the system reflects "after it occurs (or after it has been inactive for some period)" for "higher recall". Source: https://langchain-ai.github.io/langmem/concepts/conceptual_guide/. `ReflectionExecutor` **debounces** processing, since per-message processing causes "redundant work … incomplete context when processing mid-conversation … unnecessary token consumption": https://langchain-ai.github.io/langmem/guides/delayed_processing/

**A-MEM (arXiv 2502.12110).** Every write builds a structured note and links it to related notes. A new write can also trigger **memory evolution**, which updates the attributes of existing notes. Read timing is not described in the abstract and is **[unverified]**. Source: https://arxiv.org/abs/2502.12110

**MemoryOS (arXiv 2506.06326).** Memory has three tiers. STM moves to MTM by dialogue-chain FIFO (the STM page queue holds 7 pages). MTM moves to long-term personal memory by **heat-based** promotion (threshold τ = 5). Retrieval at response time combines all tiers. Source: https://arxiv.org/html/2506.06326

**Generative Agents (arXiv 2304.03442).** Reflections are generated "when the sum of the importance scores for the latest events … exceeds a threshold (150 in our implementation)", which came to roughly 2 or 3 times a day. Source: https://arxiv.org/html/2304.03442

**LongMemEval (arXiv 2410.10813).** Read-side design is split into indexing, retrieval and reading. The paper finds that **session decomposition** (rounds as values), **fact-augmented key expansion**, **time-aware query expansion** (which narrows the search scope for temporal questions) and **Chain-of-Note** reading each improve results. It also finds that long-context LLMs lose about 30% accuracy on sustained-interaction memory, so "just keep everything in context" does not substitute. Source: https://arxiv.org/html/2410.10813

**MemoryBank (arXiv 2305.10250).** An Ebbinghaus-curve update rule forgets or reinforces a memory "based on time elapsed and the relative significance of the memory". Source: https://arxiv.org/html/2305.10250

**Jev-Mem (arXiv 2609.23986).** A System-One controller (fast, typed, probabilistic) governs typing, relations, routing, budget allocation, traversal, scoring and adaptive stopping. System Two is used only for reasoning and synthesis. Source: https://arxiv.org/abs/2609.23986

**jevmem (reference implementation).** The `SessionStart` hook injects pinned notes first, then the strongest conventions, gotchas, decisions and preferences. It skips notes that restate CLAUDE.md/AGENTS.md (embedding similarity ≥0.82, with a Jev call in the 0.70–0.82 band) and is boosted by recent git activity. The `UserPromptSubmit` hook asks **one Jev call whether the prompt needs memory at all**, runs lite recall only if it does, then applies a second subject filter. Superseded notes are never injected. Strongly stated preferences in a prompt are **auto-captured**. Consolidation runs **every 20 writes** and compares recent notes with their 3 nearest neighbours. Co-retrieved pairs are queued for judging. Sources: https://raw.githubusercontent.com/illescasDaniel/jev-mem/main/README.md, https://raw.githubusercontent.com/illescasDaniel/jev-mem/main/docs/architecture.md, https://raw.githubusercontent.com/illescasDaniel/jev-mem/main/docs/installation.md

### 1.3 2026 coding-agent memory tools and papers

- **memsearch** (zilliztech): capture happens in the **Stop hook after each turn**. A small model summarizes the turn, appends it to a dated markdown journal and indexes it. Recall is a 3-level progressive search (chunk, section, raw transcript). The DSH integration injects memories "before the first model step only when they are useful". Background `PROJECT.md`/`USER.md` maintenance refreshes "after a session only when the journals changed and a minimum interval has passed" and is off by default. Source: https://raw.githubusercontent.com/zilliztech/memsearch/main/README.md
- **OpenWolf** (cytostack): keeps a task checkpoint (objective, completed work, unresolved problems, next action) that "Supported Claude and Codex hooks can restore" after "compaction, a restart or a change of agent", and detects repeated reads of unchanged files. Source: https://raw.githubusercontent.com/cytostack/openwolf/main/README.md
- Recent arXiv work is directly relevant to unimem's weak points. "Relevance Is Not Sufficiency" (2610.09348) finds that relevance ranking does not close evidence gaps, which matches the weak `sufficient` signal. "Whose Credential Is It?" (2610.25707) covers cross-owner credential exposure in shared memory. "Many Steps into One" (2610.17544) shows that a contracted memory "quietly answers wrong" once its facts change. "Knowing the User Is Not Knowing What We Established" (2610.18684) argues that a remembered assistant proposal must not become an authorized instruction, which is the n174 failure mode. Only abstracts were fetched, via the arXiv API: https://export.arxiv.org/api/query?search_query=abs:%22agent%20memory%22

### 1.4 Comparison table

| System | Write trigger(s) | Who extracts | Read trigger(s) | Injection budget / placement | Update / forget |
|---|---|---|---|---|---|
| Claude Code CLAUDE.md | explicit user edit, `/init`, "add to CLAUDE.md" | user (agent on request) | session start; nested files on file access; **re-injected after compaction** | target <200 lines; user message after system | manual |
| Claude Code auto memory | **hot path**: agent writes files mid-session (corrections, preferences) | in-loop model | session start (index: first 200 lines/25KB); topic files on demand via tools; **re-injected after compaction** | hard index cap with reminders to shorten | agent edits/merges; `modified` timestamps |
| Claude Code hooks | any lifecycle event (`Stop`, `PreCompact`, `SessionEnd`, …) | hook script | `SessionStart` (incl. `compact`), `UserPromptSubmit`, `PostToolUse` | `additionalContext` ≤10,000 chars per hook | n/a |
| Codex memories | **idle-based background** per-chat extraction; skipped when rate limit low or external context used | separate extract model plus consolidation model | injected into future sessions (`use_memories`) | not documented | background consolidation |
| ChatGPT memory [unverified] | saved-memory tool when user states durable info; implicit chat-history reference | product model | every turn (saved memories); retrieval from history | not documented | user-managed |
| mem0 | **every message pair** | LLM extractor | per query retrieval | top-k | **ADD/UPDATE/DELETE/NOOP** |
| MemGPT | agent function calls; **memory-pressure warning at ~70%**; recursive summary at flush | in-loop model | agent-initiated search; working context always in prompt | OS-style tiers, explicit thresholds | self-editing |
| Letta dreaming | **every N steps or on compaction**, background | sleep-time subagents (optional review) | memory blocks in context | block sizes | background rewrite |
| Zep/Graphiti | every ingested episode | LLM extraction pipeline | per query search + rerank | top-k | **bi-temporal invalidation** |
| LangMem | hot-path tools **or** debounced background reflection | in-loop model or background manager | tool or prefetch | app-defined | manager extracts, updates, consolidates |
| A-MEM | every add → link + **evolve old notes** | LLM | per query [unverified] | top-k | evolution updates |
| MemoryOS | STM full → MTM (FIFO); **heat > τ** → long-term | system | each response, all tiers | top-m segments, top-k pages | heat-based replacement |
| Generative Agents | observation stream; **reflection when Σimportance > 150** | LLM | each action, by recency × importance × relevance | top-k | none (reflections accumulate) |
| MemoryBank | each interaction | LLM summary | each turn | top-k | **Ebbinghaus decay/reinforcement** |
| jevmem | hot-path tool; **auto-capture in prompt hook**; consolidation **every 20 writes** | in-loop model + System One | SessionStart (pinned + conventions, CLAUDE.md-deduped); **per-prompt gate + subject filter** | 1–3 notes per recall | supersede/duplicate flags, nothing deleted |
| memsearch | **Stop hook every turn** (small-model summary); post-session PROJECT/USER refresh | small model | before first model step when useful; progressive search via skill | 3-level progressive | journals + periodic summaries |
| OpenWolf | hook checkpoints (task state) | hooks | **restore after compaction/restart** | not documented | archive old notes |
| **unimem v0.3** | hot-path `unimem_remember` (default); idle distiller (**off**, unevaluated) | in-loop model; optional cheap model | preamble **once per session key**; passive recall on **prompt turns only**; agent tool | **none** (5 recall notes, 8 preamble notes, k≤20 tool; ≤2400 chars + uncapped `why` each) | write-time supersede/duplicate vs top-10; forget tool |

---

## 2. Taxonomy of triggers

### 2.1 Write triggers

| Trigger | Examples | Pros | Cons |
|---|---|---|---|
| **W1 Hot-path tool call by agent** | Claude Code auto memory, MemGPT, LangMem conscious, unimem `unimem_remember` | Highest-quality authoring (System Two knows the context); immediate availability; no extra model | Depends on compliance with the instruction; adds latency and a tool round to the user's turn; recall is low on implicit facts; competes with the task for attention |
| **W2 Per-turn extraction** | mem0 (message pair), memsearch (Stop hook) | High recall; deterministic coverage | Cost per turn; sees incomplete context mid-task; many redundant candidates; needs a strong update phase |
| **W3 Idle / session-end batch** | Codex (idle + rate-limit gate), LangMem background, unimem distiller | Full context of the episode; off the hot path; can be rate-limited | Delay; "idle" ≠ "ended"; must handle sessions that never go idle; quota contention |
| **W4 Pre-compaction flush** | MemGPT memory-pressure warning; Letta dreaming on compaction; Claude Code `PreCompact` hook | Captures exactly what is about to be lost | Needs a reliable context-size estimate; compaction is harness-controlled |
| **W5 Importance-threshold reflection** | Generative Agents (Σ>150), MemoryOS heat | Spends compute where signal is | Needs an importance scorer; threshold tuning |
| **W6 Sleep-time consolidation** | Letta dreaming, Codex consolidation model, jevmem every-20-writes | Fixes duplicates, stale facts, cross-note structure; cheap at idle | Can rewrite or destroy good memories; needs review/reversibility |
| **W7 Explicit user command** | "remember X", `/remember`, `/memories` | Highest precision; user intent | Low volume; users forget to do it |

### 2.2 Read triggers

| Trigger | Examples | Pros | Cons |
|---|---|---|---|
| **R1 Session-start preamble** | CLAUDE.md, auto-memory index, jevmem SessionStart, unimem preamble | Cache-friendly (stable prefix); standing rules always present | Static; ages over the session; lost on compaction unless re-sent; budget pressure |
| **R2 Per-prompt passive retrieval with gate** | jevmem UserPromptSubmit, ChatGPT chat-history reference [unverified], unimem passive recall | No dependence on agent compliance; cheap with System One | Prompt ≠ need (needs appear mid-task); duplicates across turns; hot-path latency; poisoning exposure |
| **R3 Agent-initiated tool** | MemGPT, Claude Code topic files, unimem `unimem_recall` | Precise, need-driven | Under-used; extra round trip; agent doesn't know what it doesn't know |
| **R4 Per-step / tool-result triggered** | Claude Code `PostToolUse` additionalContext, path-scoped rules loading on file access, memsearch pre-step injection | Catches mid-task needs (errors, files) | High frequency → needs strict gate + rate limit; cache churn |
| **R5 Post-compaction re-injection** | Claude Code (CLAUDE.md, auto memory, SessionStart `compact`), OpenWolf restore | Restores lost context exactly when needed | Requires detecting compaction |

Cross-cutting trade-offs:

- **Latency.** R2 and W1 sit on the hot path. W3, W5 and W6 are off it.
- **Cost.** W2 is the most expensive. W3 and W6 amortize.
- **Cache.** R1 is best. R2, R4 and R5 have to be tail-appended and replayed, which is what unimem does.
- **Precision.** R3 and W7 are best. R2 needs a gate.
- **Staleness.** R1 goes stale within a session. W6 is the only remedy for accumulated stale memory.

---

## 3. Critical review of unimem

unimem implements W1, an optional W3, R1, R2 and R3. It does not implement W2, W4, W5, W6, R4 or R5, and W7 only works through W1.

### 3.1 Context-window blow-up

#### 3.1.1 What is injected, from the code

- **Preamble** (`engine.rs::preamble`): about 250 chars of fixed text, plus the project line and summary (≤600 chars, `update_project`), plus up to `preamble_max_notes = 8` notes. Total ≈ 1 kchar + 8 × note.
- **Passive recall** (`gateway/mod.rs::build_augmentation`): about 170 chars of header and footer, plus up to `max_recall_notes = 5` notes.
- **Agent recall tool** (`tools.rs`): `k.clamp(1, 20)` notes, and up to `max_continuations = 6` rounds per visible response.
- **Note render** (`engine.rs::render_hits`): about 70 chars of overhead, plus `content` (≤2400 chars, the `write` cap), plus `" — because " + why`. **`why` has no length cap anywhere on the write path.**
- `unimem_project` returns up to 30 other projects at 120 chars each, about 4 kchar.

Tokenizer assumptions: English ≈ 4 chars/token, CJK ≈ 1–1.5 chars/token **[unverified, rough]**.

| Scenario | Per prompt turn | 100 prompt turns | vs 200k window |
|---|---|---|---|
| Typical (bench: mean note 92 chars, 1.6 notes per recall) | ≈ 430 chars ≈ **110 tok** | ≈ 11k tok + preamble ≈ 0.5k | 6% |
| Realistic heavy (notes of 400–600 chars, 3–5 hits, `why` present) | ≈ 2.5–3.5 kchar ≈ **700 tok** | ≈ 70k | 35% |
| Worst within constants, English (5 × 2.5 kchar) | ≈ 12.7 kchar ≈ **3.2k tok** | ≈ **320k** (preamble ≈ 5k) | **160%**, overflows by about turn 60 |
| Worst, CJK notes (2400 CJK chars ≈ 1.6–2.4k tok each) | ≈ **8–12k tok** | 0.8–1.2M | overflows by about turn 20 |
| Agent recall, k=20, 6 rounds in one visible turn | ≤ 120 × 2.5 kchar ≈ **75k tok** in one turn | n/a | 37% in a single turn |

In the typical regime this is not a problem. The issue is that **nothing bounds the tail**: the caps are on counts, not tokens, and `why` has no cap at all. The cost also accumulates, because every augmentation is stored as an `augment_user` overlay and replayed byte-for-byte on every later request until compaction. Hidden continuation rounds (`expand_assistant` / `expand_items`) are replayed the same way, including the full `tool_result` text of every `unimem_recall`. So each request's context is

```
harness_history + Σ_{prompt turns so far} aug_i + Σ_{hidden rounds so far} (tool_use + tool_result)_j
```

This sum only grows. Neither the harness nor unimem ever sees it as a budget.

#### 3.1.2 Duplicates across turns

`recall` is stateless with respect to the session. A project with about 50 notes in active use will return its 3–5 central notes over and over, each time it is relevant. Every hit is appended again and replayed forever. The same applies to preamble notes being recalled again on turn 1 and later. No ledger exists of the form "note n042 is already in live context". In a focused 100-turn session I would expect a large share of injected tokens to be repeats. The bench never measures this, so treat it as a hypothesis to measure (§4, P0-2).

#### 3.1.3 Interaction with harness compaction (correctness bug)

When Claude Code compacts (auto, or `/compact`), it replaces history with a summary. The documented behaviour is that hook-added context is "summarized with the rest" and only CLAUDE.md, auto memory and SessionStart(`compact`) output are re-injected (https://code.claude.com/docs/en/context-window.md). Inside unimem:

1. The old `augment_user` / `expand_*` anchors no longer match, so the replayed memory context disappears. That is good for size and bad for continuity.
2. `build_augmentation` checks `meta_get("pre:<session>")`. The flag was set on the first prompt, and the session id survives compaction (**[unverified]** for Claude Code, but the session id is how the gateway tracks the conversation). So **the preamble is never re-sent**. After the first compaction the agent loses the project card and the standing preferences, which are exactly the things Claude Code's own memory re-injects.
3. The same failure follows `/rewind`, editing the first prompt, and `--resume` after the 30-day overlay GC. The `pre:` flags themselves are never garbage-collected.
4. **[unverified, worth a trace]** If the harness's compaction request is sent with `tools` attached and ends in a user-text "summarize…" message, unimem classifies it as a prompt turn. It then runs passive recall on the summarization instruction, records that instruction as a user episode, and offers unimem tools to the summarizer.

#### 3.1.4 Usage accounting inflates apparent context

`anthropic.rs::sum_usage` and `responses.rs::add_usage` add *every* numeric field across rounds, including `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens`. A visible response that took 3 rounds over a 120k context reports about 360k input tokens. Harnesses that derive "context used" from the last response's usage, which I believe Claude Code does (**[unverified]**), will over-estimate. That leads to premature auto-compaction or a wrong `/context` display. Meanwhile `count_tokens` adds the tools but **not** the overlays (documented in `anthropic.rs`), so that path *under*-estimates. These two numbers are inconsistent, and neither is the true context size.

#### 3.1.5 Continuation inflation

Each hidden round is a full-context upstream request. E2 measured +60% upstream bytes in a 30-turn session with 6 continuations (paper Table 8). Those bytes are mostly cache reads, but they are still billed, rate-limited, and on subscriptions they draw down the user's quota invisibly.

### 3.2 Read timing

- **Only prompt turns trigger passive recall.** In agentic coding, a single user prompt is followed by tens of tool-result turns. The moments when memory matters most, such as an error message matching a stored gotcha ("MS5837 NACKs at 400 kHz"), a file path with a known convention, or a failing command whose correct invocation is in memory, all arrive in `tool_result` content. `prompt_text` explicitly excludes those turns. The only route left is the agent remembering to call `unimem_recall`.
- **The gate adds nothing measurable.** `unimem-passive` = `unimem-full` on all three datasets (FINAL.md §1.1). The heuristic gate prior (`screen::gate_prior`) is driven by length. Fewer than 6 info units gives 0.15, below `T_GATE = 0.35`, so the prompt is gated. 6–12 units give 0.3, gated unless phrased as a question. So short but memory-critical imperatives ("use the deploy script", "部署一下") are gated, and long pasted logs pass. The gate also cannot know what is already in context.
- **The preamble is sent once and frozen**, attached to whatever prompt turn the gateway first sees for that session key. With hash-fallback session keys (`h:hash(harness, first_user_text)`), two sessions that open with the same prompt share one `pre:` flag, so the second session gets no preamble.
- **No re-injection after compaction** (§3.1.3).
- **Latency budget.** Recall gets 1500 ms (`config.rs`; DESIGN.md still says 1.2 s) on the hot path before the first upstream byte. The heuristic decider uses milliseconds, but the Jev chain default has an 8 s timeout and the LLM decider 15 s, so the 1.5 s gate truncates them and the turn then gets no memory at all. Timeouts are logged but not counted, so the user cannot see how often memory was skipped.
- **No staleness or provenance in rendering.** Superseded notes can be injected (`superseded` is rendered as a tag) on the passive path. jevmem never injects superseded notes passively.

### 3.3 Write timing

- **Default write path is W1 only.** The *only* instruction is the `unimem_remember` tool description, and `system` is deliberately untouched. Claude Code's auto memory has harness-level system-prompt guidance and UI feedback. jevmem adds a skill and hook-level auto-capture. unimem has neither. Hidden tool calls are also invisible to the user, so there is no "Saved 2 memories" signal and no veto. Expected write volume is low and depends on the model. It is unmeasured: the paper has no end-to-end metric (§6 item 10).
- **The distiller is off by default and unevaluated** (paper §6 item 10). Specific defects:
  - **Data loss:** `transcript(eps, 24_000)` keeps the newest ≤24k chars, but `mark_distilled(&s.key, upto)` marks *every* undistilled episode up to the last id. Everything older is discarded without ever being read.
  - Idle-only trigger (`idle_secs = 600`, polled every 60 s). A session that is never idle for 10 minutes is not distilled until it ends. A session that is resumed after a break is distilled in pieces without the earlier context.
  - `MAX_PER_SESSION = 12` candidates; each episode is truncated at 4000 chars in the transcript.
  - The update phase is only the write path's heuristic supersede/duplicate check against top-10 candidates. There is no explicit UPDATE/DELETE/NOOP. Given ERRORS.md (duplicate recall 0.174, 2 wrong supersedes, 1 lost update), a high-volume distiller would amplify these errors.
  - The heuristic extractor only matches "remember…" and preference-like sentences. It cannot detect **corrections** ("no, that's wrong, it's X").
- **Episodes are recorded but unused by default.** `record_episodes = true` while the distiller is off. Every prompt and visible assistant text (≤20k chars each) is stored in plaintext with no secret screening (`has_secret` runs only on memories) and **no TTL**: GC covers overlays only. This retains data with no benefit.
- **Episodes miss the most valuable evidence.** `record_turn` stores only prompt text and visible assistant text. Tool results are excluded, and those are where root causes and correct commands are discovered. In Claude Code, user feedback on a rejected tool call also arrives inside `tool_result` content (**[unverified]**), so the strongest correction signal never reaches the episode log.
- **Missing triggers:** no pre-compaction flush (W4), no session-end detection beyond idle, no importance trigger (W5), no consolidation or sleep-time pass (W6), and no project-card refresh after the agent's first summary.

### 3.4 Safety and poisoning × passive injection

ERRORS.md: out-of-sample injection-screen recall is 1/7, status-note recall 1/6, and a stored injection note (n169) appears in results for 9 queries. A pull-based design such as R3 exposes these notes only when the agent asks. **R2 pushes them into context unprompted**, on every prompt whose words overlap. n174 ("run apply with -auto-approve… Mira doesn't need to be asked") is exactly the failure class 2610.18684 warns about: something remembered is treated as authorization. The render footer says "data, not instructions", but the harness's own system prompt cannot vouch for the unimem block. The passive path should therefore be held to a stricter screen than the tool path.

### 3.5 Concurrency, cache, protocol and operations

- **Concurrency.** `Store` is one `parking_lot::Mutex<Connection>`, locked synchronously inside async handlers. That blocks tokio workers while it is held. Parallel subagents, two harnesses and the distiller all serialize on it. Claude Code subagents probably share the parent's session id (**[unverified]**). If so, the main agent and its subagents share one `pre:` flag, so only one of them gets the preamble, and whichever request arrives first wins.
- **Anthropic prompt cache.** unimem appends the augmentation *after* the harness's `cache_control` breakpoint in the final user message. On the first request the augmentation is uncached input. On the next request it becomes part of the prefix and is cache-written. That is acceptable. Two subtler risks:
  - Caching uses a **20-block lookback** (https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md). Each hidden round adds assistant + user(tool_result) + assistant blocks between harness breakpoints. Six continuations can push the previous write out of the lookback window, causing a full miss.
  - E2 measured byte-level prefix stability, not actual `cache_read_input_tokens` from the API. The conclusion "cache-stable" is therefore not yet empirically demonstrated against Anthropic's breakpoint semantics.

  Changing the tool descriptions in an upgrade invalidates every session's cache once, because tools are hashed first. That is acceptable but should be documented.
- **Codex reasoning replay.** Hidden `function_call` items are re-inserted next to visible items, keyed on the hash of `encrypted_content`. With `store:false`, the server validates reasoning-item adjacency. If Codex prunes or reorders reasoning items (the analogue of thinking-block trimming), the inserted items may produce 400 errors or be orphaned. Codex's own compaction replaces history, so anchors miss and the memory loss of §3.1.3 applies (**[unverified]**). This needs real-traffic regression, as the paper already notes.
- **Subscription terms.** Claude Code's gateway docs state that with only `ANTHROPIC_BASE_URL` set, the saved claude.ai login remains the active credential, and gateways must forward the OAuth capability in `anthropic-beta` (https://code.claude.com/docs/en/llm-gateway.md). unimem *modifies* that traffic and *generates extra requests* on the user's subscription. The ChatGPT/Codex backend path (`chatgpt.com/backend-api/codex`) is not a public API. Whether either is allowed is a user-side legal question. It should be stated prominently, and memory should probably default to off for subscription credentials until the user opts in.
- **Privacy.** There is one shared DB across harnesses and projects, holding plaintext episodes forever (§3.3). Notes are scoped, but `unimem_project` lists every known project to any agent. That leaks cross-client project names in a consulting setting. Codex excludes chats with external context from memory generation by default-able config. unimem's distiller has no equivalent, even though web and MCP content is the main injection vector.
- **Failure modes.** Engine errors fail open. **Process death does not**: when unimem is down, every harness loses its upstream. The fail-open story does not cover the most likely operational failure. A continuation that fails mid-stream after visible bytes were sent ends the turn with `end_turn`, which is acceptable, but the user gets no signal.

---

## 4. Recommendations

Each item lists the mechanism, the expected effect and how to evaluate it. "Model table" means a static map from model id to context window and compaction threshold, overridable in config. Examples: Claude Sonnet/Opus 200k (1M for `[1m]` and newer models, per model-config docs), GPT-5-codex class ≈ 272–400k (**[unverified]**, make configurable).

### P0: correctness and safety

**P0-1. Token-budgeted injection with hard caps.**
- Estimate tokens per string. A cheap estimator is enough: ASCII chars / 4 plus CJK chars / 1.3, or `tiktoken-rs`/`o200k` for OpenAI.
- Apply the budgets at render time:
  - `aug_turn ≤ min(1200 tok, 0.6% of window)`;
  - preamble ≤ `min(2000 tok, 1% of window)`;
  - agent-tool output ≤ 3000 tok, k ≤ 8;
  - **cumulative per-session injected ≤ 5% of window**, tracked in a session ledger. Once exhausted, only notes with p ≥ 0.6 get through.
- Cap the rendered note at about 600 chars with "… (full: unimem_recall id=n042)". Cap `why` at 400 chars at write time.
- Expected effect: the worst case of §3.1.1 drops from 320k–1.2M to ≤10k per 100 turns on a 200k model, with no change for typical notes.
- Eval: a synthetic 100- and 300-turn session generator with long and CJK notes. Report injected tokens per turn, cumulative tokens and p99. Check that heldout2 R@5/MRR stay within CI (truncation should not change ranking).

**P0-2. Cross-turn de-dup ledger.**
- Add a table `injected(session, note_id, anchor, turn, tokens)`.
- Before rendering, drop notes whose `anchor` is still present in the current request's history (computed from the replay keys already available).
- When a note is dropped this way, optionally emit "(already in context: n042)" for ≤3 ids. That costs about 10 tokens and lets the model refer back.
- If the anchor is gone (compacted or rewound), the note becomes eligible again.
- Expected effect: removes repeat injections, plausibly the majority of passive tokens in focused sessions.
- Eval: duplicate-injection rate and tokens saved on replayed real traces. Add a bench assertion that no note id appears twice in live context.

**P0-3. Compaction detection and re-injection (R5).**
- Store `last_msg_count`, `first_user_key` and `aug_anchor_hits` per session.
- Flag compaction when any of these holds:
  - the message count drops by more than 30%;
  - `first_user_key` changes within the same session id;
  - a known summary marker appears in the first user message (Claude Code's "continued from a previous conversation" preamble **[unverified]**, Codex compacted items **[unverified]**);
  - the anchor hit rate for previously augmented messages falls to 0.
- On detection, clear `pre:` and re-send the preamble plus the top 3 notes from the ledger's working set on the **next request**, prompt or tool-result turn alike, appended to the final user message and recorded as an overlay as usual.
- Detect the compaction *request* itself, either by its signature or because it has the same session but an unusual final instruction. On that request, skip recall and tool injection. Optionally append "preserve these memory ids: …" to it.
- Also GC `pre:` keys together with the overlays.
- Expected effect: memory persists across compactions, matching what Claude Code does for CLAUDE.md and auto memory.
- Eval: scripted Claude Code and Codex traces with `/compact` and auto-compaction. Assert that the preamble is present within one request after compaction and that the compaction request is unmodified. Then run a real-traffic replay.

**P0-4. Honest usage accounting.**
- In the merged usage, report the **last round's** `input_tokens`/`cache_*` (the true context size) and the **sum** of `output_tokens`.
- Expose real totals in an `x-unimem-usage` response header and in `unimem stats`, so billing stays transparent.
- Make `count_tokens` replay overlays so the harness's estimate matches the request actually sent.
- Expected effect: the harness's compaction threshold and context display become correct.
- Eval: E2 with continuations. Compare the usage reported to the client with the real upstream request size. On a real Claude Code session, check `/context` against the gateway logs.

**P0-5. Distiller data-loss fix and episode hygiene.**
- Distill in chunks of ≤24k chars and mark only the episodes actually processed.
- Run `has_secret` redaction before `add_episode`.
- Add an episode TTL (for example 30 days, or delete after distillation). Make `record_episodes` default to `distill.enabled`.
- Also record compact tool-result snippets: error lines and the tail of failing commands, at most 2k chars per turn. These are the input for gotcha extraction.
- Expected effect: no silent loss; less sensitive data retained.
- Eval: unit test with a 100k-char session (all episodes are processed). Secret-in-prompt test (nothing persisted).

**P0-6. Stricter passive path (anti-poisoning).**
- Passive recall and the preamble inject only notes that meet all of these:
  - `injection p < 0.2` at write time (stored), which is stricter than the 0.6 used to reject;
  - not superseded;
  - not imperative-to-agent ("you should", "do not ask", "auto-approve") according to a dedicated screen.
- Notes written by the distiller from sessions with external context (web or MCP tool results) are excluded from the passive path (Codex's `disable_on_external_context` idea).
- Notes containing authorization-like content are rendered with provenance (harness, date, who wrote it).
- Expected effect: n169 and n174 stop appearing unprompted.
- Eval: heldout2 "rejected-note returned" on the passive path (6.6% now; target ≤1%), plus a new adversarial set of at least 50 injection notes not seen during tuning.

### P1: missing triggers

**P1-1. Tool-result-triggered recall (R4) with a strict gate.**
- On tool-result turns, extract candidate cues:
  - error lines, matched by regexes for `error|Error:|FAILED|panic|Traceback|exit code [1-9]|E[0-9]{4}|✗`;
  - file paths and identifiers in the result, and the command string from the preceding `tool_use` input.
- Build the query from the error line, the path and the command. Require at least one rare token (high IDF in the memory lexicon) to match. Use `T_GATE_TOOL = 0.6` and keep ≤2 notes within ≤400 tokens.
- Rate-limit to at most one injection per 5 tool turns and one per distinct error signature. Respect the de-dup ledger (P0-2).
- Placement:
  - Anthropic: an extra `text` block after the `tool_result`s in the same user message.
  - Responses: append an `input_text` user message after the outputs (verify that Codex accepts this).
  - In both cases store the injection as an overlay, so it stays cache-stable.
- Expected effect: covers the mid-task needs that passive recall misses today, at bounded cost.
- Eval: a "mid-task" bench of synthetic traces where a stored gotcha matches an error that appears only in a tool output. Report hit rate, false-injection rate per 100 tool turns, added tokens and added latency (budget 300 ms).

**P1-2. Pre-compaction flush (W4) and memory-pressure nudge.**
- Track the estimated context per session from the last real usage (P0-4) against the model table.
- At **70%** (MemGPT's threshold), first enqueue an immediate background distillation of the session's undistilled episodes, off the hot path.
- Then, once per session, append a nudge to the next user message: "Context is nearly full. If you learned durable facts this session (corrections, decisions and why, gotchas, commands), save them now with unimem_remember." Keep it under 60 tokens.
- Optionally ship a Claude Code `PreCompact` hook snippet that POSTs to `/unimem/flush` (http hook) for exactness.
- Expected effect: facts are captured before they are summarized away.
- Eval: scripted long sessions with planted facts that occur before a forced compaction. Report the fraction of planted facts stored as memories, with and without the flush.

**P1-3. Session-end detection and debounce.**
- Keep idle-based extraction (as Codex does), plus a **"new session in the same pid/cwd"** signal that ends the previous session and distills it immediately.
- Accept an optional `SessionEnd`/`Stop` hook POST.
- Debounce: cancel and reschedule while the session is active, as LangMem's ReflectionExecutor does.
- Gate on quota. If the distiller uses the user's own credential, skip when the rate-limit headers show less than X% remaining (Codex's `min_rate_limit_remaining_percent`; the header names are **[unverified]**).
- Expected effect: timely extraction without paying for in-progress work.
- Eval: extraction latency distribution, plus duplicate candidates per session compared with a per-turn baseline.

**P1-4. mem0-style ADD/UPDATE/DELETE/NOOP update phase for distiller and tool writes.**
- For each candidate, retrieve the top 5 neighbours across languages, using the multilingual embedder once available.
- Ask one batched System-One `Choice` question (or the distill model) for the op:
  - UPDATE: write the new note and supersede the old, keeping the old note's id in `why`;
  - DELETE: supersede with a tombstone note;
  - NOOP: drop the candidate.
- Make the value-difference check token-aware ("large" vs "xlarge"), to fix the n186 lost update.
- Expected effect: duplicate recall (0.174) and supersede recall (0.640) improve, and the distiller can safely run at volume.
- Eval: heldout2 write metrics (duplicates, supersede P/R, lost updates), plus a labelled distillation set with known facts. Report extraction precision/recall and op accuracy.

**P1-5. Correction capture and write nudges.**
- On each prompt turn, run a cheap correction detector against the previous assistant turn: negations or contrast ("no,", "not X", "instead", "don't", "wrong", "不是", "别", "改成"), plus an overlap between the user's text and the assistant's last claim. Include tool-rejection feedback found in `tool_result`s.
- When it fires:
  - (a) append a ≤40-token hint to the augmentation: "The user corrected you. If this is durable, save it with unimem_remember (kind=preference/decision).";
  - (b) mark the episode `importance += 3` for P2-2.
- Expected effect: raises W1 compliance exactly where it matters most.
- Eval: on scripted correction traces, report the rate of `unimem_remember` calls after a correction and the precision of the resulting notes.

### P2: quality and operations

**P2-1. Sleep-time consolidation job (W6).**
- Runs nightly or after N writes, as jevmem does every 20.
- (a) Cluster near-duplicates per scope by cross-language embedding and mark them `duplicate_of`.
- (b) Re-judge supersede pairs among co-retrieved notes (queue pairs from the recall traces).
- (c) Refresh the project card from recent decision and procedure notes when `summary_by` is older than K days.
- (d) Decay salience Ebbinghaus-style, `salience · exp(-Δt/τ)` with reinforcement on hits. This feeds preamble ranking.
- All changes are flags or supersessions and can be reversed (`unimem undo`).
- Eval: heldout2 duplicate recall, stale-returned rate, preamble precision on a labelled "standing rules" set.

**P2-2. Importance-weighted extraction trigger (W5).**
- Accumulate an importance score per episode: corrections +3, decision words +2, an error→fix pair +2 (error in a tool result followed by success), a long prompt +1.
- When Σ ≥ θ (tune it; Generative Agents used 150 on a 1–10 scale), trigger extraction early instead of waiting for idle.
- Eval: extraction cost per stored memory and time-to-availability.

**P2-3. Preamble quality.**
- Skip notes that restate CLAUDE.md/AGENTS.md (embedding ≥0.82, as jevmem does).
- Rank by recent git activity (branch, touched paths).
- Exclude superseded notes and notes with injection-score p above a threshold. Respect P0-1's budget.
- Eval: preamble redundancy against the instruction files, and click-through (later recall hits on preamble notes).

**P2-4. Read-side techniques from LongMemEval.**
- Time-aware query expansion for history-type queries (`asks history` already detected), which narrows candidates by date.
- Fact-augmented keys, i.e. index notes with extracted entities and paraphrases, including translations for the cross-language gap (R@5 0.33 vs 0.79).
- Eval: heldout2 temporal and cross-language categories; LongMemEval retrieval subset (task #11).

**P2-5. Concurrency and operations.**
- Move SQLite to `spawn_blocking` with a WAL-mode connection pool (one writer, N readers).
- Ship a launchd/systemd keep-alive and a documented harness-side fallback (unset the base URL).
- Measure real Anthropic `cache_read_input_tokens` over a 30-turn session with continuations, to validate the cache claim against the 20-block lookback.
- Eval: E2 at concurrency 8 (p95 TTFB) and a real-API cache-hit ratio.

**P2-6. Transparency and control.**
- `unimem log --session` shows what was injected and saved per turn.
- A per-session opt-out marker (a `#nomem` prompt prefix, analogous to Codex `/memories`).
- A visible but non-intrusive save notice. One option: a one-line `[unimem: saved n042]` appended to the next augmentation, so the model can mention it.
- Default memory to off for subscription credentials until the user opts in (§3.5).

---

## 5. Suggested evaluation additions

The current bench (E1–E4) measures retrieval quality and gateway mechanics but **not timing**. To make the recommendations above falsifiable, add:

1. **E5 session simulator.** Synthetic multi-hundred-turn agent sessions: prompt turns, tool-result turns with planted errors, corrections, and a forced compaction at 70–90%. Metrics: injected tokens per turn and cumulative, duplicate injections, post-compaction preamble presence, mid-task gotcha hit rate, planted-fact capture rate, false injections.
2. **E6 real-traffic replay.** Recorded Claude Code and Codex sessions, including `/compact`, `/rewind`, subagents and resume. Assertions: valid protocol, reported usage equals real usage, no cross-session or cross-subagent leakage, cache-read ratio measured from the real API.
3. **E7 end-to-end task metric**, already listed in paper §7: repeated-correction count and task success with and without memory. This is the only metric that can justify the hot-path costs.

---

## Sources fetched for this review

- Claude Code memory: https://code.claude.com/docs/en/memory.md
- Claude Code hooks: https://code.claude.com/docs/en/hooks.md
- Claude Code context window and compaction: https://code.claude.com/docs/en/context-window.md
- Claude Code model config (auto-compact window, gateway windows): https://code.claude.com/docs/en/model-config.md
- Claude Code LLM gateway (subscriptions): https://code.claude.com/docs/en/llm-gateway.md
- Anthropic prompt caching (20-block lookback): https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md
- Codex / ChatGPT memories: https://learn.chatgpt.com/docs/customization/memories.md, https://learn.chatgpt.com/docs/personalize.md
- mem0: https://arxiv.org/abs/2504.19413, https://arxiv.org/html/2504.19413v1
- MemGPT: https://arxiv.org/html/2310.08560
- Sleep-time compute: https://arxiv.org/abs/2504.13171; Letta memory/dreaming: https://docs.letta.com/guides/agents/sleep-time-agents
- Zep/Graphiti: https://arxiv.org/html/2501.13956
- LangMem: https://langchain-ai.github.io/langmem/concepts/conceptual_guide/, https://langchain-ai.github.io/langmem/guides/delayed_processing/
- A-MEM: https://arxiv.org/abs/2502.12110
- MemoryOS: https://arxiv.org/html/2506.06326
- Generative Agents: https://arxiv.org/html/2304.03442
- LongMemEval: https://arxiv.org/html/2410.10813
- MemoryBank: https://arxiv.org/html/2305.10250
- Jev-Mem: https://arxiv.org/abs/2609.23986
- jevmem: https://raw.githubusercontent.com/illescasDaniel/jev-mem/main/README.md, …/docs/architecture.md, …/docs/installation.md
- memsearch: https://raw.githubusercontent.com/zilliztech/memsearch/main/README.md
- OpenWolf: https://raw.githubusercontent.com/cytostack/openwolf/main/README.md
- 2026 arXiv abstracts (2610.09348, 2610.25707, 2610.17544, 2610.18684) via https://export.arxiv.org/api/query
- Repository discovery: https://api.github.com/search/repositories?q=memory+claude-code+codex+in:description

Not fetchable (403): help.openai.com memory FAQ and openai.com memory announcement. The ChatGPT mechanism details above are therefore marked [unverified].
