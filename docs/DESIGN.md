# unimem 设计文档（v0.4）

> 一句话：一个跑在本机的 LLM 网关。所有 coding harness（Claude Code、Codex、任何 OpenAI/Anthropic 兼容客户端）只需把 base URL 指过来，就共享同一份、按项目自动归档的长期记忆。

## 0. 第一性原理

从问题本身出发，而不是从现有产品出发，可以推出几条约束：

1. **记忆属于人，不属于工具。** 同一个人在 Claude Code 里纠正过的事，Codex 也应该知道。因此记忆层必须位于所有 harness 之外，并且它们共用。
2. **唯一普适的接入点是模型 API。** 每个 harness 都有自己的插件、hook、MCP 体系，彼此不兼容；但每个 harness 都一定会调用模型 API，并且都支持改 base URL。网关是唯一"零适配"的位置。
3. **网关在本机，因此能看到文件系统。** harness 在 prompt 里会告诉模型 cwd，网关拿到 cwd 后可以直接读 git root、remote、README。项目识别的硬信号是确定性的，只有语义部分（这个项目做什么、这条记忆归全局还是项目）才需要模型判断。
4. **环路里已经有最聪明的模型（System Two）。** harness 背后的大模型本身就是最好的记忆撰写者、项目描述者。不需要再额外花钱请一个大模型来总结，只需要给它工具。
5. **高频小决策交给 System One。** 判断"这条要不要记、归哪个作用域、和哪条旧记忆冲突、这次提问需不需要记忆、证据够不够"，是大量的类型化小判断，应交给 Jev 这类快速、便宜、给出概率的决策模型，阈值写在代码里（Jev-Mem 思路）。
6. **不能让用户为记忆付出缓存代价。** Anthropic/OpenAI 的 prompt cache 是前缀匹配。任何注入要么在整个会话内逐字节稳定，要么在尾部且之后被原样重放。
7. **失败必须透明。** 网关挂在所有流量前面，记忆层任何异常都必须退化为纯透传，绝不能阻断 agent。

## 1. 总体架构

```
 Claude Code ─┐  /v1/messages            ┌──────────────── unimem (单二进制) ────────────────┐
 Codex ───────┼─ /v1/responses ─────────►│ Dialect 适配 ─► Turn 分类 ─► 上下文提取(cwd/会话)    │──► 上游
 其它客户端 ──┘  /v1/chat/completions     │      │              │            │                 │   (Anthropic /
                                          │      ▼              ▼            ▼                 │    OpenAI /
 MCP 客户端 ──── /mcp ──────────────────►│  Overlay 重放   工具注入    项目解析(git)          │    ChatGPT)
                                          │      │              │            │                 │
                                          │      └──── 流式改写器：拦截 unimem_* 工具调用 ──────│
                                          │                      │  内部执行 + 续写             │
                                          │                      ▼                              │
                                          │   Memory Engine (Jev-Mem)                           │
                                          │   ├─ System One: Decider trait (Jev / LLM / 启发式) │
                                          │   ├─ 共享记忆图: 语义/时间/因果/实体 四视图          │
                                          │   ├─ 索引: FTS5(BM25) + 向量 (RRF 融合)             │
                                          │   └─ SQLite 单文件 (~/.unimem/unimem.db)            │
                                          └─────────────────────────────────────────────────────┘
```

## 2. 网关：三个核心机制

### 2.1 Turn 分类

不是每个请求都应该注入记忆。harness 会发出大量辅助请求（标题生成、bash 前缀判定、压缩摘要、quota 探测）。规则如下：

- 请求带工具定义（`tools` 非空）：视为 **agent turn**，注入工具，执行 overlay 重放。
- 最后一条消息是用户新输入（不是纯 tool_result）：视为 **prompt turn**，额外执行被动召回。
- 最后一条消息是 tool_result（Responses：`*_output` 项；Chat：`role: tool`）：视为 **tool turn**，只在输出像报错时触发工具错误召回（见 2.5）。
- **压缩请求**（harness 让模型总结对话，如 Claude Code 的 "Your task is to create a detailed summary of the conversation so far"、Codex 的 compaction/handoff 指令）：原样透传，不加工具、不召回、不记 episode。
- `count_tokens`：加上我们的工具，并且**重放 overlay**，这样 harness 估算的就是真正发给上游的请求。
- 无工具的请求、模型列表等其它请求：纯透传。

### 2.2 缓存友好的注入：Overlay 重放

这是整个设计最关键的一点。网关对请求做的每一处修改都记录为 overlay，之后同一会话的每次请求都原样重放这些修改，从而保证：

- prompt cache 前缀逐字节稳定；
- 模型在后续轮次中"记得"自己查过什么、写过什么。

overlay 有两类，都以内容锚点为 key（锚点来自 harness 原样回传的内容）：

| 类型 | 锚点 | 作用 |
|---|---|---|
| `augment_user` | 会话 key + 该 user 消息的文本/tool_result id 哈希 | 在该消息末尾追加 `<unimem>` 块（会话序言、被动召回结果） |
| `expand_assistant` | 可见 assistant 消息的文本 + tool_use id 哈希 | 把 harness 看到的"合并后"消息展开回真实序列：`assistant(含 unimem 工具调用) → user(tool_result) → assistant(续写)`；混合轮次时还要把我们的 tool_result 前插到下一条 user 消息 |

注入位置的选择：

- **工具定义**追加到 `tools` 数组末尾，内容固定，逐字节稳定。
- **不改 system prompt**。部分订阅鉴权会校验 system 前缀，不碰它最安全。
- **会话序言**（项目卡片 + 置顶记忆）追加到会话第一条 prompt；压缩后在下一条请求重新发送（见 2.5）。
- **被动召回**追加到当前这条 user 消息，即请求尾部，之后由 overlay 重放。
- **工具错误召回**：Anthropic 追加为 tool_result 之后的一个 text block（同一条 user 消息）；Responses 在 `function_call_output` 之后插入一条 `input_text` user 消息；Chat 在 `role: tool` 之后插入一条 user 消息。三者都存为 `augment_user` overlay，锚点是该 tool 结果消息/项。

找不到锚点时（被压缩、被编辑）直接跳过，请求仍然合法，只是丢了一点上下文；压缩检测会让序言和笔记重新注入。

### 2.3 工具拦截与续写（流式）

我们向模型暴露 `unimem_recall / unimem_remember / unimem_forget / unimem_project` 四个工具。harness 并不认识它们，所以它们的调用绝不能流到 harness。

流式改写器是一个状态机：

1. 边读上游 SSE 边转发，并维护 block/item 索引映射。
2. 遇到 `unimem_*` 的 tool_use block，缓冲起来不转发，同时把后续 block 的索引重新编号。
3. 上游以 `stop_reason=tool_use` 结束时，分两种情况：
   - **只有我们的工具**：吞掉 `message_delta/message_stop`，执行工具，把 `assistant + tool_result` 拼进请求再请求上游；跳过新流的 `message_start`，索引偏移后接在原流后面。harness 看到的是一条连续的消息，并记录一条 `expand_assistant` overlay。最多续写 N 轮。
   - **混合**（同时有 harness 自己的工具）：照常把 harness 的工具调用交给它；我们的工具在网关内执行，结果存入 overlay。下一次请求时，把我们的 tool_use 补回 assistant 消息，把 tool_result 前插回 user 消息。
4. usage 如实上报：合并后的 `input_tokens` / `cache_*` / `prompt_tokens` 取**最后一轮**（这才是真实上下文大小，harness 据此决定何时压缩），`output_tokens` / `completion_tokens` 求和。所有轮次的真实总量放在响应头 `x-unimem-usage`（JSON，含 `rounds`）；流式响应在结尾追加一个 SSE 注释 `: x-unimem-usage {…}`；同时写入 `usage` trace。

Responses API（Codex）采用同一套机制，粒度从 content block 换成 output item：`output_index`、`sequence_number` 重新编号，最终 `response.completed` 里的 `output` 只保留可见 item；重放时以相邻可见 item（call_id、reasoning 密文哈希、消息文本哈希）为锚点插回隐藏 item。

### 2.4 失败策略

- 记忆引擎超时（默认召回 1.5s 预算，`inject.recall_budget_ms`）：不注入，直接透传。
- 改写阶段出错：退回原始请求。
- 工具执行出错：把错误作为 tool_result 返回给模型，模型自行处理。
- 上游错误：状态码和 body 原样返回。

### 2.5 预算、去重与压缩（v0.4，`gateway/turn.rs`、`budget.rs`）

每个 agent 请求在 overlay 重放之后、注入之前都过一遍 `Turn::begin`，它读写每个对话的 `session_state`（SQLite 表，一行 JSON，内存中有写穿缓存）。状态的 key 是 `session#首条 user 消息哈希`。Claude Code 的子代理和父代理共享 session id，但首条消息不同，所以各自拿到自己的序言和账本。

**Token 预算。** 估算器不依赖分词器：ASCII 约 4 字符一个 token，CJK 每字一个 token，其它非 ASCII 字符 2 个一个 token。误差 ±30%，但上限远低于窗口，这个精度够用。窗口来自模型表：

- claude-* 为 200k；模型 id 含 `[1m]` 或 `anthropic-beta` 含 `context-1m` 时为 1M；
- gpt-5* / codex 为 272k；o 系列为 200k；
- 未知模型为 128k；
- 可用 `inject.context_windows` 按前缀覆盖，`"*"` 设兜底值。

上限：

| 注入 | 上限 |
|---|---|
| 单条消息（被动召回 + 提示） | min(`max_inject_tokens`=1200, 0.6% 窗口) |
| 序言 | min(`max_preamble_tokens`=1500, 1% 窗口) |
| `unimem_recall` 工具结果 | `max_tool_tokens`=3000，k ≤ 8 |
| 工具错误召回 | ≤ 2 条笔记，≤ `tool_recall_max_tokens`=400 |
| 每个对话的 live 总量（重放的 augmentation + 隐藏轮次） | `max_session_share`=5% 窗口；超出后停止被动召回，压缩后清零，序言可以再发 |

渲染时笔记内容截到 600 字符，`why` 截到 400 字符（写入时也截到 400），并附上指针："unimem_recall 用该 id 可取全文"。`inject.budgets = false` 会关闭全部上限，只用于对比实验。

**跨轮去重账本。** 每注入一条笔记（序言、被动召回、工具错误召回、工具结果）就记一条 `(note id, overlay 锚点, tokens, turn, via)`。每个请求都会修剪账本：锚点的 overlay 本次被重放了，或者笔记 id `(m_…)` 仍出现在请求里，条目就保留，否则丢弃。仍在账本里的笔记不再注入。被压缩或回退掉的笔记自然离开账本，之后可以再次注入。

**压缩检测**，满足任一条件即判定：

- 同一线程的消息数下降超过 30%（且上次至少 6 条）；
- 首条 user 消息变了；
- 出现摘要标记，如 Claude Code 的 "This session is being continued from a previous conversation"、Codex 的 `compaction*` 项或 handoff 文本；
- 这个对话存过 augment overlay，但本次一个都没命中。

判定后清空序言标记、账本和已注入计数，下一条请求（prompt 或 tool turn）重新发送序言。

**工具错误召回（R4）。**

- 触发：tool turn 的输出里有像报错的行（error / FAILED / panic / Traceback / exit code N / NACK / refused … ；"0 failed" 这类测试摘要不算）。
- 查询：用报错行、出现的路径和触发它的命令拼成，走被动召回。
- 门控：笔记必须与查询共享至少一个罕见 token，即只出现在不超过 max(2, 10%) 条笔记里，数字不算。
- 限流：每个对话每 5 个 tool turn 最多一次；同一个报错签名只召回一次；遵守去重账本。
- 报错片段（≤ 500 字符）会记成 role 为 `tool` 的 episode。

**压缩前 flush。** 估算的上下文取请求估算和上一轮真实 usage 中的较大值。超过 `inject.flush_at`（70%）窗口时，每个线程只做一次：后台蒸馏该会话（如果配置了蒸馏），并在下一条消息末尾加一句不超过 60 token 的提示，请 agent 用 unimem_remember 保存持久事实。

**纠正捕获。** prompt 以 "no," / "actually," / "不对" / "别…" 开头，或出现 "that's wrong" / "don't use" / "不是这样" / "别用" 时（问句除外），追加一句不超过 40 token 的提示，建议用 unimem_remember 保存，并把这条 episode 的 importance 加 3。启发式蒸馏会保留高 importance 的纠正。

## 3. 记忆引擎（Jev-Mem in Rust）

### 3.1 数据模型

```
Memory {
  id, scope: "global" | "project:<pid>",
  kind: fact | decision | preference | procedure | gotcha | episode,
  content, why?, entities[], source{harness, session, cwd, git_branch, git_commit},
  created_at, superseded_by?, salience, hits, last_hit_at, pinned, embedding
}
Edge { src, dst, rel: semantic | temporal | causal | entity | supersedes | duplicate, weight, judged_by }
Project { pid, root, remote, name, summary, stack, aliases[], created_at, updated_at }
Episode { session, harness, pid, role, text, ts }   // 无损流水，被动蒸馏的原料
Overlay { anchor, kind, payload, session, created_at }
```

`pid` 的生成规则：有 git remote 时用规范化的 `host/owner/repo`（同一个 repo 的所有 worktree、clone 共享）；否则用 git root 路径；再否则用 cwd。

### 3.2 System One：可插拔的 Decider

```rust
enum Question { Noul { instructions }, Choice { instructions, criteria: [(name, desc)] }, Score { instructions, levels } }
trait Decider { async fn ask(&self, state: Value, questions: [(key, Question)]) -> Map<key, Answer> }
```

后端实现：

- `JevDecider`：TypeSafe `/v1/systemone`，`base_url` 可换成任何兼容服务（本地 jevk5 等）。
- `LlmDecider`：任何 OpenAI 兼容的小模型，用 JSON 输出模拟类型化问题。
- `HeuristicDecider`：离线、确定性，规则加相似度阈值，保证零配置也能用。
- `ChainDecider`：按顺序尝试，前一个不可用就降级到下一个；缺答案按 fail-closed 处理。

所有阈值写在代码和配置里，决策过程可追踪（`unimem trace` / 日志）。

### 3.3 写路径

1. **代码筛查**：空内容、超长、密钥（正则）、近重复。近重复阈值**按 embedder 区分**（hash 0.97，e5 等 transformer 0.95，可用 `embedder.near_dup_cos` 覆盖），并且还要求词法骨架 Jaccard ≥ 0.5。各向异性 embedder 会把无关短文本的余弦推到 0.8 以上，所以代码侧先验看到的是校准后的余弦 `(cos − baseline)/(1 − baseline)`（e5 的 baseline 为 0.75，`embedder.cos_baseline`）。`why` 截到 400 字符。
2. **一次批量 System-One 调用**：
   - `kind`：选择题；
   - `scope`：选择题，在 global / 当前项目 / 其它已知项目之间选。这就是"模型决定写全局还是写项目"；
   - `durable`：是否持久事实，而不是"下一步做 X"这类会腐烂的状态；
   - `injection`：是否是针对 agent 的提示注入。
3. **候选邻居**：在相关作用域内做混合检索，取 top-k。
4. **一次批量关系判断**（每个候选一组问题）：`duplicate`、`supersedes`、`explains_why`（因果）、`same_entity`。概率超过阈值才建边；`supersedes` 会把旧记忆标记为 superseded（不删除，只降权）。
5. 写入节点、边、FTS、向量；代码负责确定性边（同会话的时间序）。

### 3.4 读路径

1. **Gate**（只用于被动召回）：判断"这句提问需不需要记忆"，不需要就零注入。
2. **Anchor**：BM25 加向量，用 RRF 融合，范围是当前项目 + global。查询向量用 `Embedder::embed_query`（E5 用 "query: " 前缀，笔记用 "passage: "）。如果提问提到其它项目名或别名，把那个项目也加入范围。
3. **Route**：选择要展开的视图（语义 / 时间 / 因果 / 实体）并分配预算。
4. **Expand**：从 anchor 沿选定的边展开，节点数、边数、调用次数都有上限。
5. **Score**：批量判断每个候选"是否与问题相关"，按概率排序，过滤掉 superseded。**被动路径更严**（被动召回和序言都适用）：superseded、被判为过时的笔记，以及注入先验 > 0.2 或像是对 agent 下指令/授权（`screen::addresses_agent`：如 "auto-approve"、"doesn't need to be asked"、"note for AI agents"）的笔记一律不推送。agent 主动调用 `unimem_recall` 时仍然能看到它们，但会带 superseded 标记。
6. **Sufficiency**：判断证据是否充分。不充分就放宽预算再展开一轮；返回 `sufficient/missing` 信号，让 agent 知道"我不知道"。

### 3.5 项目智能识别

- **硬信号（确定性）**：cwd → git root → remote → manifest（Cargo.toml / package.json / pyproject 的 name、description）→ README 首段。
- **软信号（模型）**：
  - 项目卡片的 `summary`（项目做什么）先用 README/manifest 生成启发式初稿；agent 可以随时用 `unimem_project` 工具改写。环路里的大模型最清楚项目在做什么，这比网关自己再调用一个模型更准、也更便宜。
  - 写入时的 scope 判定：见 3.3。
  - 跨项目引用：检测提问里的项目名和别名。
- **别名合并**：同一个 remote 的不同路径自动归为同一项目；路径改名后，第一次见到时按 remote 认回原项目。

## 4. 被动捕获

网关能看到所有对话，所以每个 prompt turn 都把用户输入、assistant 文本，以及 tool 输出里的报错片段（role 为 `tool`，≤ 500 字符）写入 `episodes`，带 cwd 和 pid。写入前先做**密钥脱敏**：`screen::has_secret` 命中的片段替换成 «redacted»，仍可疑的整行丢弃。episode 有 TTL（`distill.episode_ttl_days` = 30，每 6 小时 GC 一次）。

后台蒸馏器（可配置一个便宜的模型）从流水里抽取用户纠正、明确偏好和已做出的决定：

- **分块处理**：按 `distill.chunk_chars`（24k）从旧到新分块，只把处理过的块标记为已蒸馏。v0.3 只读最新 24k 却把全部标记掉，丢数据。
- **mem0 式更新**：有模型时，每个候选连同它最近的 5 条已有记忆交给模型，返回 ADD / UPDATE(id) / DELETE(id) / NOOP。id 不在候选邻居里时按 ADD 处理。没有模型或调用失败时，退回标准写路径，它自带去重和取代。
- **触发**：
  - 空闲（`idle_secs`）；
  - 同一项目、同一 harness 出现新会话，视为上一个会话结束；
  - 压缩前 flush；
  - 上游响应头显示剩余额度低于 `distill.min_rate_limit_remaining`（10%，读 `anthropic-ratelimit-*-remaining/limit`、`x-ratelimit-remaining/limit-*`）时，以上触发全部跳过。

蒸馏器默认关闭；只有显式写入（agent 调用工具）是默认开启的。

## 5. 接入方式

| Harness | 配置 |
|---|---|
| Claude Code | `export ANTHROPIC_BASE_URL=http://127.0.0.1:4777`（API key 和订阅 OAuth 的鉴权头都原样透传） |
| Codex CLI | `~/.codex/config.toml` 加一个 `model_provider`，`base_url = "http://127.0.0.1:4777/v1"`，`wire_api = "responses"` |
| 其它 OpenAI 兼容客户端 | `base_url = http://127.0.0.1:4777/v1` |
| 只想用 MCP | `http://127.0.0.1:4777/mcp`（streamable HTTP） |

上游路由：Anthropic 方言默认 `https://api.anthropic.com`；Responses 方言如果请求带 `chatgpt-account-id`，走 `https://chatgpt.com/backend-api/codex`，否则走 `https://api.openai.com`。都可以在配置里改。

## 6. 已知风险和待验证

- Codex 使用 ChatGPT 登录时的自定义 provider 鉴权透传，需要在真机上验证。
- 改写订阅流量是否符合各家条款，需要用户自行确认。
- 混合轮次在不同 harness 版本下的回传格式（尤其是 thinking block 被裁剪的情况）依赖锚点的容错设计，需要用真实流量回归。
- 启发式 Decider 的阈值需要用真实数据调参；Jev 的阈值同样需要按所用模型重新标定。
- 压缩检测和压缩请求识别依赖 Claude Code / Codex 当前的文本和项类型（摘要标记、"create a detailed summary" 指令），需要用真实流量回归；消息数下降和锚点零命中这两条与 harness 无关，是兜底。
- Responses 在 `function_call_output` 之后插入 user 消息（工具错误召回），需要在真实 Codex 后端上验证是否被接受。
- E5 结果见 `bench/results/v0.4/SUMMARY.md`。
