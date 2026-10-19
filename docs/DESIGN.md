# uni-mem 设计文档（v0.1）

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
 Claude Code ─┐  /v1/messages            ┌──────────────── uni-mem (单二进制) ────────────────┐
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
- `count_tokens`、无工具的请求、模型列表等其它请求：纯透传。

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
- **会话序言**（项目卡片 + 置顶记忆）追加到第一条 user 消息，会话内冻结。
- **被动召回**追加到当前这条 user 消息，即请求尾部，之后由 overlay 重放。

找不到锚点时（被压缩、被编辑）直接跳过，请求仍然合法，只是丢了一点上下文。

### 2.3 工具拦截与续写（流式）

我们向模型暴露 `unimem_recall / unimem_remember / unimem_forget / unimem_project` 四个工具。harness 并不认识它们，所以它们的调用绝不能流到 harness。

流式改写器是一个状态机：

1. 边读上游 SSE 边转发，并维护 block/item 索引映射。
2. 遇到 `unimem_*` 的 tool_use block，缓冲起来不转发，同时把后续 block 的索引重新编号。
3. 上游以 `stop_reason=tool_use` 结束时，分两种情况：
   - **只有我们的工具**：吞掉 `message_delta/message_stop`，执行工具，把 `assistant + tool_result` 拼进请求再请求上游；跳过新流的 `message_start`，索引偏移后接在原流后面。harness 看到的是一条连续的消息，并记录一条 `expand_assistant` overlay。最多续写 N 轮。
   - **混合**（同时有 harness 自己的工具）：照常把 harness 的工具调用交给它；我们的工具在网关内执行，结果存入 overlay。下一次请求时，把我们的 tool_use 补回 assistant 消息，把 tool_result 前插回 user 消息。
4. usage 在多轮间累加，最终的 `message_delta` 给出合并值。

Responses API（Codex）采用同一套机制，粒度从 content block 换成 output item：`output_index`、`sequence_number` 重新编号，最终 `response.completed` 里的 `output` 只保留可见 item；重放时以相邻可见 item（call_id、reasoning 密文哈希、消息文本哈希）为锚点插回隐藏 item。

### 2.4 失败策略

- 记忆引擎超时（默认召回 1.2s 预算）：不注入，直接透传。
- 改写阶段出错：退回原始请求。
- 工具执行出错：把错误作为 tool_result 返回给模型，模型自行处理。
- 上游错误：状态码和 body 原样返回。

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

1. **代码筛查**：空内容、超长、密钥（正则）、近重复（向量 ≥0.95）。
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
2. **Anchor**：BM25 加向量，用 RRF 融合，范围是当前项目 + global。如果提问提到其它项目名或别名，把那个项目也加入范围。
3. **Route**：选择要展开的视图（语义 / 时间 / 因果 / 实体）并分配预算。
4. **Expand**：从 anchor 沿选定的边展开，节点数、边数、调用次数都有上限。
5. **Score**：批量判断每个候选"是否与问题相关"，按概率排序，过滤掉 superseded。
6. **Sufficiency**：判断证据是否充分。不充分就放宽预算再展开一轮；返回 `sufficient/missing` 信号，让 agent 知道"我不知道"。

### 3.5 项目智能识别

- **硬信号（确定性）**：cwd → git root → remote → manifest（Cargo.toml / package.json / pyproject 的 name、description）→ README 首段。
- **软信号（模型）**：
  - 项目卡片的 `summary`（项目做什么）先用 README/manifest 生成启发式初稿；agent 可以随时用 `unimem_project` 工具改写。环路里的大模型最清楚项目在做什么，这比网关自己再调用一个模型更准、也更便宜。
  - 写入时的 scope 判定：见 3.3。
  - 跨项目引用：检测提问里的项目名和别名。
- **别名合并**：同一个 remote 的不同路径自动归为同一项目；路径改名后，第一次见到时按 remote 认回原项目。

## 4. 被动捕获

网关能看到所有对话，所以每个 prompt turn 都把用户输入和 assistant 文本写入 `episodes`（无损流水，带 cwd 和 pid）。会话空闲后，后台蒸馏器（可配置一个便宜的模型）从流水里抽取"用户纠正、明确偏好、已做出的决定"，再走标准写路径。蒸馏器默认关闭；只有显式写入（agent 调用工具）是默认开启的。

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
