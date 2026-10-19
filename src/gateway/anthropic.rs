//! Anthropic Messages dialect (`/v1/messages`).
//!
//! Pipeline for an agent turn (request carries `tools`):
//!
//! 1. **Overlay replay** against the harness-visible `messages` (before anything else):
//!    * `expand_assistant` — an assistant message the harness saw as one merged message is replaced *entirely*
//!      by the true sequence the upstream model produced (`assistant(… unimem tool_use) → user(tool_result) →
//!      assistant(…)`). For mixed turns our `tool_result`s are prepended to the following user message.
//!    * `augment_user` — the `<unimem…>` text we appended to a user prompt is re-appended byte-identically.
//! 2. **Injection**: our tool definitions are appended to `tools`; on prompt turns the passive-recall /
//!    preamble text is appended to the last user message and stored as an `augment_user` overlay.
//!    `system` is never touched and no `cache_control` breakpoints are added.
//! 3. **Interception**: `unimem_*` tool calls are executed inside the gateway and the upstream is re-asked
//!    (up to `inject.max_continuations` times); the harness sees one continuous message with contiguous
//!    content-block indices, no `unimem_*` blocks, and merged usage.
//!
//! ## Overlay payloads
//!
//! `expand_assistant` (anchor = `exp_anchor(session, key(previous message), hash("vis", visible_key(msg)))`):
//! ```json
//! {"v": 1,
//!  "messages": [ {"role":"assistant","content":[…round 1, incl. thinking + unimem tool_use…]},
//!                {"role":"user","content":[{"type":"tool_result",…}]},
//!                {"role":"assistant","content":[…round 2…]} ],
//!  "trailing_results": [ {"type":"tool_result",…} ]}
//! ```
//! `messages` replaces the visible assistant message one-for-one (full replacement keeps thinking blocks with
//! the tool calls they precede). `trailing_results` (non-empty only when the final stored assistant message
//! still contains executed `unimem_*` calls: mixed turns, or the continuation limit / an upstream error
//! stopped us) are prepended to the next user message.
//!
//! `augment_user` (anchor = `aug_anchor(session, key(previous message), key(this message))`):
//! `{"text": "<unimem-context>…"}`; an empty text records "already processed, nothing to add" so retries
//! of the same prompt are idempotent.
//!
//! The visible key of an assistant message uses only trimmed text of `text` blocks and ids of `tool_use`
//! blocks (thinking/signatures, citations and cache_control are ignored), so it is computed identically from
//! what we streamed and from what the harness later sends back.
//!
//! `count_tokens` is a passthrough that also appends our tool definitions (when the body has tools), so the
//! harness's context estimate includes them. Overlays are not replayed there.

use axum::{
    body::Bytes,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use super::{
    aug_anchor, build_augmentation, decode_body, detect_harness, exp_anchor, find_cwd, forward_headers, header_str, join_url, make_ctx, record_turn, relay,
    response_headers, send_json, session_from_metadata, sse, sse_response, upstream_error, AppState, ReqInfo, Shared,
};
use crate::engine::Ctx;
use crate::tools;
use crate::util::{hash_hex, strip_harness_noise};

const MAX_BODY: usize = 64 * 1024 * 1024;
pub const K_EXPAND: &str = "expand_assistant";
pub const K_AUG: &str = "augment_user";

// ================================================================ content helpers

fn btype(b: &Value) -> &str {
    b.get("type").and_then(Value::as_str).unwrap_or("")
}

/// Content as a list of blocks (string content becomes one text block).
fn blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(a) => a.clone(),
        _ => vec![],
    }
}

fn is_our_use(b: &Value) -> bool {
    btype(b) == "tool_use" && b.get("name").and_then(Value::as_str).is_some_and(tools::is_ours)
}

fn is_harness_use(b: &Value) -> bool {
    btype(b) == "tool_use" && !is_our_use(b)
}

fn hash_parts(parts: &[String]) -> String {
    hash_hex(&parts.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Key of an assistant message as the harness sees it: trimmed text of text blocks + harness tool_use ids.
/// Empty string when there is nothing to key on.
pub fn visible_key(content: &Value) -> String {
    let mut parts = vec![];
    for b in blocks(content) {
        match btype(&b) {
            "text" => {
                let t = b.get("text").and_then(Value::as_str).unwrap_or("").trim();
                if !t.is_empty() {
                    parts.push(format!("t:{t}"));
                }
            }
            "tool_use" if !is_our_use(&b) => parts.push(format!("u:{}", b.get("id").and_then(Value::as_str).unwrap_or(""))),
            _ => {}
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        hash_parts(&parts)
    }
}

/// Key of a user message: trimmed texts, tool_result ids, and a marker for media blocks.
pub fn user_key(content: &Value) -> String {
    let mut parts = vec![];
    for b in blocks(content) {
        match btype(&b) {
            "text" => parts.push(format!("t:{}", b.get("text").and_then(Value::as_str).unwrap_or("").trim())),
            "tool_result" => parts.push(format!("r:{}", b.get("tool_use_id").and_then(Value::as_str).unwrap_or(""))),
            other => parts.push(format!("o:{other}")),
        }
    }
    hash_parts(&parts)
}

pub fn msg_key(m: &Value) -> String {
    let c = m.get("content").cloned().unwrap_or(Value::Null);
    if m.get("role").and_then(Value::as_str) == Some("assistant") {
        format!("a:{}", visible_key(&c))
    } else {
        format!("u:{}", user_key(&c))
    }
}

/// Overlay anchor for expanding the assistant message with visible key `vis` that follows a message keyed `prev`
/// in conversation `session`.
pub fn expand_anchor(session: &str, prev: &str, vis: &str) -> String {
    exp_anchor(session, prev, &hash_hex(&["vis", vis]))
}

fn append_text(msg: &mut Value, text: &str) {
    let mut b = blocks(&msg["content"]);
    b.push(json!({"type": "text", "text": text}));
    msg["content"] = Value::Array(b);
}

fn prepend_blocks(msg: &mut Value, pre: Vec<Value>) {
    let mut b = pre;
    b.extend(blocks(&msg["content"]));
    msg["content"] = Value::Array(b);
}

/// The last `cache_control` set on any block of a message (the harness's breakpoint, if any).
fn cache_control_of(msg: &Value) -> Option<Value> {
    msg.get("content")?.as_array()?.iter().rev().find_map(|b| b.get("cache_control").cloned())
}

fn put_cache_control(msg: &mut Value, cc: Value) {
    if let Some(a) = msg.get_mut("content").and_then(Value::as_array_mut) {
        if let Some(b) = a.iter_mut().rev().find(|b| !matches!(btype(b), "thinking" | "redacted_thinking")) {
            if let Some(o) = b.as_object_mut() {
                o.insert("cache_control".into(), cc);
            }
        }
    }
}

fn text_of_blocks(content: &[Value]) -> String {
    content.iter().filter(|b| btype(b) == "text").filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n")
}

/// The user's new input if `m` is a prompt (a user message with real text, not only tool results / reminders).
fn prompt_text(m: &Value) -> Option<String> {
    if m.get("role").and_then(Value::as_str) != Some("user") {
        return None;
    }
    let texts: Vec<String> = blocks(&m["content"])
        .iter()
        .filter(|b| btype(b) == "text")
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .map(strip_harness_noise)
        .filter(|t| !t.is_empty())
        .collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

fn texts_of(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|b| b.get("text").and_then(Value::as_str).map(String::from)).collect(),
        _ => vec![],
    }
}

fn req_info(headers: &HeaderMap, body: &Value) -> ReqInfo {
    let harness = detect_harness(headers);
    let first_user = body["messages"].as_array().and_then(|m| m.iter().find(|m| m["role"] == "user")).map(|m| texts_of(&m["content"])).unwrap_or_default();
    let session = header_str(headers, "x-claude-code-session-id")
        .or_else(|| session_from_metadata(&body["metadata"]))
        .unwrap_or_else(|| format!("h:{}", hash_hex(&[&harness, &first_user.join("\n")])));
    let mut texts = texts_of(&body["system"]);
    texts.extend(first_user);
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    ReqInfo { harness, session, cwd: find_cwd(&refs) }
}

// ================================================================ overlay replay

struct Replayed {
    messages: Vec<Value>,
    /// Key of the last harness-visible message and of the one before it.
    last_key: Option<String>,
    last_prev_key: String,
    last_role: String,
}

fn replay(st: &AppState, session: &str, orig: Vec<Value>) -> Replayed {
    let keys: Vec<String> = orig.iter().map(msg_key).collect();
    let last_role = orig.last().and_then(|m| m["role"].as_str()).unwrap_or("").to_string();
    let mut out: Vec<Value> = Vec::with_capacity(orig.len() + 4);
    let mut pending: Option<Vec<Value>> = None;
    for (i, m) in orig.into_iter().enumerate() {
        let prev = if i == 0 { "" } else { keys[i - 1].as_str() };
        let role = m["role"].as_str().unwrap_or("").to_string();
        if role == "assistant" {
            if let Some(p) = pending.take() {
                // Hidden tool calls with no following user message: give them their results in one.
                out.push(json!({"role": "user", "content": p}));
            }
            let vis = visible_key(&m["content"]);
            let ov = if vis.is_empty() { None } else { st.engine.store.get_overlay(&expand_anchor(session, prev, &vis), session) };
            match ov {
                Some((k, p)) if k == K_EXPAND && p["messages"].as_array().is_some_and(|a| !a.is_empty()) => {
                    let mut msgs = p["messages"].as_array().cloned().unwrap_or_default();
                    if let (Some(cc), Some(last)) = (cache_control_of(&m), msgs.last_mut()) {
                        put_cache_control(last, cc);
                    }
                    out.extend(msgs);
                    pending = p["trailing_results"].as_array().filter(|a| !a.is_empty()).cloned();
                }
                _ => out.push(m),
            }
        } else {
            let mut m = m;
            if let Some(p) = pending.take() {
                prepend_blocks(&mut m, p);
            }
            if let Some((k, p)) = st.engine.store.get_overlay(&aug_anchor(session, prev, &keys[i]), session) {
                if let Some(t) = p["text"].as_str().filter(|t| k == K_AUG && !t.is_empty()) {
                    append_text(&mut m, t);
                }
            }
            out.push(m);
        }
    }
    if let Some(p) = pending.take() {
        out.push(json!({"role": "user", "content": p}));
    }
    let n = keys.len();
    Replayed { messages: out, last_key: keys.last().cloned(), last_prev_key: if n >= 2 { keys[n - 2].clone() } else { String::new() }, last_role }
}

// ================================================================ rounds & finalization

/// One upstream round: the assistant content exactly as produced, plus results of our tools we executed.
#[derive(Debug, Clone, Default)]
struct RoundRec {
    content: Vec<Value>,
    results: Vec<Value>,
}

/// Execute our tool calls of a finished round. Returns (tool_result blocks, whether to continue).
async fn after_round(st: &AppState, ctx: &Ctx, content: &[Value], stop_reason: &str, conts: usize) -> (Vec<Value>, bool) {
    let ours: Vec<&Value> = content.iter().filter(|b| is_our_use(b)).collect();
    if ours.is_empty() || stop_reason != "tool_use" {
        return (vec![], false);
    }
    let mut results = vec![];
    for b in ours {
        let name = b["name"].as_str().unwrap_or("");
        let args = b.get("input").cloned().unwrap_or(json!({}));
        let out = tools::exec(&st.engine, ctx, name, &args).await;
        tracing::debug!(tool = name, "executed uni-mem tool");
        let mut r = json!({"type": "tool_result", "tool_use_id": b["id"], "content": out});
        if out.starts_with("uni-mem error:") {
            r["is_error"] = json!(true);
        }
        results.push(r);
    }
    let harness_tools = content.iter().any(is_harness_use);
    (results, !harness_tools && conts < st.engine.cfg.inject.max_continuations)
}

fn continue_body(body: &mut Value, r: &RoundRec) {
    if let Some(m) = body.get_mut("messages").and_then(Value::as_array_mut) {
        m.push(json!({"role": "assistant", "content": r.content}));
        m.push(json!({"role": "user", "content": r.results}));
    }
}

/// The stop reason the harness should see for the merged message.
fn visible_stop(rounds: &[RoundRec], upstream_last: Value) -> Value {
    match rounds.last() {
        Some(r) if !r.results.is_empty() && !r.content.iter().any(is_harness_use) => json!("end_turn"),
        _ => upstream_last,
    }
}

/// Build the visible content, store the expansion overlay and record the assistant text.
fn finalize(st: &AppState, ctx: &Ctx, prev_key: Option<&str>, mut rounds: Vec<RoundRec>) -> Vec<Value> {
    let visible: Vec<Value> = rounds.iter().flat_map(|r| r.content.iter().filter(|b| !is_our_use(b)).cloned()).collect();
    record_turn(st, ctx, "assistant", &text_of_blocks(&visible));

    // Unexecuted calls of ours in the final round (cut off by max_tokens…) are dropped from the real sequence.
    if let Some(last) = rounds.last_mut() {
        if last.results.is_empty() {
            last.content.retain(|b| !is_our_use(b));
            if last.content.is_empty() {
                rounds.pop();
            }
        }
    }
    let Some(last) = rounds.last() else { return visible };
    let trailing = last.results.clone();
    if rounds.len() < 2 && trailing.is_empty() {
        return visible; // nothing hidden
    }
    let Some(prev) = prev_key else { return visible };
    let vis = visible_key(&Value::Array(visible.clone()));
    if vis.is_empty() {
        tracing::debug!("visible assistant message has no text/tool ids; expansion overlay not stored");
        return visible;
    }
    let n = rounds.len();
    let mut messages = vec![];
    for (i, r) in rounds.iter().enumerate() {
        messages.push(json!({"role": "assistant", "content": r.content}));
        if i + 1 < n {
            messages.push(json!({"role": "user", "content": r.results}));
        }
    }
    let payload = json!({"v": 1, "messages": messages, "trailing_results": trailing});
    if let Err(e) = st.engine.store.put_overlay(&expand_anchor(&ctx.session, prev, &vis), &ctx.session, K_EXPAND, &payload) {
        tracing::warn!("storing expansion overlay failed: {e}");
    }
    visible
}

fn sum_usage(acc: &mut Map<String, Value>, u: &Value) {
    let Some(u) = u.as_object() else { return };
    for (k, v) in u {
        match (acc.get(k).and_then(Value::as_u64), v.as_u64()) {
            (Some(a), Some(b)) => {
                acc.insert(k.clone(), json!(a + b));
            }
            _ => {
                acc.insert(k.clone(), v.clone());
            }
        }
    }
}

// ================================================================ handlers

async fn raw_passthrough(st: &AppState, url: &str, h: &HeaderMap, bytes: Bytes) -> Response {
    let mut hdrs = forward_headers(h);
    if let Some(ce) = h.get("content-encoding") {
        hdrs.insert("content-encoding", ce.clone());
    }
    match st.client.post(url).headers(hdrs).body(bytes).send().await {
        Ok(r) => relay(r),
        Err(e) => upstream_error(e),
    }
}

fn inject_tools(body: &mut Value) {
    if let Some(t) = body.get_mut("tools").and_then(Value::as_array_mut) {
        if !t.iter().any(|x| x.get("name").and_then(Value::as_str).is_some_and(tools::is_ours)) {
            t.extend(tools::anthropic_tools());
        }
    }
}

fn has_tools(body: &Value) -> bool {
    body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty())
}

pub async fn count_tokens(State(st): State<Shared>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let url = join_url(&st.engine.cfg.upstream.anthropic, "/v1/messages/count_tokens", parts.uri.query());
    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let cfg = &st.engine.cfg.inject;
    if !cfg.enabled || !cfg.tools {
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    }
    let parsed = decode_body(&parts.headers, &bytes).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    match parsed {
        Some(mut v) if has_tools(&v) => {
            inject_tools(&mut v);
            match send_json(&st, &url, &forward_headers(&parts.headers), &v).await {
                Ok(r) => relay(r),
                Err(e) => upstream_error(e),
            }
        }
        _ => raw_passthrough(&st, &url, &parts.headers, bytes).await,
    }
}

pub async fn handle(State(st): State<Shared>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let url = join_url(&st.engine.cfg.upstream.anthropic, "/v1/messages", parts.uri.query());
    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    if !st.engine.cfg.inject.enabled {
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    }
    let Some(mut body) = decode_body(&parts.headers, &bytes).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else {
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    };
    // Turn classification: only agent turns (with tools) are rewritten.
    if !has_tools(&body) || !body["messages"].is_array() {
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    }

    let info = req_info(&parts.headers, &body);
    let ctx = make_ctx(&st, &info);

    // 1. overlay replay
    let orig = body["messages"].as_array().cloned().unwrap_or_default();
    let prompt = orig.last().and_then(prompt_text);
    let rp = replay(&st, &info.session, orig);
    let mut messages = rp.messages;

    // 2. injection
    if st.engine.cfg.inject.tools {
        inject_tools(&mut body);
    }
    if let (Some(text), Some(last_key)) = (prompt.as_deref(), rp.last_key.as_deref()) {
        let anchor = aug_anchor(&info.session, &rp.last_prev_key, last_key);
        if st.engine.store.get_overlay(&anchor, &info.session).is_none() {
            record_turn(&st, &ctx, "user", text);
            let aug = build_augmentation(&st, &ctx, text).await;
            if let (Some(a), Some(m)) = (aug.as_deref(), messages.last_mut()) {
                append_text(m, a);
            }
            if let Err(e) = st.engine.store.put_overlay(&anchor, &info.session, K_AUG, &json!({"text": aug.unwrap_or_default()})) {
                tracing::warn!("storing augment overlay failed: {e}");
            }
        }
    }
    body["messages"] = Value::Array(messages);

    // The expansion overlay of the response is keyed on the message the response will follow.
    let prev_key = if rp.last_role == "user" { rp.last_key.clone() } else { None };
    let headers = forward_headers(&parts.headers);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let resp = match send_json(&st, &url, &headers, &body).await {
        Ok(r) => r,
        Err(e) => return upstream_error(e),
    };
    if !resp.status().is_success() {
        return relay(resp);
    }
    if stream {
        let is_sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.contains("event-stream"));
        if !is_sse {
            return relay(resp);
        }
        let status = resp.status().as_u16();
        let up_headers = resp.headers().clone();
        let (tx, rx) = mpsc::channel::<Bytes>(64);
        let st2 = st.clone();
        tokio::spawn(async move {
            stream_task(st2, ctx, url, headers, body, resp, tx, prev_key).await;
        });
        sse_response(status, &up_headers, rx)
    } else {
        non_stream(&st, &ctx, &url, &headers, body, resp, prev_key.as_deref()).await
    }
}

// ================================================================ non-streaming

async fn non_stream(st: &AppState, ctx: &Ctx, url: &str, headers: &reqwest::header::HeaderMap, mut body: Value, first: reqwest::Response, prev_key: Option<&str>) -> Response {
    let status = first.status().as_u16();
    let up_headers = first.headers().clone();
    let bytes = match first.bytes().await {
        Ok(b) => b,
        Err(e) => return upstream_error(e),
    };
    let raw_response = |b: Bytes| {
        let mut r = Response::new(axum::body::Body::from(b));
        *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
        *r.headers_mut() = response_headers(&up_headers);
        r
    };
    let Ok(mut cur) = serde_json::from_slice::<Value>(&bytes) else { return raw_response(bytes) };
    if !cur["content"].is_array() {
        return raw_response(bytes);
    }
    let mut merged = cur.clone();
    let mut usage = Map::new();
    let mut rounds: Vec<RoundRec> = vec![];
    let mut conts = 0;
    loop {
        sum_usage(&mut usage, &cur["usage"]);
        let content = cur["content"].as_array().cloned().unwrap_or_default();
        let stop = cur["stop_reason"].as_str().unwrap_or("").to_string();
        merged["stop_reason"] = cur["stop_reason"].clone();
        merged["stop_sequence"] = cur.get("stop_sequence").cloned().unwrap_or(Value::Null);
        let (results, cont) = after_round(st, ctx, &content, &stop, conts).await;
        let rec = RoundRec { content, results };
        rounds.push(rec.clone());
        if !cont {
            break;
        }
        continue_body(&mut body, &rec);
        conts += 1;
        let next = match send_json(st, url, headers, &body).await {
            Ok(r) if r.status().is_success() => r.json::<Value>().await.ok().filter(|v| v["content"].is_array()),
            Ok(r) => {
                tracing::warn!("continuation failed: upstream status {}", r.status());
                None
            }
            Err(e) => {
                tracing::warn!("continuation failed: {e}");
                None
            }
        };
        match next {
            Some(v) => cur = v,
            None => break,
        }
    }
    merged["stop_reason"] = visible_stop(&rounds, merged["stop_reason"].clone());
    merged["content"] = Value::Array(finalize(st, ctx, prev_key, rounds));
    merged["usage"] = Value::Object(usage);
    let mut r = raw_response(Bytes::from(serde_json::to_vec(&merged).unwrap_or_default()));
    r.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    r
}

// ================================================================ streaming

/// A content block being reconstructed from stream events.
struct Blk {
    up: u64,
    v: Value,
    json: String,
    ours: bool,
    out: Option<usize>,
    closed: bool,
}

fn push_str(v: &mut Value, k: &str, s: &str) {
    match v.get_mut(k) {
        Some(Value::String(x)) => x.push_str(s),
        _ => v[k] = json!(s),
    }
}

impl Blk {
    fn apply(&mut self, d: &Value) {
        match btype(d) {
            "text_delta" => push_str(&mut self.v, "text", d["text"].as_str().unwrap_or("")),
            "thinking_delta" => push_str(&mut self.v, "thinking", d["thinking"].as_str().unwrap_or("")),
            "signature_delta" => self.v["signature"] = d["signature"].clone(),
            "input_json_delta" => self.json.push_str(d["partial_json"].as_str().unwrap_or("")),
            "citations_delta" => match self.v.get_mut("citations").and_then(Value::as_array_mut) {
                Some(a) => a.push(d["citation"].clone()),
                None => self.v["citations"] = json!([d["citation"]]),
            },
            _ => {}
        }
    }
    fn finish(&self) -> Value {
        let mut v = self.v.clone();
        if !self.json.trim().is_empty() {
            v["input"] = serde_json::from_str(&self.json).unwrap_or_else(|_| json!({}));
        } else if matches!(btype(&v), "tool_use" | "server_tool_use") && !v.get("input").is_some_and(Value::is_object) {
            v["input"] = json!({});
        }
        v
    }
}

#[derive(PartialEq, Debug)]
enum RoundEnd {
    /// message_stop seen.
    Complete,
    /// Upstream stream ended / failed without message_stop.
    Truncated,
    /// Upstream sent an `error` event (already forwarded).
    UpstreamError,
    /// Client went away.
    ClientGone,
}

struct Pump<'a> {
    tx: &'a mpsc::Sender<Bytes>,
    out_next: &'a mut usize,
    started: &'a mut bool,
    blocks: Vec<Blk>,
    stop_reason: Value,
    stop_sequence: Value,
    usage: Value,
}

impl Pump<'_> {
    async fn send(&self, b: Bytes) -> bool {
        self.tx.send(b).await.is_ok()
    }

    /// Returns Some(end) when the round is over.
    async fn on_event(&mut self, ev: sse::Event) -> Option<RoundEnd> {
        let Some(mut j) = ev.json() else {
            return (!self.send(sse::encode(ev.event.as_deref(), &ev.data)).await).then_some(RoundEnd::ClientGone);
        };
        let ty = btype(&j).to_string();
        let ok = match ty.as_str() {
            "message_start" => {
                if *self.started {
                    true
                } else {
                    *self.started = true;
                    self.send(sse::encode_json(Some("message_start"), &j)).await
                }
            }
            "content_block_start" => {
                let up = j["index"].as_u64().unwrap_or(0);
                let cb = j["content_block"].clone();
                let ours = is_our_use(&cb);
                let out = if ours {
                    None
                } else {
                    let o = *self.out_next;
                    *self.out_next += 1;
                    Some(o)
                };
                self.blocks.push(Blk { up, v: cb, json: String::new(), ours, out, closed: false });
                match out {
                    Some(o) => {
                        j["index"] = json!(o);
                        self.send(sse::encode_json(Some(&ty), &j)).await
                    }
                    None => true,
                }
            }
            "content_block_delta" | "content_block_stop" => {
                let up = j["index"].as_u64().unwrap_or(0);
                match self.blocks.iter_mut().rev().find(|b| b.up == up) {
                    Some(b) => {
                        if ty == "content_block_delta" {
                            b.apply(&j["delta"]);
                        } else {
                            b.closed = true;
                        }
                        match (b.ours, b.out) {
                            (false, Some(o)) => {
                                j["index"] = json!(o);
                                self.send(sse::encode_json(Some(&ty), &j)).await
                            }
                            _ => true,
                        }
                    }
                    None => self.send(sse::encode_json(Some(&ty), &j)).await,
                }
            }
            "message_delta" => {
                self.stop_reason = j["delta"]["stop_reason"].clone();
                self.stop_sequence = j["delta"].get("stop_sequence").cloned().unwrap_or(Value::Null);
                self.usage = j["usage"].clone();
                true
            }
            "message_stop" => return Some(RoundEnd::Complete),
            "error" => {
                let _ = self.send(sse::encode(ev.event.as_deref(), &ev.data)).await;
                return Some(RoundEnd::UpstreamError);
            }
            _ => self.send(sse::encode(ev.event.as_deref(), &ev.data)).await,
        };
        (!ok).then_some(RoundEnd::ClientGone)
    }

    /// Close forwarded blocks the upstream never closed (truncated stream), so the client sees valid framing.
    async fn close_open(&mut self) {
        let open: Vec<usize> = self.blocks.iter_mut().filter(|b| !b.closed && b.out.is_some()).map(|b| {
            b.closed = true;
            b.out.unwrap_or(0)
        }).collect();
        for o in open {
            let _ = self.send(sse::encode_json(Some("content_block_stop"), &json!({"type": "content_block_stop", "index": o}))).await;
        }
    }

    fn content(&self) -> Vec<Value> {
        let mut b: Vec<&Blk> = self.blocks.iter().collect();
        b.sort_by_key(|x| x.up);
        b.into_iter().map(Blk::finish).collect()
    }
}

async fn pump_round(resp: reqwest::Response, p: &mut Pump<'_>) -> RoundEnd {
    let mut parser = sse::Parser::default();
    let mut s = resp.bytes_stream();
    while let Some(chunk) = s.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("upstream stream error: {e}");
                return RoundEnd::Truncated;
            }
        };
        for ev in parser.push(&chunk) {
            if let Some(end) = p.on_event(ev).await {
                return end;
            }
        }
    }
    if let Some(ev) = parser.finish() {
        if let Some(end) = p.on_event(ev).await {
            return end;
        }
    }
    RoundEnd::Truncated
}

#[allow(clippy::too_many_arguments)]
async fn stream_task(
    st: Shared,
    ctx: Ctx,
    url: String,
    headers: reqwest::header::HeaderMap,
    mut body: Value,
    first: reqwest::Response,
    tx: mpsc::Sender<Bytes>,
    prev_key: Option<String>,
) {
    let mut resp = first;
    let mut rounds: Vec<RoundRec> = vec![];
    let mut out_next = 0usize;
    let mut started = false;
    let mut usage = Map::new();
    let mut last_stop: Value;
    let mut last_seq: Value;
    let mut conts = 0;
    loop {
        let mut p = Pump { tx: &tx, out_next: &mut out_next, started: &mut started, blocks: vec![], stop_reason: Value::Null, stop_sequence: Value::Null, usage: Value::Null };
        let end = pump_round(resp, &mut p).await;
        if matches!(end, RoundEnd::ClientGone | RoundEnd::UpstreamError) {
            tracing::debug!("stream ended early: {end:?}");
            return;
        }
        if end == RoundEnd::Truncated {
            p.close_open().await;
        }
        let content = p.content();
        sum_usage(&mut usage, &p.usage);
        last_stop = if p.stop_reason.is_null() { json!("end_turn") } else { p.stop_reason.clone() };
        last_seq = p.stop_sequence.clone();
        let stop = if end == RoundEnd::Complete { last_stop.as_str().unwrap_or("").to_string() } else { String::new() };
        let (results, cont) = after_round(&st, &ctx, &content, &stop, conts).await;
        let rec = RoundRec { content, results };
        rounds.push(rec.clone());
        if !cont {
            break;
        }
        continue_body(&mut body, &rec);
        conts += 1;
        match send_json(&st, &url, &headers, &body).await {
            Ok(r) if r.status().is_success() => resp = r,
            Ok(r) => {
                tracing::warn!("continuation failed: upstream status {}", r.status());
                break;
            }
            Err(e) => {
                tracing::warn!("continuation failed: {e}");
                break;
            }
        }
    }
    let stop = visible_stop(&rounds, last_stop);
    finalize(&st, &ctx, prev_key.as_deref(), rounds);
    if !started {
        return; // nothing was ever streamed; nothing sensible to close
    }
    if !usage.contains_key("output_tokens") {
        usage.insert("output_tokens".into(), json!(0));
    }
    let delta = json!({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": last_seq}, "usage": usage});
    let _ = tx.send(sse::encode_json(Some("message_delta"), &delta)).await;
    let _ = tx.send(sse::encode_json(Some("message_stop"), &json!({"type": "message_stop"}))).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_key_ignores_thinking_and_whitespace() {
        let a = json!([{"type":"thinking","thinking":"x","signature":"s1"},{"type":"text","text":" hi \n"},{"type":"tool_use","id":"t1","name":"Bash","input":{}}]);
        let b = json!([{"type":"thinking","thinking":"y","signature":"s2"},{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}},{"type":"tool_use","id":"t1","name":"Bash","input":{"a":1}}]);
        assert_eq!(visible_key(&a), visible_key(&b));
        let c = json!([{"type":"text","text":"hi"},{"type":"tool_use","id":"u9","name":"unimem_recall","input":{}},{"type":"tool_use","id":"t1","name":"Bash","input":{}}]);
        assert_eq!(visible_key(&a), visible_key(&c));
        assert_eq!(visible_key(&json!("hi")), visible_key(&json!([{"type":"text","text":"hi"}])));
        assert!(visible_key(&json!([{"type":"thinking","thinking":"x"}])).is_empty());
    }

    #[test]
    fn prompt_detection() {
        assert!(prompt_text(&json!({"role":"user","content":"fix it"})).is_some());
        assert!(prompt_text(&json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"ok"}]})).is_none());
        assert!(prompt_text(&json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"ok"},{"type":"text","text":"<system-reminder>r</system-reminder>"}]})).is_none());
    }

    #[test]
    fn blk_reconstruction() {
        let mut b = Blk { up: 0, v: json!({"type":"tool_use","id":"a","name":"unimem_recall","input":{}}), json: String::new(), ours: true, out: None, closed: false };
        b.apply(&json!({"type":"input_json_delta","partial_json":"{\"query\":"}));
        b.apply(&json!({"type":"input_json_delta","partial_json":"\"x\"}"}));
        assert_eq!(b.finish()["input"], json!({"query":"x"}));
        let mut t = Blk { up: 1, v: json!({"type":"thinking","thinking":"","signature":""}), json: String::new(), ours: false, out: Some(0), closed: false };
        t.apply(&json!({"type":"thinking_delta","thinking":"ab"}));
        t.apply(&json!({"type":"signature_delta","signature":"sig"}));
        assert_eq!(t.finish(), json!({"type":"thinking","thinking":"ab","signature":"sig"}));
    }
}
