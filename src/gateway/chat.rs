//! OpenAI Chat Completions dialect (`/v1/chat/completions`, `/chat/completions`).
//!
//! Same architecture as the Anthropic dialect (see `anthropic.rs`) in the flatter message format:
//!
//! 1. **Overlay replay**: an assistant message the harness saw as one merged message is replaced by the true
//!    sequence `assistant(… tool_calls incl. unimem_*) → tool(results) → assistant(…)`. For mixed turns the
//!    stored sequence ends with the assistant message carrying both the harness's and our `tool_calls`, followed
//!    by our `role: tool` results; the harness's own tool messages then follow as usual. `augment_user`
//!    re-appends the `<unimem…>` text to user prompts byte-identically.
//! 2. **Injection**: `tools::chat_tools()` appended to `tools`; on prompt turns the augmentation is appended to
//!    the last user message (string content: `"\n\n" + text`; array content: an extra `text` part). System /
//!    developer messages are never touched.
//! 3. **Interception**: our `tool_calls` are hidden from the client (streaming: buffered by `tool_calls[].index`,
//!    the remaining calls renumbered contiguously), executed in the gateway, and the upstream is re-asked. The
//!    client sees one completion: text of later rounds continues the same message (separated by a blank line),
//!    one final chunk with the visible `finish_reason`, and — if `stream_options.include_usage` — one usage
//!    chunk with usage summed over all rounds, then `[DONE]`.
//!
//! ## Overlay payload
//!
//! `expand_assistant` (anchor = `exp_anchor(session, key(previous message), hash("chat-vis", visible_key(msg)))`):
//! `{"v": 1, "messages": [ {"role":"assistant",…}, {"role":"tool",…}, …, {"role":"assistant",…} (, {"role":"tool",…}) ]}`.
//! The visible key hashes the assistant text with all whitespace removed (robust to how rounds were joined)
//! plus the harness's tool call ids. Requests with `n > 1` are relayed untouched (merging several choices across
//! continuation rounds is not supported).

use std::collections::HashMap;

use axum::{
    body::Bytes,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::responses::raw_passthrough;
use super::turn::{self, Live, Turn as MemTurn};
use super::{
    aug_anchor, decode_body, detect_harness, exp_anchor, find_cwd, forward_headers, header_str, join_url, make_ctx, merge_usage, record_turn, relay,
    response_headers, send_json, sse, sse_response, trace_usage, upstream_error, usage_header, AppState, ReqInfo, Shared,
};
use crate::engine::Ctx;
use crate::tools;
use crate::util::{hash_hex, strip_harness_noise};

const MAX_BODY: usize = 64 * 1024 * 1024;
pub const K_EXPAND: &str = "expand_assistant";
pub const K_AUG: &str = "augment_user";
const SEP: &str = "\n\n";

// ================================================================ message helpers

fn role(m: &Value) -> &str {
    m.get("role").and_then(Value::as_str).unwrap_or("")
}

fn call_name(c: &Value) -> &str {
    c["function"]["name"].as_str().unwrap_or("")
}

fn is_our_call(c: &Value) -> bool {
    tools::is_ours(call_name(c))
}

fn calls(m: &Value) -> Vec<Value> {
    m.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn texts(c: &Value) -> Vec<String> {
    match c {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|p| p.get("text").and_then(Value::as_str).map(String::from)).collect(),
        _ => vec![],
    }
}

fn hash_parts(parts: &[String]) -> String {
    hash_hex(&parts.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Key of an assistant message as the harness sees it. Empty when there is nothing to key on.
pub fn visible_key(m: &Value) -> String {
    let t: String = texts(&m["content"]).concat().chars().filter(|c| !c.is_whitespace()).collect();
    let mut parts = vec![];
    if !t.is_empty() {
        parts.push(format!("t:{t}"));
    }
    for c in calls(m).iter().filter(|c| !is_our_call(c)) {
        parts.push(format!("u:{}", c["id"].as_str().unwrap_or("")));
    }
    if parts.is_empty() {
        String::new()
    } else {
        hash_parts(&parts)
    }
}

pub fn msg_key(m: &Value) -> String {
    match role(m) {
        "assistant" => format!("a:{}", visible_key(m)),
        "tool" => format!("t:{}", m["tool_call_id"].as_str().unwrap_or("")),
        r => {
            let mut parts = vec![format!("r:{r}")];
            match &m["content"] {
                Value::Array(a) => {
                    for p in a {
                        match p.get("text").and_then(Value::as_str) {
                            Some(t) => parts.push(format!("t:{}", t.trim())),
                            None => parts.push(format!("o:{}", p["type"].as_str().unwrap_or(""))),
                        }
                    }
                }
                c => parts.extend(texts(c).iter().map(|t| format!("t:{}", t.trim()))),
            }
            format!("u:{}", hash_parts(&parts))
        }
    }
}

pub fn expand_anchor(session: &str, prev: &str, vis: &str) -> String {
    exp_anchor(session, prev, &hash_hex(&["chat-vis", vis]))
}

fn prompt_text(m: &Value) -> Option<String> {
    if role(m) != "user" {
        return None;
    }
    let t: Vec<String> = texts(&m["content"]).iter().map(|t| strip_harness_noise(t)).filter(|t| !t.is_empty()).collect();
    if t.is_empty() {
        None
    } else {
        Some(t.join("\n"))
    }
}

fn append_text(m: &mut Value, text: &str) {
    match &mut m["content"] {
        Value::String(s) => {
            s.push_str(SEP);
            s.push_str(text);
        }
        Value::Array(a) => a.push(json!({"type": "text", "text": text})),
        c => *c = json!(text),
    }
}

fn req_info(headers: &HeaderMap, body: &Value, msgs: &[Value]) -> ReqInfo {
    let harness = detect_harness(headers);
    let first_user = msgs.iter().find(|m| role(m) == "user").map(|m| texts(&m["content"]).join("\n")).unwrap_or_default();
    let session = header_str(headers, "session_id")
        .or_else(|| header_str(headers, "conversation_id"))
        .or_else(|| body.get("prompt_cache_key").and_then(Value::as_str).filter(|s| !s.is_empty()).map(String::from))
        .unwrap_or_else(|| format!("h:{}", hash_hex(&[&harness, &first_user])));
    let ctx_texts: Vec<String> = msgs.iter().filter(|m| matches!(role(m), "system" | "developer")).flat_map(|m| texts(&m["content"])).chain(std::iter::once(first_user)).collect();
    let refs: Vec<&str> = ctx_texts.iter().map(String::as_str).collect();
    ReqInfo { harness, session, cwd: find_cwd(&refs) }
}

// ================================================================ overlay replay

/// A user message carrying text unimem added after `role: tool` results (tool-error recall, flush nudge).
fn aug_message(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

/// Replayed messages, matched augment anchors, hidden messages put back.
fn replay(st: &AppState, session: &str, orig: Vec<Value>) -> (Vec<Value>, Vec<String>, Vec<Value>) {
    let keys: Vec<String> = orig.iter().map(msg_key).collect();
    let mut out = Vec::with_capacity(orig.len() + 4);
    let mut anchors = vec![];
    let mut hidden = vec![];
    let n = orig.len();
    for (i, m) in orig.into_iter().enumerate() {
        let prev = if i == 0 { "" } else { keys[i - 1].as_str() };
        match role(&m) {
            "assistant" => {
                let vis = visible_key(&m);
                let ov = if vis.is_empty() { None } else { st.engine.store.get_overlay(&expand_anchor(session, prev, &vis), session) };
                match ov {
                    Some((k, p)) if k == K_EXPAND && p["messages"].as_array().is_some_and(|a| !a.is_empty()) => {
                        let msgs = p["messages"].as_array().cloned().unwrap_or_default();
                        hidden.extend(msgs.iter().take(msgs.len().saturating_sub(1)).cloned());
                        out.extend(msgs);
                    }
                    _ => out.push(m),
                }
            }
            "user" => {
                let mut m = m;
                let anchor = aug_anchor(session, prev, &keys[i]);
                if let Some((k, p)) = st.engine.store.get_overlay(&anchor, session) {
                    if let Some(t) = p["text"].as_str().filter(|t| k == K_AUG && !t.is_empty()) {
                        append_text(&mut m, t);
                        anchors.push(anchor);
                    }
                }
                out.push(m);
            }
            "tool" => {
                // the last tool message of a run may carry a tool-error augmentation (a user message after it)
                let anchor = aug_anchor(session, prev, &keys[i]);
                out.push(m);
                if i + 1 == n || keys.get(i + 1).is_some_and(|k| !k.starts_with("t:")) {
                    if let Some((k, p)) = st.engine.store.get_overlay(&anchor, session) {
                        if let Some(t) = p["text"].as_str().filter(|t| k == K_AUG && !t.is_empty()) {
                            out.push(aug_message(t));
                            anchors.push(anchor);
                        }
                    }
                }
            }
            _ => out.push(m),
        }
    }
    (out, anchors, hidden)
}

/// Text of the trailing `role: tool` messages and the arguments of the calls they answer.
fn trailing_tool_results(msgs: &[Value]) -> (String, String) {
    let n = msgs.iter().rev().take_while(|m| role(m) == "tool").count();
    if n == 0 {
        return (String::new(), String::new());
    }
    let outs = &msgs[msgs.len() - n..];
    let ids: Vec<&str> = outs.iter().filter_map(|m| m["tool_call_id"].as_str()).collect();
    let text = outs.iter().map(|m| texts(&m["content"]).join("\n")).collect::<Vec<_>>().join("\n");
    let cmds = msgs
        .iter()
        .flat_map(calls)
        .filter(|c| c["id"].as_str().is_some_and(|i| ids.contains(&i)))
        .map(|c| {
            let a = c["function"]["arguments"].as_str().unwrap_or("").to_string();
            serde_json::from_str::<Value>(&a).ok().and_then(|v| v["command"].as_str().map(String::from)).unwrap_or(a)
        })
        .collect::<Vec<_>>()
        .join("\n");
    (text, cmds)
}

// ================================================================ rounds & finalization

struct Turn {
    ctx: Ctx,
    url: String,
    headers: reqwest::header::HeaderMap,
    body: Value,
    prev_key: Option<String>,
}

/// One upstream round: the assistant message exactly as produced (content, tool_calls, extras like
/// `reasoning_content`), plus `role: tool` results of our calls we executed.
#[derive(Debug, Clone, Default)]
struct RoundRec {
    msg: Value,
    results: Vec<Value>,
}

fn text_of(m: &Value) -> String {
    texts(&m["content"]).concat()
}

/// The assistant message as it goes back upstream (without empty `tool_calls`, with null content if empty).
fn upstream_msg(m: &Value) -> Value {
    let mut v = m.clone();
    if let Some(o) = v.as_object_mut() {
        o.insert("role".into(), json!("assistant"));
        if o.get("tool_calls").and_then(Value::as_array).is_some_and(|a| a.is_empty()) {
            o.remove("tool_calls");
        }
        for k in ["refusal", "annotations", "audio"] {
            if o.get(k).is_some_and(Value::is_null) {
                o.remove(k);
            }
        }
        if !o.get("content").is_some_and(|c| c.is_string() && !c.as_str().unwrap_or("").is_empty() || c.is_array()) {
            o.insert("content".into(), Value::Null);
        }
    }
    v
}

async fn after_round(st: &AppState, ctx: &Ctx, msg: &Value, finish: &str, conts: usize) -> (Vec<Value>, bool) {
    let all = calls(msg);
    let ours: Vec<&Value> = all.iter().filter(|c| is_our_call(c)).collect();
    if ours.is_empty() || finish != "tool_calls" {
        return (vec![], false);
    }
    let mut results = vec![];
    for c in ours {
        let name = call_name(c);
        let args: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or("")).unwrap_or_else(|_| json!({}));
        let out = tools::exec(&st.engine, ctx, name, &args).await;
        tracing::debug!(tool = name, "executed unimem tool");
        results.push(json!({"role": "tool", "tool_call_id": c["id"], "content": out}));
    }
    let harness = all.iter().any(|c| !is_our_call(c));
    (results, !harness && conts < st.engine.cfg.inject.max_continuations)
}

fn continue_body(body: &mut Value, r: &RoundRec) {
    if let Some(m) = body.get_mut("messages").and_then(Value::as_array_mut) {
        m.push(upstream_msg(&r.msg));
        m.extend(r.results.iter().cloned());
    }
}

fn visible_finish(rounds: &[RoundRec], upstream_last: Value) -> Value {
    match rounds.last() {
        Some(r) if !r.results.is_empty() && !calls(&r.msg).iter().any(|c| !is_our_call(c)) => json!("stop"),
        _ => upstream_last,
    }
}

/// Merge the rounds into the visible message, store the expansion overlay and record the assistant text.
fn finalize(st: &AppState, t: &Turn, mut rounds: Vec<RoundRec>) -> Value {
    let text = rounds.iter().map(|r| text_of(&r.msg)).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(SEP);
    let vcalls: Vec<Value> = rounds.iter().flat_map(|r| calls(&r.msg)).filter(|c| !is_our_call(c)).collect();
    let mut visible = rounds.first().map(|r| r.msg.clone()).unwrap_or_else(|| json!({"role": "assistant"}));
    visible["content"] = if text.is_empty() { Value::Null } else { json!(text) };
    if let Some(o) = visible.as_object_mut() {
        o.remove("tool_calls");
        if !vcalls.is_empty() {
            o.insert("tool_calls".into(), Value::Array(vcalls));
        }
    }
    record_turn(st, &t.ctx, "assistant", &text);

    // Unexecuted calls of ours in the final round (cut off by length…) are dropped from the real sequence.
    if let Some(last) = rounds.last_mut() {
        if last.results.is_empty() {
            let kept: Vec<Value> = calls(&last.msg).into_iter().filter(|c| !is_our_call(c)).collect();
            if let Some(o) = last.msg.as_object_mut() {
                o.remove("tool_calls");
                if !kept.is_empty() {
                    o.insert("tool_calls".into(), Value::Array(kept));
                }
            }
            if text_of(&last.msg).is_empty() && calls(&last.msg).is_empty() && rounds.len() > 1 {
                rounds.pop();
            }
        }
    }
    let hidden = rounds.iter().any(|r| !r.results.is_empty());
    let (Some(prev), true) = (t.prev_key.as_deref(), hidden) else { return visible };
    let vis = visible_key(&visible);
    if vis.is_empty() {
        tracing::debug!("visible assistant message has no text/tool ids; expansion overlay not stored");
        return visible;
    }
    let mut messages = vec![];
    for r in &rounds {
        messages.push(upstream_msg(&r.msg));
        messages.extend(r.results.iter().cloned());
    }
    if let Err(e) = st.engine.store.put_overlay(&expand_anchor(&t.ctx.session, prev, &vis), &t.ctx.session, K_EXPAND, &json!({"v": 1, "messages": messages})) {
        tracing::warn!("storing expansion overlay failed: {e}");
    }
    visible
}

// ================================================================ handler

fn has_tools(body: &Value) -> bool {
    body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty())
}

fn inject_tools(body: &mut Value) {
    if let Some(t) = body.get_mut("tools").and_then(Value::as_array_mut) {
        if !t.iter().any(|x| tools::is_ours(x["function"]["name"].as_str().unwrap_or(""))) {
            t.extend(tools::chat_tools());
        }
    }
}

pub async fn handle(State(st): State<Shared>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let url = join_url(&st.engine.cfg.upstream.openai, "/v1/chat/completions", parts.uri.query());
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
    let multi = body.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1);
    if !has_tools(&body) || !body["messages"].is_array() || multi {
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    }
    let orig = body["messages"].as_array().cloned().unwrap_or_default();
    let info = req_info(&parts.headers, &body, &orig);
    let ctx = make_ctx(&st, &info);

    let keys: Vec<String> = orig.iter().map(msg_key).collect();
    let last_role = orig.last().map(|m| role(m).to_string()).unwrap_or_default();
    let prompt = orig.last().and_then(prompt_text);
    if prompt.as_deref().is_some_and(turn::is_compaction_request) {
        st.engine.store.trace("compaction_request", &json!({"dialect": "chat", "msgs": orig.len()}));
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    }
    let (results, commands) = trailing_tool_results(&orig);
    let first_user = orig.iter().find(|m| role(m) == "user");
    let first_hash = first_user.map(msg_key).unwrap_or_default();
    let summary_key = first_user.filter(|m| texts(&m["content"]).iter().any(|t| turn::has_summary_marker(t))).map(msg_key);
    let n_orig = orig.len();
    let (mut messages, anchors, hidden) = replay(&st, &info.session, orig);

    if st.engine.cfg.inject.tools {
        inject_tools(&mut body);
    }
    let model = body["model"].as_str().unwrap_or("").to_string();
    let live = Live::scan(&json!({"tools": body["tools"], "messages": messages}), anchors, n_orig, first_hash, summary_key).with_hidden(&hidden);
    let mut tn = MemTurn::begin(&st, &ctx, live, &model, "");
    if let Some(last_key) = keys.last() {
        let prev = if keys.len() >= 2 { keys[keys.len() - 2].as_str() } else { "" };
        let anchor = aug_anchor(&info.session, prev, last_key);
        if (prompt.is_some() || !results.is_empty()) && st.engine.store.get_overlay(&anchor, &info.session).is_none() {
            let aug = match prompt.as_deref() {
                Some(text) => {
                    record_turn(&st, &ctx, "user", text);
                    tn.prompt_augmentation(&st, &ctx, text, &anchor).await
                }
                None => tn.tool_augmentation(&st, &ctx, &results, &commands, &anchor).await,
            };
            if let Some(a) = aug.as_deref() {
                if prompt.is_some() {
                    if let Some(m) = messages.last_mut() {
                        append_text(m, a);
                    }
                } else {
                    messages.push(aug_message(a));
                }
            }
            if let Err(e) = st.engine.store.put_overlay(&anchor, &info.session, K_AUG, &json!({"text": aug.unwrap_or_default()})) {
                tracing::warn!("storing augment overlay failed: {e}");
            }
        }
    }
    tn.finish(&st);
    body["messages"] = Value::Array(messages);

    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let prev_key = if last_role != "assistant" { keys.last().cloned() } else { None };
    let turn = Turn { ctx, url, headers: forward_headers(&parts.headers), body, prev_key };
    let resp = match send_json(&st, &turn.url, &turn.headers, &turn.body).await {
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
        tokio::spawn(async move { stream_task(st2, turn, resp, tx).await });
        sse_response(status, &up_headers, rx)
    } else {
        non_stream(&st, turn, resp).await
    }
}

// ================================================================ non-streaming

async fn non_stream(st: &AppState, mut t: Turn, first: reqwest::Response) -> Response {
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
    if !cur["choices"][0]["message"].is_object() {
        return raw_response(bytes);
    }
    let mut merged = cur.clone();
    let (mut usage, mut totals) = (Value::Null, Value::Null);
    let mut rounds: Vec<RoundRec> = vec![];
    let mut finish: Value;
    let mut conts = 0;
    loop {
        merge_usage(&mut usage, &mut totals, &cur["usage"]);
        turn::note_usage(st, &t.ctx.session, &cur["usage"]);
        let msg = cur["choices"][0]["message"].clone();
        finish = cur["choices"][0]["finish_reason"].clone();
        let (results, cont) = after_round(st, &t.ctx, &msg, finish.as_str().unwrap_or(""), conts).await;
        let rec = RoundRec { msg, results };
        rounds.push(rec.clone());
        if !cont {
            break;
        }
        continue_body(&mut t.body, &rec);
        conts += 1;
        let next = match send_json(st, &t.url, &t.headers, &t.body).await {
            Ok(r) if r.status().is_success() => r.json::<Value>().await.ok().filter(|v| v["choices"][0]["message"].is_object()),
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
    let fin = visible_finish(&rounds, finish);
    let n_rounds = rounds.len();
    merged["choices"][0]["message"] = finalize(st, &t, rounds);
    merged["choices"][0]["finish_reason"] = fin;
    if !usage.is_null() {
        merged["usage"] = usage;
    }
    trace_usage(st, &t.ctx, &totals, &merged["usage"], n_rounds);
    let mut r = raw_response(Bytes::from(serde_json::to_vec(&merged).unwrap_or_default()));
    r.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    if let Some(h) = usage_header(&totals, n_rounds) {
        r.headers_mut().insert("x-unimem-usage", h);
    }
    r
}

// ================================================================ streaming

#[derive(PartialEq, Debug)]
enum RoundEnd {
    Done,
    Truncated,
    ClientGone,
}

/// Per-stream state shared across rounds.
struct Pump {
    tx: mpsc::Sender<Bytes>,
    /// Template for synthesized chunks (first round's id/created/model/…).
    head: Option<Value>,
    first_round: bool,
    next_out: u64,
    text_sent: bool,
}

/// Per-round state.
#[derive(Default)]
struct Round {
    map: HashMap<u64, Option<u64>>,
    calls: Vec<(u64, Value)>,
    content: String,
    reasoning: String,
    extra: serde_json::Map<String, Value>,
    finish: Value,
    usage: Value,
    started_text: bool,
}

impl Round {
    fn message(&self) -> Value {
        let mut m = json!({"role": "assistant", "content": if self.content.is_empty() { Value::Null } else { json!(self.content) }});
        for (k, v) in &self.extra {
            m[k] = v.clone();
        }
        if !self.reasoning.is_empty() {
            m["reasoning_content"] = json!(self.reasoning);
        }
        let mut cs = self.calls.clone();
        cs.sort_by_key(|c| c.0);
        if !cs.is_empty() {
            m["tool_calls"] = Value::Array(cs.into_iter().map(|c| c.1).collect());
        }
        m
    }
}

fn push_str(v: &mut Value, s: &str) {
    match v {
        Value::String(x) => x.push_str(s),
        _ => *v = json!(s),
    }
}

impl Pump {
    async fn send(&self, v: &Value) -> bool {
        self.tx.send(sse::encode(None, &v.to_string())).await.is_ok()
    }

    fn rehead(&self, j: &mut Value) {
        if let Some(h) = &self.head {
            for k in ["id", "created"] {
                if let Some(v) = h.get(k) {
                    j[k] = v.clone();
                }
            }
        }
    }

    async fn on_event(&mut self, r: &mut Round, ev: sse::Event) -> Option<RoundEnd> {
        if ev.data.trim() == "[DONE]" {
            return Some(RoundEnd::Done);
        }
        let Some(mut j) = ev.json() else {
            return self.tx.send(sse::encode(ev.event.as_deref(), &ev.data)).await.is_err().then_some(RoundEnd::ClientGone);
        };
        if self.head.is_none() {
            let mut h = j.clone();
            if let Some(o) = h.as_object_mut() {
                o.remove("choices");
                o.remove("usage");
            }
            self.head = Some(h);
        }
        if let Some(u) = j.as_object_mut().and_then(|o| o.remove("usage")) {
            if !u.is_null() {
                r.usage = u;
            }
        }
        let Some(choices) = j.get_mut("choices").and_then(Value::as_array_mut) else {
            return None; // usage-only / unknown chunk: held back
        };
        let mut keep = vec![];
        for mut c in std::mem::take(choices) {
            if let Some(f) = c.get_mut("finish_reason").map(Value::take) {
                if !f.is_null() {
                    r.finish = f;
                }
            }
            let d = &mut c["delta"];
            if let Some(o) = d.as_object_mut() {
                if let Some(t) = o.get("content").and_then(Value::as_str).map(String::from) {
                    if !t.is_empty() && !r.started_text && self.text_sent {
                        o.insert("content".into(), json!(format!("{SEP}{t}")));
                    }
                    if !t.is_empty() {
                        r.started_text = true;
                        self.text_sent = true;
                    }
                    r.content.push_str(&t);
                }
                if let Some(t) = o.get("reasoning_content").and_then(Value::as_str) {
                    r.reasoning.push_str(t);
                }
                for (k, v) in o.iter() {
                    if !matches!(k.as_str(), "content" | "reasoning_content" | "tool_calls" | "role") && !v.is_null() {
                        r.extra.insert(k.clone(), v.clone());
                    }
                }
                if !self.first_round {
                    o.remove("role");
                }
                if let Some(tc) = o.remove("tool_calls") {
                    let mut vis = vec![];
                    for mut e in tc.as_array().cloned().unwrap_or_default() {
                        let up = e["index"].as_u64().unwrap_or(0);
                        let out = match r.map.get(&up) {
                            Some(o) => *o,
                            None => {
                                let o = if is_our_call(&e) {
                                    None
                                } else {
                                    self.next_out += 1;
                                    Some(self.next_out - 1)
                                };
                                r.map.insert(up, o);
                                r.calls.push((up, json!({"id": e["id"], "type": "function", "function": {"name": call_name(&e), "arguments": ""}})));
                                o
                            }
                        };
                        if let Some((_, call)) = r.calls.iter_mut().find(|c| c.0 == up) {
                            if let Some(a) = e["function"]["arguments"].as_str() {
                                push_str(&mut call["function"]["arguments"], a);
                            }
                            if call["id"].is_null() && !e["id"].is_null() {
                                call["id"] = e["id"].clone();
                            }
                        }
                        if let Some(o) = out {
                            e["index"] = json!(o);
                            vis.push(e);
                        }
                    }
                    if !vis.is_empty() {
                        o.insert("tool_calls".into(), Value::Array(vis));
                    }
                }
                let empty = o.iter().all(|(k, v)| v.is_null() || (k == "content" && v.as_str() == Some("")));
                if empty && !(self.first_round && o.contains_key("role")) {
                    continue;
                }
            }
            if let Some(o) = c.as_object_mut() {
                o.insert("finish_reason".into(), Value::Null);
            }
            keep.push(c);
        }
        if keep.is_empty() {
            return None;
        }
        *choices = keep;
        self.rehead(&mut j);
        (!self.send(&j).await).then_some(RoundEnd::ClientGone)
    }
}

async fn pump_round(resp: reqwest::Response, p: &mut Pump, r: &mut Round) -> RoundEnd {
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
            if let Some(end) = p.on_event(r, ev).await {
                return end;
            }
        }
    }
    if let Some(ev) = parser.finish() {
        if let Some(end) = p.on_event(r, ev).await {
            return end;
        }
    }
    if r.finish.is_null() {
        RoundEnd::Truncated
    } else {
        RoundEnd::Done
    }
}

async fn stream_task(st: Shared, mut t: Turn, first: reqwest::Response, tx: mpsc::Sender<Bytes>) {
    let include_usage = t.body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let mut p = Pump { tx, head: None, first_round: true, next_out: 0, text_sent: false };
    let mut resp = first;
    let mut rounds: Vec<RoundRec> = vec![];
    let (mut usage, mut totals) = (Value::Null, Value::Null);
    let mut finish: Value;
    let mut conts = 0;
    loop {
        let mut r = Round::default();
        let end = pump_round(resp, &mut p, &mut r).await;
        p.first_round = false;
        if end == RoundEnd::ClientGone {
            return;
        }
        merge_usage(&mut usage, &mut totals, &r.usage);
        turn::note_usage(&st, &t.ctx.session, &r.usage);
        finish = if r.finish.is_null() { json!("stop") } else { r.finish.clone() };
        let f = if end == RoundEnd::Done { finish.as_str().unwrap_or("").to_string() } else { String::new() };
        let msg = r.message();
        let (results, cont) = after_round(&st, &t.ctx, &msg, &f, conts).await;
        let rec = RoundRec { msg, results };
        rounds.push(rec.clone());
        if !cont {
            break;
        }
        continue_body(&mut t.body, &rec);
        conts += 1;
        match send_json(&st, &t.url, &t.headers, &t.body).await {
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
    let fin = visible_finish(&rounds, finish);
    trace_usage(&st, &t.ctx, &totals, &usage, rounds.len());
    finalize(&st, &t, rounds);
    let Some(head) = p.head.clone() else { return };
    let mut last = head.clone();
    last["choices"] = json!([{"index": 0, "delta": {}, "finish_reason": fin}]);
    let _ = p.send(&last).await;
    if include_usage {
        let mut u = head;
        u["choices"] = json!([]);
        u["usage"] = usage;
        let _ = p.send(&u).await;
    }
    let _ = p.tx.send(sse::encode(None, "[DONE]")).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_key_ignores_whitespace_and_our_calls() {
        let a = json!({"role":"assistant","content":"Let me save that.\n\nSaved it.","tool_calls":[{"id":"c1","type":"function","function":{"name":"shell","arguments":"{}"}}]});
        let b = json!({"role":"assistant","content":"Let me save that.Saved it.","tool_calls":[
            {"id":"u1","type":"function","function":{"name":"unimem_recall","arguments":"{}"}},
            {"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"x\":1}"}}]});
        assert_eq!(visible_key(&a), visible_key(&b));
        assert!(visible_key(&json!({"role":"assistant","content":null})).is_empty());
    }

    #[test]
    fn prompt_detection_and_append() {
        assert!(prompt_text(&json!({"role":"user","content":"fix it"})).is_some());
        assert!(prompt_text(&json!({"role":"tool","tool_call_id":"x","content":"ok"})).is_none());
        let mut m = json!({"role":"user","content":"hi"});
        append_text(&mut m, "<unimem-context>x</unimem-context>");
        assert_eq!(m["content"], "hi\n\n<unimem-context>x</unimem-context>");
        let mut m = json!({"role":"user","content":[{"type":"text","text":"hi"}]});
        append_text(&mut m, "z");
        assert_eq!(m["content"][1], json!({"type":"text","text":"z"}));
    }
}
