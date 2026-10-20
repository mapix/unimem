//! OpenAI Responses dialect (`/v1/responses`, `/responses`, `/backend-api/codex/responses`) — Codex and friends.
//!
//! Same architecture as the Anthropic dialect (see `anthropic.rs`), with the *output item* as the unit instead
//! of the content block:
//!
//! 1. **Overlay replay** against the harness-visible `input` items:
//!    * `expand_items` — hidden items (our `function_call`s and their `function_call_output`s) are re-inserted
//!      next to the visible model-output items they were produced with.
//!    * `augment_user` — the `<unimem…>` text appended to a user prompt is re-appended byte-identically.
//! 2. **Injection**: `tools::responses_tools()` are appended to `tools`; on prompt turns (last input item is a
//!    user message with real text) the preamble / passive recall is appended as an extra `input_text` part and
//!    stored as an `augment_user` overlay. `instructions` is never touched.
//! 3. **Interception**: `function_call` items named `unimem_*` never reach the client. They are executed in the
//!    gateway and the upstream is re-asked with `input + round items + function_call_output`s (up to
//!    `inject.max_continuations` times). The client sees one response: `response.created` / `in_progress` once,
//!    contiguous `output_index` and `sequence_number`, and a single final `response.completed` whose
//!    `response.output` holds only visible items and whose `usage` is summed over all rounds.
//!
//! ## Overlay payloads
//!
//! A turn's real item sequence is, per round: visible items, then our calls, then their outputs. Runs of hidden
//! items are anchored on the visible item they precede or — for a trailing run (mixed turns, or continuation
//! stopped by the limit / an upstream error) — on the last visible item they follow:
//!
//! * before visible item `Y`: anchor `exp_anchor(session, key(item before Y), hash("rsp-before", key(Y)))`
//! * after visible item `X`:  anchor `exp_anchor(session, key(item before X), hash("rsp-after",  key(X)))`
//!
//! payload `{"v": 1, "items": [ {"type":"function_call",…}, {"type":"function_call_output",…} ]}`.
//! Item keys: `call_id` for calls / outputs, trimmed text for messages, `encrypted_content` (else summary text,
//! else id) for reasoning. Anchors only fire on model-output items, which carry generated content / unique ids,
//! so a retry of the same prompt never picks up hidden items from an earlier attempt. In a mixed round our call
//! is placed after the harness's calls and our output right after it, i.e. before the harness's own
//! `function_call_output`s in the next request.
//!
//! `id` fields are stripped from items we put back into `input` unless the request has `store: true`
//! (items are not persisted server-side otherwise; Codex strips them too).
//!
//! ## Server-side state (`store: true` + `previous_response_id`)
//!
//! When the client chains responses on the server, the server holds the conversation including our hidden items,
//! so no expansion replay is attempted and no expansion overlays are stored. Tools are still injected and our
//! calls still intercepted; continuations are sent as `previous_response_id = <round id>` with only the
//! `function_call_output`s as input, and the merged response carries the *last* round's id so the client's next
//! `previous_response_id` points at the complete server-side state.

use std::collections::{BTreeMap, HashMap};

use axum::{
    body::Bytes,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::{
    aug_anchor, build_augmentation, decode_body, detect_harness, exp_anchor, find_cwd, forward_headers, header_str, join_url, make_ctx, record_turn, relay,
    response_headers, send_json, sse, sse_response, upstream_error, AppState, ReqInfo, Shared,
};
use crate::engine::Ctx;
use crate::tools;
use crate::util::{hash_hex, strip_harness_noise};

const MAX_BODY: usize = 64 * 1024 * 1024;
pub const K_ITEMS: &str = "expand_items";
pub const K_AUG: &str = "augment_user";

/// Item types the harness executes (anything else with a call shape is server-side, e.g. web_search_call).
const HARNESS_CALLS: &[&str] = &["function_call", "custom_tool_call", "local_shell_call", "shell_call", "apply_patch_call", "computer_call", "mcp_approval_request"];

// ================================================================ item helpers

fn ty(v: &Value) -> &str {
    v.get("type").and_then(Value::as_str).unwrap_or("")
}

fn st_of<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

fn is_message(it: &Value) -> bool {
    ty(it) == "message" || (ty(it).is_empty() && it.get("role").is_some())
}

fn is_ours(it: &Value) -> bool {
    ty(it) == "function_call" && tools::is_ours(st_of(it, "name"))
}

fn is_harness_call(it: &Value) -> bool {
    HARNESS_CALLS.contains(&ty(it)) && !is_ours(it)
}

/// Items the model produced (the only ones expansion anchors may fire on).
fn is_model_item(it: &Value) -> bool {
    if is_message(it) {
        st_of(it, "role") == "assistant"
    } else {
        !ty(it).ends_with("_output")
    }
}

fn hash_parts(parts: &[String]) -> String {
    hash_hex(&parts.iter().map(String::as_str).collect::<Vec<_>>())
}

fn content_texts(c: &Value) -> Vec<String> {
    match c {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|p| p.get("text").and_then(Value::as_str).map(String::from)).collect(),
        _ => vec![],
    }
}

/// Content-derived key of an input / output item, identical for the item we streamed and the item the
/// harness sends back (ids, status, annotations are ignored where something better exists).
pub fn item_key(it: &Value) -> String {
    let t = ty(it);
    if is_message(it) {
        let mut parts = vec![format!("r:{}", st_of(it, "role"))];
        match &it["content"] {
            Value::String(s) => parts.push(format!("t:{}", s.trim())),
            Value::Array(a) => {
                for p in a {
                    match p.get("text").and_then(Value::as_str) {
                        Some(x) => parts.push(format!("t:{}", x.trim())),
                        None => parts.push(format!("o:{}", ty(p))),
                    }
                }
            }
            _ => {}
        }
        return format!("m:{}", hash_parts(&parts));
    }
    match t {
        "reasoning" => {
            if let Some(e) = it.get("encrypted_content").and_then(Value::as_str).filter(|e| !e.is_empty()) {
                return format!("r:e:{}", hash_hex(&[e]));
            }
            let sum = it.get("summary").map(content_texts).unwrap_or_default();
            if !sum.concat().trim().is_empty() {
                return format!("r:s:{}", hash_parts(&sum));
            }
            format!("r:i:{}", st_of(it, "id"))
        }
        _ if t.ends_with("_output") => format!("o:{t}:{}", st_of(it, "call_id")),
        _ => {
            if let Some(c) = it.get("call_id").and_then(Value::as_str) {
                return format!("c:{t}:{c}");
            }
            if let Some(i) = it.get("id").and_then(Value::as_str) {
                return format!("i:{t}:{i}");
            }
            format!("h:{}", hash_hex(&[&it.to_string()]))
        }
    }
}

pub fn before_anchor(session: &str, prev: &str, key: &str) -> String {
    exp_anchor(session, prev, &hash_hex(&["rsp-before", key]))
}

pub fn after_anchor(session: &str, prev: &str, key: &str) -> String {
    exp_anchor(session, prev, &hash_hex(&["rsp-after", key]))
}

/// The user's new input if `it` is a prompt (a user message with real text, not only harness context).
fn prompt_text(it: &Value) -> Option<String> {
    if !is_message(it) || st_of(it, "role") != "user" {
        return None;
    }
    let texts: Vec<String> = content_texts(&it["content"]).iter().map(|t| strip_harness_noise(t)).filter(|t| !t.is_empty()).collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

fn append_part(msg: &mut Value, text: &str) {
    let mut parts = match &msg["content"] {
        Value::String(s) => vec![json!({"type": "input_text", "text": s})],
        Value::Array(a) => a.clone(),
        _ => vec![],
    };
    parts.push(json!({"type": "input_text", "text": text}));
    msg["content"] = Value::Array(parts);
}

/// An item as we put it back into `input`.
fn clean(it: &Value, keep_ids: bool) -> Value {
    let mut v = it.clone();
    if !keep_ids {
        if let Some(o) = v.as_object_mut() {
            o.remove("id");
        }
    }
    v
}

fn text_of_items(items: &[Value]) -> String {
    items.iter().filter(|i| is_message(i)).flat_map(|i| content_texts(&i["content"])).collect::<Vec<_>>().join("\n")
}

/// Sum token counters recursively (`input_tokens_details.cached_tokens` …); non-numbers take the latest value.
pub(super) fn add_usage(acc: &mut Value, u: &Value) {
    let Some(u) = u.as_object() else { return };
    if !acc.is_object() {
        *acc = json!({});
    }
    let Some(a) = acc.as_object_mut() else { return };
    for (k, v) in u {
        match a.get_mut(k) {
            Some(x) if x.is_u64() && v.is_u64() => *x = json!(x.as_u64().unwrap_or(0) + v.as_u64().unwrap_or(0)),
            Some(x) if x.is_object() && v.is_object() => add_usage(x, v),
            _ => {
                a.insert(k.clone(), v.clone());
            }
        }
    }
}

fn req_info(headers: &HeaderMap, body: &Value, input: &[Value]) -> ReqInfo {
    let harness = detect_harness(headers);
    let first_user = input.iter().find_map(prompt_text).unwrap_or_default();
    let session = header_str(headers, "session_id")
        .or_else(|| header_str(headers, "conversation_id"))
        .or_else(|| body.get("prompt_cache_key").and_then(Value::as_str).filter(|s| !s.is_empty()).map(String::from))
        .unwrap_or_else(|| format!("h:{}", hash_hex(&[&harness, &first_user])));
    // Latest context first (Codex re-sends <environment_context> when the cwd changes), then instructions.
    let mut texts: Vec<String> = input.iter().rev().filter(|i| is_message(i) && st_of(i, "role") != "assistant").flat_map(|i| content_texts(&i["content"])).collect();
    if let Some(s) = body.get("instructions").and_then(Value::as_str) {
        texts.push(s.to_string());
    }
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    ReqInfo { harness, session, cwd: find_cwd(&refs) }
}

// ================================================================ overlay replay

fn load_items(st: &AppState, anchor: &str, session: &str) -> Option<Vec<Value>> {
    match st.engine.store.get_overlay(anchor, session) {
        Some((k, p)) if k == K_ITEMS => p["items"].as_array().filter(|a| !a.is_empty()).cloned(),
        _ => None,
    }
}

fn replay(st: &AppState, session: &str, orig: Vec<Value>, expand: bool) -> Vec<Value> {
    let keys: Vec<String> = orig.iter().map(item_key).collect();
    let mut out = Vec::with_capacity(orig.len() + 4);
    for (i, it) in orig.into_iter().enumerate() {
        let prev = if i == 0 { "" } else { keys[i - 1].as_str() };
        let model = expand && is_model_item(&it);
        if model {
            if let Some(items) = load_items(st, &before_anchor(session, prev, &keys[i]), session) {
                out.extend(items);
            }
        }
        let mut it = it;
        if is_message(&it) && st_of(&it, "role") == "user" {
            if let Some((k, p)) = st.engine.store.get_overlay(&aug_anchor(session, prev, &keys[i]), session) {
                if let Some(t) = p["text"].as_str().filter(|t| k == K_AUG && !t.is_empty()) {
                    append_part(&mut it, t);
                }
            }
        }
        out.push(it);
        if model {
            if let Some(items) = load_items(st, &after_anchor(session, prev, &keys[i]), session) {
                out.extend(items);
            }
        }
    }
    out
}

// ================================================================ rounds & finalization

struct Turn {
    ctx: Ctx,
    url: String,
    headers: reqwest::header::HeaderMap,
    body: Value,
    /// `store: true` + `previous_response_id`: the server holds the conversation.
    chained: bool,
    keep_ids: bool,
    /// Key of the last harness-visible input item (the response follows it).
    prev_key: Option<String>,
}

/// One upstream round: output items exactly as produced, plus outputs of our calls we executed.
#[derive(Debug, Clone, Default)]
struct RoundRec {
    items: Vec<Value>,
    results: Vec<Value>,
}

impl RoundRec {
    /// The real sequence: visible items, then our (executed) calls, then their outputs.
    fn real(&self, keep_ids: bool) -> Vec<(bool, Value)> {
        let mut v: Vec<(bool, Value)> = self.items.iter().filter(|i| !is_ours(i)).map(|i| (true, i.clone())).collect();
        if !self.results.is_empty() {
            v.extend(self.items.iter().filter(|i| is_ours(i)).map(|i| (false, clean(i, keep_ids))));
            v.extend(self.results.iter().map(|r| (false, r.clone())));
        }
        v
    }
}

/// Execute our calls of a finished round. Returns (function_call_output items, whether to continue).
async fn after_round(st: &AppState, ctx: &Ctx, items: &[Value], completed: bool, conts: usize) -> (Vec<Value>, bool) {
    let ours: Vec<&Value> = items.iter().filter(|i| is_ours(i)).collect();
    if ours.is_empty() || !completed {
        return (vec![], false);
    }
    let mut results = vec![];
    for it in ours {
        let name = st_of(it, "name");
        let args: Value = serde_json::from_str(st_of(it, "arguments")).unwrap_or_else(|_| json!({}));
        let out = tools::exec(&st.engine, ctx, name, &args).await;
        tracing::debug!(tool = name, "executed unimem tool");
        results.push(json!({"type": "function_call_output", "call_id": it["call_id"], "output": out}));
    }
    let harness = items.iter().any(is_harness_call);
    (results, !harness && conts < st.engine.cfg.inject.max_continuations)
}

fn continue_body(t: &mut Turn, r: &RoundRec, response_id: &Value) {
    if t.chained {
        t.body["previous_response_id"] = response_id.clone();
        t.body["input"] = Value::Array(r.results.clone());
    } else if let Some(input) = t.body.get_mut("input").and_then(Value::as_array_mut) {
        input.extend(r.real(t.keep_ids).into_iter().map(|(_, v)| if is_ours(&v) { v } else { clean(&v, t.keep_ids) }));
    }
}

/// Visible output, expansion overlays, assistant episode.
fn finalize(st: &AppState, t: &Turn, rounds: &[RoundRec]) -> Vec<Value> {
    let visible: Vec<Value> = rounds.iter().flat_map(|r| r.items.iter().filter(|i| !is_ours(i)).cloned()).collect();
    record_turn(st, &t.ctx, "assistant", &text_of_items(&visible));
    if t.chained {
        return visible;
    }
    let Some(prev) = t.prev_key.as_deref() else { return visible };
    let mut vis_keys = vec![];
    let mut vis_prev = vec![];
    let mut p = prev.to_string();
    for it in &visible {
        let k = item_key(it);
        vis_prev.push(p.clone());
        vis_keys.push(k.clone());
        p = k;
    }
    let put = |anchor: String, items: &[Value]| {
        if let Err(e) = st.engine.store.put_overlay(&anchor, &t.ctx.session, K_ITEMS, &json!({"v": 1, "items": items})) {
            tracing::warn!("storing expansion overlay failed: {e}");
        }
    };
    let mut j = 0usize;
    let mut pending: Vec<Value> = vec![];
    for (vis, it) in rounds.iter().flat_map(|r| r.real(t.keep_ids)) {
        if vis {
            if !pending.is_empty() {
                put(before_anchor(&t.ctx.session, &vis_prev[j], &vis_keys[j]), &pending);
                pending.clear();
            }
            j += 1;
        } else {
            pending.push(it);
        }
    }
    if !pending.is_empty() {
        if j == 0 {
            tracing::debug!("response has no visible items; hidden items not stored");
        } else {
            put(after_anchor(&t.ctx.session, &vis_prev[j - 1], &vis_keys[j - 1]), &pending);
        }
    }
    visible
}

// ================================================================ handler

pub(super) async fn raw_passthrough(st: &AppState, url: &str, h: &HeaderMap, bytes: Bytes) -> Response {
    let mut hdrs = forward_headers(h);
    if let Some(ce) = h.get("content-encoding") {
        hdrs.insert("content-encoding", ce.clone());
    }
    match st.client.post(url).headers(hdrs).body(bytes).send().await {
        Ok(r) => relay(r),
        Err(e) => upstream_error(e),
    }
}

fn has_tools(body: &Value) -> bool {
    body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty())
}

fn inject_tools(body: &mut Value) {
    if let Some(t) = body.get_mut("tools").and_then(Value::as_array_mut) {
        if !t.iter().any(|x| x.get("name").and_then(Value::as_str).is_some_and(tools::is_ours)) {
            t.extend(tools::responses_tools());
        }
    }
}

pub async fn handle(State(st): State<Shared>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let cfg_up = &st.engine.cfg.upstream;
    let chatgpt = parts.headers.contains_key("chatgpt-account-id") || parts.uri.path().starts_with("/backend-api/");
    let url = if chatgpt { join_url(&cfg_up.chatgpt, "/responses", parts.uri.query()) } else { join_url(&cfg_up.openai, "/v1/responses", parts.uri.query()) };
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
    if !has_tools(&body) {
        return raw_passthrough(&st, &url, &parts.headers, bytes).await;
    }
    let orig: Vec<Value> = match &body["input"] {
        Value::Array(a) => a.clone(),
        Value::String(s) => vec![json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": s}]})],
        _ => return raw_passthrough(&st, &url, &parts.headers, bytes).await,
    };

    let info = req_info(&parts.headers, &body, &orig);
    let ctx = make_ctx(&st, &info);
    let keep_ids = body.get("store").and_then(Value::as_bool) == Some(true);
    let chained = keep_ids && body.get("previous_response_id").and_then(Value::as_str).is_some_and(|s| !s.is_empty());

    // 1. overlay replay
    let keys: Vec<String> = orig.iter().map(item_key).collect();
    let prompt = orig.last().and_then(prompt_text);
    let mut input = replay(&st, &info.session, orig, !chained);

    // 2. injection
    if st.engine.cfg.inject.tools {
        inject_tools(&mut body);
    }
    if let (Some(text), Some(last_key)) = (prompt.as_deref(), keys.last()) {
        let prev = if keys.len() >= 2 { keys[keys.len() - 2].as_str() } else { "" };
        let anchor = aug_anchor(&info.session, prev, last_key);
        if st.engine.store.get_overlay(&anchor, &info.session).is_none() {
            record_turn(&st, &ctx, "user", text);
            let aug = build_augmentation(&st, &ctx, text).await;
            if let (Some(a), Some(m)) = (aug.as_deref(), input.last_mut()) {
                append_part(m, a);
            }
            if let Err(e) = st.engine.store.put_overlay(&anchor, &info.session, K_AUG, &json!({"text": aug.unwrap_or_default()})) {
                tracing::warn!("storing augment overlay failed: {e}");
            }
        }
    }
    body["input"] = Value::Array(input);

    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let turn = Turn { ctx, url, headers: forward_headers(&parts.headers), body, chained, keep_ids, prev_key: keys.last().cloned() };
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
    if !cur["output"].is_array() {
        return raw_response(bytes);
    }
    let first_obj = cur.clone();
    let mut usage = Value::Null;
    let mut rounds: Vec<RoundRec> = vec![];
    let mut conts = 0;
    let last = loop {
        add_usage(&mut usage, &cur["usage"]);
        let items = cur["output"].as_array().cloned().unwrap_or_default();
        let completed = cur.get("status").and_then(Value::as_str).is_none_or(|s| s == "completed");
        let (results, cont) = after_round(st, &t.ctx, &items, completed, conts).await;
        let rec = RoundRec { items, results };
        rounds.push(rec.clone());
        if !cont {
            break cur;
        }
        continue_body(&mut t, &rec, &cur["id"]);
        conts += 1;
        let next = match send_json(st, &t.url, &t.headers, &t.body).await {
            Ok(r) if r.status().is_success() => r.json::<Value>().await.ok().filter(|v| v["output"].is_array()),
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
            None => break cur,
        }
    };
    let mut merged = last;
    if !t.chained {
        merged["id"] = first_obj["id"].clone();
        if let Some(c) = first_obj.get("created_at") {
            merged["created_at"] = c.clone();
        }
    }
    merged["output"] = Value::Array(finalize(st, &t, &rounds));
    if !usage.is_null() {
        merged["usage"] = usage;
    }
    let mut r = raw_response(Bytes::from(serde_json::to_vec(&merged).unwrap_or_default()));
    r.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    r
}

// ================================================================ streaming

#[derive(PartialEq, Debug)]
enum RoundEnd {
    /// A terminal event (`response.completed` / `incomplete` / `failed`) was seen (and held back).
    Terminal,
    /// Upstream stream ended / failed without a terminal event.
    Truncated,
    /// Upstream sent an `error` event (already forwarded).
    UpstreamError,
    ClientGone,
}

/// Per-stream state shared across rounds.
struct Pump {
    tx: mpsc::Sender<Bytes>,
    seq: u64,
    next_out: u64,
    first_round: bool,
    created: Option<Value>,
}

/// Per-round state.
#[derive(Default)]
struct Round {
    /// upstream output_index → client output_index (None = ours, hidden).
    map: HashMap<u64, Option<u64>>,
    done: BTreeMap<u64, Value>,
    terminal: Option<Value>,
}

impl Round {
    fn items(&self) -> Vec<Value> {
        if let Some(a) = self.terminal.as_ref().and_then(|t| t["response"]["output"].as_array()).filter(|a| !a.is_empty()) {
            return a.clone();
        }
        self.done.values().cloned().collect()
    }
}

impl Pump {
    async fn emit(&mut self, mut j: Value) -> bool {
        if j.get("sequence_number").is_some() {
            j["sequence_number"] = json!(self.seq);
            self.seq += 1;
        }
        let name = ty(&j).to_string();
        self.tx.send(sse::encode_json(Some(&name), &j)).await.is_ok()
    }

    async fn on_event(&mut self, r: &mut Round, ev: sse::Event) -> Option<RoundEnd> {
        let Some(mut j) = ev.json() else {
            if ev.data.trim() == "[DONE]" {
                return None;
            }
            return self.tx.send(sse::encode(ev.event.as_deref(), &ev.data)).await.is_err().then_some(RoundEnd::ClientGone);
        };
        let t = ty(&j).to_string();
        let ok = match t.as_str() {
            "response.created" | "response.in_progress" | "response.queued" => {
                if t == "response.created" && self.created.is_none() {
                    self.created = Some(j["response"].clone());
                }
                !self.first_round || self.emit(j).await
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                r.terminal = Some(j);
                return Some(RoundEnd::Terminal);
            }
            "error" => {
                let _ = self.emit(j).await;
                return Some(RoundEnd::UpstreamError);
            }
            _ if j.get("output_index").is_some() => {
                let up = j["output_index"].as_u64().unwrap_or(0);
                if t == "response.output_item.done" {
                    r.done.insert(up, j["item"].clone());
                }
                let out = match r.map.get(&up) {
                    Some(o) => *o,
                    None => {
                        let o = if j.get("item").is_some_and(is_ours) {
                            None
                        } else {
                            self.next_out += 1;
                            Some(self.next_out - 1)
                        };
                        r.map.insert(up, o);
                        o
                    }
                };
                match out {
                    Some(o) => {
                        j["output_index"] = json!(o);
                        self.emit(j).await
                    }
                    None => true,
                }
            }
            _ => self.emit(j).await,
        };
        (!ok).then_some(RoundEnd::ClientGone)
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
    RoundEnd::Truncated
}

async fn stream_task(st: Shared, mut t: Turn, first: reqwest::Response, tx: mpsc::Sender<Bytes>) {
    let mut p = Pump { tx, seq: 0, next_out: 0, first_round: true, created: None };
    let mut resp = first;
    let mut rounds: Vec<RoundRec> = vec![];
    let mut usage = Value::Null;
    let mut last_terminal: Option<Value> = None;
    let mut conts = 0;
    loop {
        let mut r = Round::default();
        let end = pump_round(resp, &mut p, &mut r).await;
        p.first_round = false;
        if matches!(end, RoundEnd::ClientGone | RoundEnd::UpstreamError) {
            tracing::debug!("stream ended early: {end:?}");
            return;
        }
        let items = r.items();
        let completed = r.terminal.as_ref().is_some_and(|x| ty(x) == "response.completed");
        if let Some(term) = &r.terminal {
            add_usage(&mut usage, &term["response"]["usage"]);
            last_terminal = Some(term.clone());
        }
        let (results, cont) = after_round(&st, &t.ctx, &items, completed, conts).await;
        let rec = RoundRec { items, results };
        rounds.push(rec.clone());
        if !cont {
            break;
        }
        let rid = r.terminal.as_ref().map(|x| x["response"]["id"].clone()).unwrap_or(Value::Null);
        continue_body(&mut t, &rec, &rid);
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
    let visible = finalize(&st, &t, &rounds);
    // No terminal event at all (first round truncated): end like the upstream did.
    let Some(mut term) = last_terminal else { return };
    let obj = &mut term["response"];
    obj["output"] = Value::Array(visible);
    if !usage.is_null() {
        obj["usage"] = usage;
    }
    if !t.chained {
        if let Some(c) = &p.created {
            obj["id"] = c["id"].clone();
            if let Some(ca) = c.get("created_at") {
                obj["created_at"] = ca.clone();
            }
        }
    }
    let _ = p.emit(term).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_ignore_ids_and_status() {
        let a = json!({"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":" hi ","annotations":[]}]});
        let b = json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]});
        assert_eq!(item_key(&a), item_key(&b));
        let r1 = json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"ENC"});
        let r2 = json!({"type":"reasoning","summary":[],"encrypted_content":"ENC"});
        assert_eq!(item_key(&r1), item_key(&r2));
        let f1 = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"shell","arguments":"{}","status":"completed"});
        let f2 = json!({"type":"function_call","call_id":"call_1","name":"shell","arguments":"{}"});
        assert_eq!(item_key(&f1), item_key(&f2));
        assert_ne!(item_key(&f1), item_key(&json!({"type":"function_call_output","call_id":"call_1","output":"x"})));
    }

    #[test]
    fn prompt_detection() {
        assert!(prompt_text(&json!({"type":"message","role":"user","content":[{"type":"input_text","text":"fix it"}]})).is_some());
        assert!(prompt_text(&json!({"role":"user","content":"fix it"})).is_some());
        assert!(prompt_text(&json!({"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context><cwd>/x</cwd></environment_context>"}]})).is_none());
        assert!(prompt_text(&json!({"type":"function_call_output","call_id":"c","output":"ok"})).is_none());
    }

    #[test]
    fn usage_sums_nested() {
        let mut acc = Value::Null;
        add_usage(&mut acc, &json!({"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":2}));
        add_usage(&mut acc, &json!({"input_tokens":5,"input_tokens_details":{"cached_tokens":1},"output_tokens":3}));
        assert_eq!(acc, json!({"input_tokens":15,"input_tokens_details":{"cached_tokens":5},"output_tokens":5}));
    }
}
