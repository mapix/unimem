//! End-to-end tests of the Anthropic dialect against a scripted mock upstream.

use std::collections::VecDeque;
use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::HeaderMap,
    response::Response,
    routing::post,
    Router,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

use unimem::config::Config;
use unimem::decide::{ChainDecider, HeuristicDecider};
use unimem::embed::HashEmbedder;
use unimem::engine::Engine;
use unimem::gateway::{self, sse, AppState};
use unimem::store::Store;

// ---------------------------------------------------------------- mock upstream

enum Reply {
    Json(Value),
    Sse(Vec<Value>),
    Status(u16, Value),
}

#[derive(Clone)]
struct Captured {
    path: String,
    headers: HeaderMap,
    raw: Bytes,
}

impl Captured {
    fn body(&self) -> Value {
        serde_json::from_slice(&self.raw).unwrap()
    }
}

#[derive(Default)]
struct Mock {
    script: Mutex<VecDeque<Reply>>,
    seen: Mutex<Vec<Captured>>,
}

async fn mock_handler(State(m): State<Arc<Mock>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let raw = axum::body::to_bytes(body, 1 << 26).await.unwrap();
    m.seen.lock().push(Captured { path: parts.uri.path().to_string(), headers: parts.headers.clone(), raw });
    let reply = m.script.lock().pop_front().unwrap_or(Reply::Json(json!({"input_tokens": 1})));
    match reply {
        Reply::Json(v) => Response::builder().header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        Reply::Status(s, v) => {
            Response::builder().status(s).header("content-type", "application/json").body(Body::from(v.to_string())).unwrap()
        }
        Reply::Sse(evs) => {
            let mut s = String::new();
            for e in evs {
                s.push_str(&format!("event: {}\ndata: {}\n\n", e["type"].as_str().unwrap(), e));
            }
            Response::builder().header("content-type", "text/event-stream").body(Body::from(s)).unwrap()
        }
    }
}

struct Env {
    mock: Arc<Mock>,
    gw: String,
    engine: Arc<Engine>,
    client: reqwest::Client,
}

async fn serve(r: Router) -> String {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, r).await.unwrap() });
    format!("http://{addr}")
}

async fn env(script: Vec<Reply>) -> Env {
    let mock = Arc::new(Mock::default());
    *mock.script.lock() = script.into();
    let up = serve(
        Router::new()
            .route("/v1/messages", post(mock_handler))
            .route("/v1/messages/count_tokens", post(mock_handler))
            .with_state(mock.clone()),
    )
    .await;
    let mut cfg = Config::default();
    cfg.upstream.anthropic = up;
    let engine = Arc::new(Engine::with_parts(
        cfg,
        Store::open_memory().unwrap(),
        Box::new(HashEmbedder { dim: 512 }),
        ChainDecider::new(vec![Box::new(HeuristicDecider)]),
    ));
    let state = Arc::new(AppState { engine: engine.clone(), client: reqwest::Client::new() });
    let gw = serve(gateway::router(state)).await;
    Env { mock, gw, engine, client: reqwest::Client::new() }
}

impl Env {
    fn seen(&self) -> Vec<Captured> {
        self.mock.seen.lock().clone()
    }
    async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}{}", self.gw, path))
            .header("x-api-key", "sk-test")
            .header("anthropic-beta", "beta-1")
            .header("anthropic-version", "2023-06-01")
            .header("x-claude-code-session-id", "11111111-2222-3333-4444-555555555555")
            .header("user-agent", "claude-cli/2.0 (external)")
            .json(body)
            .send()
            .await
            .unwrap()
    }
}

// ---------------------------------------------------------------- script builders

fn start(id: &str) -> Value {
    json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":"claude-x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":40}}})
}
fn text_block(i: u64, t: &str) -> Vec<Value> {
    vec![
        json!({"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text":t}}),
        json!({"type":"content_block_stop","index":i}),
    ]
}
fn thinking_block(i: u64, t: &str, sig: &str) -> Vec<Value> {
    vec![
        json!({"type":"content_block_start","index":i,"content_block":{"type":"thinking","thinking":"","signature":""}}),
        json!({"type":"content_block_delta","index":i,"delta":{"type":"thinking_delta","thinking":t}}),
        json!({"type":"content_block_delta","index":i,"delta":{"type":"signature_delta","signature":sig}}),
        json!({"type":"content_block_stop","index":i}),
    ]
}
fn tool_block(i: u64, id: &str, name: &str, input: &Value) -> Vec<Value> {
    let s = input.to_string();
    let (a, b) = s.split_at(s.len() / 2);
    vec![
        json!({"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}),
        json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":""}}),
        json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":a}}),
        json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":b}}),
        json!({"type":"content_block_stop","index":i}),
    ]
}
fn end(stop: &str, out: u64) -> Vec<Value> {
    vec![
        json!({"type":"message_delta","delta":{"stop_reason":stop,"stop_sequence":null},"usage":{"output_tokens":out}}),
        json!({"type":"message_stop"}),
    ]
}
fn sse_script(id: &str, parts: Vec<Vec<Value>>) -> Reply {
    let mut v = vec![start(id), json!({"type":"ping"})];
    for p in parts {
        v.extend(p);
    }
    Reply::Sse(v)
}

const SYSTEM: &str = "You are Claude Code.\n<env>\nWorking directory: /tmp/unimem-test-nonexistent\n</env>";

fn harness_tools() -> Value {
    json!([{"name":"Bash","description":"run","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}])
}

fn agent_req(messages: Value, stream: bool) -> Value {
    json!({
        "model": "claude-x", "max_tokens": 1024, "stream": stream,
        "system": [{"type":"text","text": SYSTEM, "cache_control": {"type":"ephemeral"}}],
        "tools": harness_tools(),
        "messages": messages,
    })
}

struct Stream {
    events: Vec<(String, Value)>,
}

impl Stream {
    async fn read(r: reqwest::Response) -> Self {
        assert!(r.status().is_success(), "status {}", r.status());
        let bytes = r.bytes().await.unwrap();
        let mut p = sse::Parser::default();
        let events = p.push(&bytes).into_iter().map(|e| (e.event.clone().unwrap_or_default(), e.json().unwrap())).collect();
        Self { events }
    }
    fn of(&self, ty: &str) -> Vec<&Value> {
        self.events.iter().filter(|e| e.0 == ty).map(|e| &e.1).collect()
    }
    /// Reconstruct the assistant message the way a harness would.
    fn message(&self) -> Vec<Value> {
        let mut blocks: Vec<Value> = vec![];
        let mut json_buf: Vec<String> = vec![];
        for (ty, v) in &self.events {
            match ty.as_str() {
                "content_block_start" => {
                    assert_eq!(v["index"].as_u64().unwrap() as usize, blocks.len(), "indices must be contiguous");
                    blocks.push(v["content_block"].clone());
                    json_buf.push(String::new());
                }
                "content_block_delta" => {
                    let i = v["index"].as_u64().unwrap() as usize;
                    let d = &v["delta"];
                    match d["type"].as_str().unwrap() {
                        "text_delta" => {
                            let t = blocks[i]["text"].as_str().unwrap().to_string() + d["text"].as_str().unwrap();
                            blocks[i]["text"] = json!(t);
                        }
                        "thinking_delta" => {
                            let t = blocks[i]["thinking"].as_str().unwrap().to_string() + d["thinking"].as_str().unwrap();
                            blocks[i]["thinking"] = json!(t);
                        }
                        "signature_delta" => blocks[i]["signature"] = d["signature"].clone(),
                        "input_json_delta" => json_buf[i].push_str(d["partial_json"].as_str().unwrap()),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        for (b, j) in blocks.iter_mut().zip(json_buf) {
            if !j.is_empty() {
                b["input"] = serde_json::from_str(&j).unwrap();
            }
        }
        blocks
    }
}

fn names_of_tools(body: &Value) -> Vec<String> {
    body["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect()
}

fn has_unimem_use(content: &Value) -> bool {
    content.as_array().unwrap().iter().any(|b| b["type"] == "tool_use" && b["name"].as_str().unwrap().starts_with("unimem_"))
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn stream_intercepts_merges_and_expands_next_request() {
    let remember = json!({"content": "We deploy the alpha service through ops/deploy.sh, never by hand", "kind": "procedure"});
    let e = env(vec![
        sse_script("msg_1", vec![thinking_block(0, "should save", "sig-1"), text_block(1, "Let me save that."), tool_block(2, "toolu_mem1", "unimem_remember", &remember), end("tool_use", 30)]),
        sse_script("msg_2", vec![text_block(0, "Saved it."), end("end_turn", 7)]),
        Reply::Json(json!({"id":"msg_3","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":1}})),
    ])
    .await;

    let u1 = json!({"role":"user","content":[{"type":"text","text":"<system-reminder>ctx</system-reminder>"},{"type":"text","text":"Remember: deploy alpha via ops/deploy.sh"}]});
    let req1 = agent_req(json!([u1.clone()]), true);
    let s = Stream::read(e.post("/v1/messages", &req1).await).await;

    // client sees one merged message
    assert_eq!(s.of("message_start").len(), 1);
    assert_eq!(s.of("message_stop").len(), 1);
    assert_eq!(s.events.last().unwrap().0, "message_stop");
    let deltas = s.of("message_delta");
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["delta"]["stop_reason"], "end_turn");
    assert_eq!(deltas[0]["usage"]["output_tokens"], 37);
    let starts: Vec<u64> = s.of("content_block_start").iter().map(|v| v["index"].as_u64().unwrap()).collect();
    assert_eq!(starts, vec![0, 1, 2]);
    let visible = s.message();
    assert!(!has_unimem_use(&Value::Array(visible.clone())));
    assert!(!s.events.iter().any(|e| e.1.to_string().contains("unimem_remember")));
    assert_eq!(visible[0]["type"], "thinking");
    assert_eq!(visible[1]["text"], "Let me save that.");
    assert_eq!(visible[2]["text"], "Saved it.");
    assert_eq!(s.of("ping").len(), 2, "pings pass through");

    let seen = e.seen();
    assert_eq!(seen.len(), 2);
    let b1 = seen[0].body();
    let b2 = seen[1].body();
    // auth/beta headers pass through untouched
    assert_eq!(seen[0].headers["x-api-key"], "sk-test");
    assert_eq!(seen[0].headers["anthropic-beta"], "beta-1");
    // system untouched, tools appended
    assert_eq!(b1["system"], req1["system"]);
    assert_eq!(names_of_tools(&b1), vec!["Bash", "unimem_recall", "unimem_remember", "unimem_forget", "unimem_project"]);
    assert_eq!(b1["tools"], b2["tools"]);
    // the prompt got the session preamble appended
    let m1 = b1["messages"][0]["content"].as_array().unwrap();
    assert_eq!(m1.len(), 3);
    assert!(m1[2]["text"].as_str().unwrap().contains("<unimem-context>"));
    // continuation = original + assistant(round 1 incl. thinking & our call) + user(tool_result)
    let msgs2 = b2["messages"].as_array().unwrap();
    assert_eq!(msgs2.len(), 3);
    assert_eq!(msgs2[0], b1["messages"][0]);
    assert_eq!(msgs2[1]["content"][0], json!({"type":"thinking","thinking":"should save","signature":"sig-1"}));
    assert_eq!(msgs2[1]["content"][2]["name"], "unimem_remember");
    assert_eq!(msgs2[1]["content"][2]["input"], remember);
    assert_eq!(msgs2[2]["content"][0]["type"], "tool_result");
    assert_eq!(msgs2[2]["content"][0]["tool_use_id"], "toolu_mem1");
    assert!(msgs2[2]["content"][0]["content"].as_str().unwrap().len() > 3);
    assert!(e.engine.store.count_memories().values().sum::<i64>() >= 1, "remember executed in gateway");

    // next harness request: echo the visible message (with cache_control moved onto it) + new prompt
    let mut vis = visible.clone();
    vis[2]["cache_control"] = json!({"type":"ephemeral"});
    let req2 = agent_req(json!([u1, {"role":"assistant","content": vis}, {"role":"user","content":"thanks, now run the tests"}]), false);
    let r = e.post("/v1/messages", &req2).await;
    assert!(r.status().is_success());
    let seen = e.seen();
    let b3 = seen[2].body();
    let msgs3 = b3["messages"].as_array().unwrap();
    assert_eq!(msgs3.len(), 5, "{}", serde_json::to_string_pretty(&b3["messages"]).unwrap());
    assert_eq!(msgs3[0], b1["messages"][0], "augment overlay replayed byte-identically");
    assert_eq!(msgs3[1], msgs2[1], "hidden round-1 assistant restored");
    assert_eq!(msgs3[2], msgs2[2], "hidden tool_result restored");
    assert_eq!(msgs3[3]["role"], "assistant");
    assert_eq!(msgs3[3]["content"], json!([{"type":"text","text":"Saved it.","cache_control":{"type":"ephemeral"}}]));
    assert_eq!(msgs3[4]["role"], "user");
    assert_eq!(b3["tools"], b1["tools"], "tool definitions byte-stable");
    assert_eq!(b3["system"], req1["system"]);
    // the tools section is byte-identical in the raw request bodies too
    let raw_tools = |c: &Captured| {
        let s = String::from_utf8(c.raw.to_vec()).unwrap();
        let i = s.find("\"tools\"").unwrap();
        s[i..i + 2000].to_string()
    };
    assert_eq!(raw_tools(&seen[0]), raw_tools(&seen[2]));
}

#[tokio::test]
async fn non_stream_merges_rounds_and_sums_usage() {
    let e = env(vec![
        Reply::Json(json!({"id":"m1","type":"message","role":"assistant","model":"x","content":[{"type":"text","text":"Checking memory."},{"type":"tool_use","id":"toolu_r1","name":"unimem_recall","input":{"query":"deploy"}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":10,"cache_read_input_tokens":50}})),
        Reply::Json(json!({"id":"m2","type":"message","role":"assistant","model":"x","content":[{"type":"text","text":"Nothing recorded."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":120,"output_tokens":7,"cache_read_input_tokens":80}})),
        Reply::Json(json!({"id":"m3","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}})),
    ])
    .await;
    let u1 = json!({"role":"user","content":"how do we deploy?"});
    let r = e.post("/v1/messages", &agent_req(json!([u1.clone()]), false)).await;
    assert!(r.status().is_success());
    let totals: Value = serde_json::from_str(r.headers()["x-unimem-usage"].to_str().unwrap()).unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["content"], json!([{"type":"text","text":"Checking memory."},{"type":"text","text":"Nothing recorded."}]));
    assert_eq!(v["stop_reason"], "end_turn");
    // honest usage: input side = the last round (the real context size), output summed over rounds
    assert_eq!(v["usage"]["input_tokens"], 120);
    assert_eq!(v["usage"]["output_tokens"], 17);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 80);
    // true billing totals over both upstream rounds in the header
    assert_eq!(totals["input_tokens"], 220);
    assert_eq!(totals["cache_read_input_tokens"], 130);
    assert_eq!(totals["output_tokens"], 17);
    assert_eq!(totals["rounds"], 2);
    assert_eq!(v["id"], "m1");

    let seen = e.seen();
    assert_eq!(seen.len(), 2);
    let b2 = seen[1].body();
    assert_eq!(b2["messages"][2]["content"][0]["tool_use_id"], "toolu_r1");
    assert!(b2["messages"][2]["content"][0]["content"].as_str().unwrap().contains("sufficient: false"));

    // follow-up expands
    let r = e
        .post("/v1/messages", &agent_req(json!([u1, {"role":"assistant","content": v["content"]}, {"role":"user","content":"ok"}]), false))
        .await;
    assert!(r.status().is_success());
    let b3 = e.seen()[2].body();
    let m = b3["messages"].as_array().unwrap();
    assert_eq!(m.len(), 5);
    assert_eq!(m[1], b2["messages"][1]);
    assert_eq!(m[2], b2["messages"][2]);
    assert_eq!(m[3]["content"], json!([{"type":"text","text":"Nothing recorded."}]));
}

#[tokio::test]
async fn mixed_turn_hands_harness_tools_and_replays_hidden_results() {
    let e = env(vec![
        sse_script(
            "m1",
            vec![text_block(0, "Looking."), tool_block(1, "toolu_mem", "unimem_recall", &json!({"query":"tests"})), tool_block(2, "toolu_bash", "Bash", &json!({"command":"cargo test"})), end("tool_use", 20)],
        ),
        Reply::Json(json!({"id":"m2","type":"message","role":"assistant","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}})),
    ])
    .await;
    let u1 = json!({"role":"user","content":"run the tests"});
    let s = Stream::read(e.post("/v1/messages", &agent_req(json!([u1.clone()]), true)).await).await;
    let vis = s.message();
    assert_eq!(vis.len(), 2);
    assert_eq!(vis[0]["text"], "Looking.");
    assert_eq!(vis[1]["name"], "Bash");
    assert_eq!(vis[1]["input"], json!({"command":"cargo test"}));
    assert_eq!(s.of("message_delta")[0]["delta"]["stop_reason"], "tool_use");
    assert_eq!(s.of("message_delta")[0]["usage"]["output_tokens"], 20);
    assert_eq!(e.seen().len(), 1, "no continuation on mixed turns");

    let harness_result = json!({"type":"tool_result","tool_use_id":"toolu_bash","content":"ok: 3 passed"});
    let req = agent_req(json!([u1, {"role":"assistant","content": vis}, {"role":"user","content":[harness_result.clone()]}]), true);
    let _ = e.post("/v1/messages", &req).await.bytes().await;
    let b = e.seen()[1].body();
    let m = b["messages"].as_array().unwrap();
    assert_eq!(m.len(), 3);
    let a: Vec<&str> = m[1]["content"].as_array().unwrap().iter().map(|b| b["type"].as_str().unwrap()).collect();
    assert_eq!(a, vec!["text", "tool_use", "tool_use"]);
    assert_eq!(m[1]["content"][1]["id"], "toolu_mem");
    assert_eq!(m[1]["content"][1]["input"], json!({"query":"tests"}));
    let u = m[2]["content"].as_array().unwrap();
    assert_eq!(u.len(), 2);
    assert_eq!(u[0]["tool_use_id"], "toolu_mem");
    assert_eq!(u[1], harness_result);
}

#[tokio::test]
async fn non_agent_requests_pass_through_byte_identical() {
    let e = env(vec![Reply::Json(json!({"id":"t","type":"message","content":[{"type":"text","text":"Title"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}))]).await;
    let raw = r#"{"model":"claude-haiku","max_tokens":32,  "messages":[{"role":"user","content":"title for: fix bug"}]}"#;
    let r = e
        .client
        .post(format!("{}/v1/messages", e.gw))
        .header("x-api-key", "k")
        .header("content-type", "application/json")
        .body(raw)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["content"][0]["text"], "Title");
    let seen = e.seen();
    assert_eq!(seen[0].raw.as_ref(), raw.as_bytes());
    assert_eq!(seen[0].headers["x-api-key"], "k");
}

#[tokio::test]
async fn upstream_errors_are_relayed() {
    let err = json!({"type":"error","error":{"type":"rate_limit_error","message":"slow down"}});
    let e = env(vec![Reply::Status(429, err.clone()), Reply::Status(400, err.clone())]).await;
    let r = e.post("/v1/messages", &agent_req(json!([{"role":"user","content":"hi there"}]), true)).await;
    assert_eq!(r.status().as_u16(), 429);
    assert_eq!(r.json::<Value>().await.unwrap(), err);
    let r = e.post("/v1/messages", &agent_req(json!([{"role":"user","content":"hi again"}]), false)).await;
    assert_eq!(r.status().as_u16(), 400);
    assert_eq!(r.json::<Value>().await.unwrap(), err);
}

#[tokio::test]
async fn continuation_error_ends_stream_gracefully() {
    let e = env(vec![
        sse_script(
            "m1",
            vec![text_block(0, "Saving."), tool_block(1, "toolu_x", "unimem_recall", &json!({"query":"q"})), end("tool_use", 4)],
        ),
        Reply::Status(500, json!({"type":"error","error":{"type":"api_error","message":"boom"}})),
    ])
    .await;
    let s = Stream::read(e.post("/v1/messages", &agent_req(json!([{"role":"user","content":"look it up"}]), true)).await).await;
    assert_eq!(s.message().len(), 1);
    assert_eq!(s.of("message_delta")[0]["delta"]["stop_reason"], "end_turn");
    assert_eq!(s.events.last().unwrap().0, "message_stop");
}

#[tokio::test]
async fn tool_result_turns_are_not_augmented_and_augment_is_stable_on_retry() {
    let ok = || {
        Reply::Json(
            json!({"id":"m","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}),
        )
    };
    let e = env(vec![ok(), ok(), ok()]).await;
    let msgs = json!([{"role":"user","content":"please fix the parser"}]);
    e.post("/v1/messages", &agent_req(msgs.clone(), false)).await.bytes().await.unwrap();
    e.post("/v1/messages", &agent_req(msgs.clone(), false)).await.bytes().await.unwrap();
    let seen = e.seen();
    assert_eq!(seen[0].body()["messages"], seen[1].body()["messages"], "retry of the same prompt gets identical augmentation");
    let tr = json!([
        {"role":"user","content":"please fix the parser"},
        {"role":"assistant","content":[{"type":"tool_use","id":"toolu_b","name":"Bash","input":{}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_b","content":"x"}]}
    ]);
    e.post("/v1/messages", &agent_req(tr, false)).await.bytes().await.unwrap();
    let b = e.seen()[2].body();
    assert_eq!(b["messages"][0], seen[0].body()["messages"][0]);
    assert_eq!(b["messages"][2]["content"].as_array().unwrap().len(), 1, "tool_result turn not augmented");
}

#[tokio::test]
async fn count_tokens_includes_our_tools() {
    let e = env(vec![Reply::Json(json!({"input_tokens": 1234}))]).await;
    let r = e
        .post("/v1/messages/count_tokens", &json!({"model":"x","tools": harness_tools(),"messages":[{"role":"user","content":"hi"}]}))
        .await;
    assert_eq!(r.json::<Value>().await.unwrap()["input_tokens"], 1234);
    let s = e.seen();
    assert_eq!(s[0].path, "/v1/messages/count_tokens");
    assert_eq!(names_of_tools(&s[0].body()).len(), 5);
}

#[tokio::test]
async fn expansion_overlays_never_cross_sessions() {
    // Two conversations send the same opener and the upstream answers both identically (a deterministic reply
    // with a hidden unimem_recall round). Session B's follow-up must not get session A's hidden rounds.
    let recall = json!({"query": "release checklist"});
    let round = |id: &str, tool_id: &str| {
        Reply::Json(
            json!({"id":id,"type":"message","role":"assistant","content":[{"type":"text","text":"One moment."},{"type":"tool_use","id":tool_id,"name":"unimem_recall","input":recall}],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":2}}),
        )
    };
    let fin = |id: &str| {
        Reply::Json(
            json!({"id":id,"type":"message","role":"assistant","content":[{"type":"text","text":"No checklist on file."}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}),
        )
    };
    let ok = |id: &str| {
        Reply::Json(
            json!({"id":id,"type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}),
        )
    };
    let e = env(vec![round("a1", "toolu_sessA"), fin("a2"), round("b1", "toolu_sessB"), fin("b2"), ok("a3"), ok("b3")]).await;
    let send = |session: &'static str, body: Value| {
        e.client
            .post(format!("{}/v1/messages", e.gw))
            .header("x-api-key", "sk-test")
            .header("anthropic-version", "2023-06-01")
            .header("x-claude-code-session-id", session)
            .header("user-agent", "claude-cli/2.0 (external)")
            .json(&body)
            .send()
    };
    const SA: &str = "aaaaaaaa-0000-4000-8000-000000000001";
    const SB: &str = "bbbbbbbb-0000-4000-8000-000000000002";
    let u1 = json!({"role":"user","content":"what is on the release checklist?"});
    let va: Value = send(SA, agent_req(json!([u1.clone()]), false)).await.unwrap().json().await.unwrap();
    let vb: Value = send(SB, agent_req(json!([u1.clone()]), false)).await.unwrap().json().await.unwrap();
    assert_eq!(va["content"], vb["content"], "both sessions see the same visible reply");

    let follow =
        |v: &Value| agent_req(json!([u1.clone(), {"role":"assistant","content": v["content"]}, {"role":"user","content":"thanks"}]), false);
    send(SA, follow(&va)).await.unwrap().bytes().await.unwrap();
    send(SB, follow(&vb)).await.unwrap().bytes().await.unwrap();
    let seen = e.seen();
    assert_eq!(seen.len(), 6);
    let a_next = seen[4].body().to_string();
    let b_next = seen[5].body().to_string();
    assert!(a_next.contains("toolu_sessA") && !a_next.contains("toolu_sessB"), "session A replays only its own rounds");
    assert!(b_next.contains("toolu_sessB"), "session B replays its own hidden round");
    assert!(!b_next.contains("toolu_sessA"), "session B must not receive session A's hidden rounds");
}
