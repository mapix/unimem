//! End-to-end tests of the Chat Completions dialect against a scripted mock upstream.

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
    let reply = m.script.lock().pop_front().unwrap_or_else(|| Reply::Json(completion("d", text_msg("ok"), "stop", 1, 1)));
    match reply {
        Reply::Json(v) => Response::builder().header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        Reply::Status(s, v) => {
            Response::builder().status(s).header("content-type", "application/json").body(Body::from(v.to_string())).unwrap()
        }
        Reply::Sse(chunks) => {
            let mut s = String::new();
            for c in chunks {
                s.push_str(&format!("data: {c}\n\n"));
            }
            s.push_str("data: [DONE]\n\n");
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
    let up = serve(Router::new().route("/v1/chat/completions", post(mock_handler)).with_state(mock.clone())).await;
    let mut cfg = Config::default();
    cfg.upstream.openai = up;
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
    async fn post(&self, body: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.gw))
            .header("authorization", "Bearer sk-test")
            .header("user-agent", "opencode/1.0")
            .json(body)
            .send()
            .await
            .unwrap()
    }
}

// ---------------------------------------------------------------- script builders

fn text_msg(t: &str) -> Value {
    json!({"role":"assistant","content":t})
}
fn call(id: &str, name: &str, args: &Value) -> Value {
    json!({"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}})
}
fn completion(id: &str, msg: Value, finish: &str, inp: u64, out: u64) -> Value {
    json!({"id":id,"object":"chat.completion","created":1700000000,"model":"gpt-x","choices":[{"index":0,"message":msg,"finish_reason":finish}],
           "usage":{"prompt_tokens":inp,"completion_tokens":out,"total_tokens":inp + out,"prompt_tokens_details":{"cached_tokens":inp / 2}}})
}

fn chunk(id: &str, delta: Value, finish: Value) -> Value {
    json!({"id":id,"object":"chat.completion.chunk","created":1700000000,"model":"gpt-x","choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
}

/// A streamed completion: text (split in two deltas), then tool calls (args split), finish, usage chunk.
fn sse_script(id: &str, text: &str, calls: Vec<(&str, &str, Value)>, finish: &str, inp: u64, out: u64) -> Reply {
    let mut v = vec![chunk(id, json!({"role":"assistant","content":""}), Value::Null)];
    if !text.is_empty() {
        let (a, b) = text.split_at(text.len() / 2);
        v.push(chunk(id, json!({"content":a}), Value::Null));
        v.push(chunk(id, json!({"content":b}), Value::Null));
    }
    for (i, (cid, name, args)) in calls.iter().enumerate() {
        v.push(chunk(
            id,
            json!({"tool_calls":[{"index":i,"id":cid,"type":"function","function":{"name":name,"arguments":""}}]}),
            Value::Null,
        ));
        let s = args.to_string();
        let (a, b) = s.split_at(s.len() / 2);
        for d in [a, b] {
            v.push(chunk(id, json!({"tool_calls":[{"index":i,"function":{"arguments":d}}]}), Value::Null));
        }
    }
    v.push(chunk(id, json!({}), json!(finish)));
    v.push(json!({"id":id,"object":"chat.completion.chunk","created":1700000000,"model":"gpt-x","choices":[],"usage":{"prompt_tokens":inp,"completion_tokens":out,"total_tokens":inp + out}}));
    Reply::Sse(v)
}

const SYSTEM: &str = "You are a coding agent.\nWorking directory: /tmp/unimem-test-nonexistent";

fn harness_tools() -> Value {
    json!([{"type":"function","function":{"name":"bash","description":"run","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}])
}
fn agent_req(messages: Value, stream: bool) -> Value {
    let mut v = json!({"model":"gpt-x","messages":messages,"tools":harness_tools(),"stream":stream});
    if stream {
        v["stream_options"] = json!({"include_usage": true});
    }
    v
}
fn sys() -> Value {
    json!({"role":"system","content":SYSTEM})
}

struct Stream {
    chunks: Vec<Value>,
    done: bool,
}

impl Stream {
    async fn read(r: reqwest::Response) -> Self {
        assert!(r.status().is_success(), "status {}", r.status());
        let bytes = r.bytes().await.unwrap();
        let mut p = sse::Parser::default();
        let evs = p.push(&bytes);
        let done = evs.last().is_some_and(|e| e.data == "[DONE]");
        let chunks = evs.into_iter().filter(|e| e.data != "[DONE]").map(|e| e.json().unwrap()).collect();
        Self { chunks, done }
    }
    /// Reconstruct the assistant message like a client; asserts tool call indices are contiguous.
    fn message(&self) -> (String, Vec<Value>, Vec<Value>) {
        let mut text = String::new();
        let mut calls: Vec<Value> = vec![];
        let mut finishes = vec![];
        for c in &self.chunks {
            for ch in c["choices"].as_array().unwrap() {
                if !ch["finish_reason"].is_null() {
                    finishes.push(ch["finish_reason"].clone());
                }
                let d = &ch["delta"];
                if let Some(t) = d["content"].as_str() {
                    text.push_str(t);
                }
                for tc in d["tool_calls"].as_array().cloned().unwrap_or_default() {
                    let i = tc["index"].as_u64().unwrap() as usize;
                    if i == calls.len() {
                        calls.push(json!({"id":tc["id"],"type":"function","function":{"name":tc["function"]["name"],"arguments":""}}));
                    }
                    assert!(i < calls.len(), "tool call indices contiguous");
                    let a = calls[i]["function"]["arguments"].as_str().unwrap().to_string()
                        + tc["function"]["arguments"].as_str().unwrap_or("");
                    calls[i]["function"]["arguments"] = json!(a);
                }
            }
        }
        (text, calls, finishes)
    }
    fn usage(&self) -> Vec<&Value> {
        self.chunks.iter().filter(|c| c.get("usage").is_some_and(|u| !u.is_null())).map(|c| &c["usage"]).collect()
    }
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect()
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn stream_intercepts_merges_and_expands_next_request() {
    let remember = json!({"content": "We deploy the alpha service through ops/deploy.sh, never by hand", "kind": "procedure"});
    let e = env(vec![
        sse_script("c1", "Let me save that.", vec![("call_mem1", "unimem_remember", remember.clone())], "tool_calls", 100, 30),
        sse_script("c2", "Saved it.", vec![], "stop", 140, 7),
        Reply::Json(completion("c3", text_msg("ok"), "stop", 5, 1)),
    ])
    .await;
    let u1 = json!({"role":"user","content":"Remember: deploy alpha via ops/deploy.sh"});
    let req1 = agent_req(json!([sys(), u1.clone()]), true);
    let s = Stream::read(e.post(&req1).await).await;
    assert!(s.done, "stream ends with [DONE]");
    assert!(!s.chunks.iter().any(|c| c.to_string().contains("unimem_")));
    let (text, calls, finishes) = s.message();
    assert_eq!(text, "Let me save that.\n\nSaved it.");
    assert!(calls.is_empty());
    assert_eq!(finishes, vec![json!("stop")], "one merged finish_reason");
    assert!(s.chunks.iter().all(|c| c["id"] == "c1"), "one completion id");
    let u = s.usage();
    assert_eq!(u.len(), 1);
    assert_eq!(u[0]["prompt_tokens"], 140, "last round's prompt = the real context");
    assert_eq!(u[0]["completion_tokens"], 37);
    assert_eq!(s.chunks.last().unwrap()["usage"]["total_tokens"], 177);

    let seen = e.seen();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].path, "/v1/chat/completions");
    assert_eq!(seen[0].headers["authorization"], "Bearer sk-test");
    let b1 = seen[0].body();
    let b2 = seen[1].body();
    assert_eq!(b1["messages"][0], sys(), "system untouched");
    assert_eq!(tool_names(&b1), vec!["bash", "unimem_recall", "unimem_remember", "unimem_forget", "unimem_project"]);
    assert_eq!(b1["tools"], b2["tools"]);
    assert!(b1["messages"][1]["content"].as_str().unwrap().contains("<unimem-context>"));
    let m2 = b2["messages"].as_array().unwrap();
    assert_eq!(m2.len(), 4);
    assert_eq!(m2[1], b1["messages"][1]);
    assert_eq!(m2[2]["role"], "assistant");
    assert_eq!(m2[2]["content"], "Let me save that.");
    assert_eq!(m2[2]["tool_calls"][0]["function"]["name"], "unimem_remember");
    assert_eq!(m2[2]["tool_calls"][0]["function"]["arguments"], remember.to_string());
    assert_eq!(m2[3]["role"], "tool");
    assert_eq!(m2[3]["tool_call_id"], "call_mem1");
    assert!(e.engine.store.count_memories().values().sum::<i64>() >= 1);

    // next request echoes the merged message
    let req2 =
        agent_req(json!([sys(), u1, {"role":"assistant","content": text}, {"role":"user","content":"thanks, now run the tests"}]), false);
    assert!(e.post(&req2).await.status().is_success());
    let seen = e.seen();
    let b3 = seen[2].body();
    let m3 = b3["messages"].as_array().unwrap();
    assert_eq!(m3.len(), 6, "{}", serde_json::to_string_pretty(m3).unwrap());
    assert_eq!(m3[1], b1["messages"][1], "augment replayed byte-identically");
    assert_eq!(m3[2], m2[2], "hidden assistant restored");
    assert_eq!(m3[3], m2[3], "hidden tool result restored");
    assert_eq!(m3[4], json!({"role":"assistant","content":"Saved it."}));
    assert_eq!(m3[5]["role"], "user");
    assert_eq!(b3["messages"][0], sys());
    let raw_tools = |c: &Captured| {
        let s = String::from_utf8(c.raw.to_vec()).unwrap();
        let i = s.find("\"tools\"").unwrap();
        s[i..i + 3000].to_string()
    };
    assert_eq!(raw_tools(&seen[0]), raw_tools(&seen[2]), "tool definitions byte-stable");
}

#[tokio::test]
async fn non_stream_merges_rounds_and_sums_usage() {
    let e = env(vec![
        Reply::Json(completion("n1", json!({"role":"assistant","content":"Checking memory.","tool_calls":[call("call_r1","unimem_recall",&json!({"query":"deploy"}))]}), "tool_calls", 100, 10)),
        Reply::Json(completion("n2", text_msg("Nothing recorded."), "stop", 120, 7)),
        Reply::Json(completion("n3", text_msg("ok"), "stop", 1, 1)),
    ])
    .await;
    let u1 = json!({"role":"user","content":"how do we deploy?"});
    let v: Value = e.post(&agent_req(json!([u1.clone()]), false)).await.json().await.unwrap();
    assert_eq!(v["id"], "n1");
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], "Checking memory.\n\nNothing recorded.");
    assert!(m.get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 120);
    assert_eq!(v["usage"]["completion_tokens"], 17);
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 60);

    let b2 = e.seen()[1].body();
    assert_eq!(b2["messages"][2]["tool_call_id"], "call_r1");
    assert!(b2["messages"][2]["content"].as_str().unwrap().contains("sufficient: false"));

    e.post(&agent_req(json!([u1, m, {"role":"user","content":"ok"}]), false)).await.bytes().await.unwrap();
    let b3 = e.seen()[2].body();
    let m3 = b3["messages"].as_array().unwrap();
    assert_eq!(m3.len(), 5);
    assert_eq!(m3[1], b2["messages"][1]);
    assert_eq!(m3[2], b2["messages"][2]);
    assert_eq!(m3[3]["content"], "Nothing recorded.");
}

#[tokio::test]
async fn mixed_turn_hands_harness_calls_and_replays_hidden_results() {
    let e = env(vec![
        sse_script(
            "x1",
            "Looking.",
            vec![("call_mem", "unimem_recall", json!({"query":"tests"})), ("call_sh", "bash", json!({"command":"cargo test"}))],
            "tool_calls",
            50,
            20,
        ),
        Reply::Json(completion("x2", text_msg("done"), "stop", 1, 1)),
    ])
    .await;
    let u1 = json!({"role":"user","content":"run the tests"});
    let s = Stream::read(e.post(&agent_req(json!([u1.clone()]), true)).await).await;
    let (text, calls, finishes) = s.message();
    assert_eq!(text, "Looking.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_sh");
    assert_eq!(calls[0]["function"]["arguments"], json!({"command":"cargo test"}).to_string());
    assert_eq!(finishes, vec![json!("tool_calls")]);
    assert_eq!(s.usage()[0]["completion_tokens"], 20);
    assert!(!s.chunks.iter().any(|c| c.to_string().contains("unimem_")));
    assert_eq!(e.seen().len(), 1, "no continuation on mixed turns");

    let harness_res = json!({"role":"tool","tool_call_id":"call_sh","content":"ok: 3 passed"});
    let req = agent_req(json!([u1, {"role":"assistant","content":text,"tool_calls":calls}, harness_res.clone()]), true);
    let _ = e.post(&req).await.bytes().await;
    let m = e.seen()[1].body()["messages"].as_array().unwrap().clone();
    assert_eq!(m.len(), 4, "{}", serde_json::to_string_pretty(&m).unwrap());
    let ids: Vec<&str> = m[1]["tool_calls"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["call_mem", "call_sh"]);
    assert_eq!(m[2]["role"], "tool");
    assert_eq!(m[2]["tool_call_id"], "call_mem");
    assert_eq!(m[3], harness_res);
}

#[tokio::test]
async fn requests_without_tools_pass_through_byte_identical() {
    let e = env(vec![Reply::Json(completion("t", text_msg("Title"), "stop", 1, 1))]).await;
    let raw = r#"{"model":"gpt-x",  "messages":[{"role":"user","content":"title for: fix bug"}]}"#;
    let r = e
        .client
        .post(format!("{}/v1/chat/completions", e.gw))
        .header("authorization", "Bearer k")
        .header("content-type", "application/json")
        .body(raw)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    assert_eq!(r.json::<Value>().await.unwrap()["choices"][0]["message"]["content"], "Title");
    assert_eq!(e.seen()[0].raw.as_ref(), raw.as_bytes());
}

#[tokio::test]
async fn upstream_errors_are_relayed() {
    let err = json!({"error":{"type":"invalid_request_error","message":"bad"}});
    let e = env(vec![Reply::Status(429, err.clone()), Reply::Status(400, err.clone())]).await;
    let r = e.post(&agent_req(json!([{"role":"user","content":"hi there"}]), true)).await;
    assert_eq!(r.status().as_u16(), 429);
    assert_eq!(r.json::<Value>().await.unwrap(), err);
    let r = e.post(&agent_req(json!([{"role":"user","content":"hi again"}]), false)).await;
    assert_eq!(r.status().as_u16(), 400);
    assert_eq!(r.json::<Value>().await.unwrap(), err);
}

#[tokio::test]
async fn continuation_error_ends_stream_cleanly() {
    let e = env(vec![
        sse_script("f1", "Looking it up.", vec![("call_x", "unimem_recall", json!({"query":"q"}))], "tool_calls", 10, 4),
        Reply::Status(500, json!({"error":{"message":"boom"}})),
    ])
    .await;
    let s = Stream::read(e.post(&agent_req(json!([{"role":"user","content":"look it up"}]), true)).await).await;
    assert!(s.done);
    let (text, calls, finishes) = s.message();
    assert_eq!(text, "Looking it up.");
    assert!(calls.is_empty());
    assert_eq!(finishes, vec![json!("stop")]);
    assert_eq!(s.usage()[0]["completion_tokens"], 4);
}

#[tokio::test]
async fn tool_result_turns_are_not_augmented() {
    let ok = || Reply::Json(completion("o", text_msg("ok"), "stop", 1, 1));
    let e = env(vec![ok(), ok()]).await;
    e.post(&agent_req(json!([{"role":"user","content":"please fix the parser"}]), false)).await.bytes().await.unwrap();
    let tr = json!([{"role":"user","content":"please fix the parser"},
        {"role":"assistant","content":null,"tool_calls":[call("call_b","bash",&json!({}))]},
        {"role":"tool","tool_call_id":"call_b","content":"x"}]);
    e.post(&agent_req(tr, false)).await.bytes().await.unwrap();
    let seen = e.seen();
    let b = seen[1].body();
    assert_eq!(b["messages"][0], seen[0].body()["messages"][0], "augment replayed");
    assert_eq!(b["messages"][2], json!({"role":"tool","tool_call_id":"call_b","content":"x"}));
}
