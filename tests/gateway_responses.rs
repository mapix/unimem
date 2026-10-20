//! End-to-end tests of the OpenAI Responses dialect against a scripted mock upstream.

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
    let reply = m.script.lock().pop_front().unwrap_or_else(|| Reply::Json(resp_obj("resp_default", vec![msg_item("msg_d", "ok")], 1, 1)));
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
    let up = serve(Router::new().route("/v1/responses", post(mock_handler)).with_state(mock.clone())).await;
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
            .post(format!("{}/v1/responses", self.gw))
            .header("authorization", "Bearer sk-test")
            .header("originator", "codex_cli_rs")
            .header("session_id", "0199aaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
            .json(body)
            .send()
            .await
            .unwrap()
    }
}

// ---------------------------------------------------------------- script builders

fn msg_item(id: &str, text: &str) -> Value {
    json!({"type":"message","id":id,"status":"completed","role":"assistant","content":[{"type":"output_text","text":text,"annotations":[]}]})
}
fn reasoning_item(id: &str, enc: &str) -> Value {
    json!({"type":"reasoning","id":id,"summary":[{"type":"summary_text","text":"thinking"}],"encrypted_content":enc})
}
fn call_item(id: &str, call_id: &str, name: &str, args: &Value) -> Value {
    json!({"type":"function_call","id":id,"status":"completed","call_id":call_id,"name":name,"arguments":args.to_string()})
}
fn resp_obj(id: &str, output: Vec<Value>, inp: u64, out: u64) -> Value {
    json!({"id":id,"object":"response","created_at":1700000000,"status":"completed","model":"gpt-x","output":output,
           "usage":{"input_tokens":inp,"input_tokens_details":{"cached_tokens":inp / 2},"output_tokens":out,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":inp + out}})
}

/// A full SSE stream for a response with the given output items.
fn sse_script(id: &str, items: Vec<Value>, inp: u64, out: u64) -> Reply {
    let mut evs: Vec<Value> = vec![];
    let mut pending = resp_obj(id, vec![], 0, 0);
    pending["status"] = json!("in_progress");
    pending["usage"] = Value::Null;
    evs.push(json!({"type":"response.created","response":pending}));
    evs.push(json!({"type":"response.in_progress","response":pending}));
    for (i, it) in items.iter().enumerate() {
        let mut added = it.clone();
        match it["type"].as_str().unwrap() {
            "message" => {
                added["content"] = json!([]);
                added["status"] = json!("in_progress");
                evs.push(json!({"type":"response.output_item.added","output_index":i,"item":added}));
                let t = it["content"][0]["text"].as_str().unwrap();
                evs.push(json!({"type":"response.content_part.added","output_index":i,"item_id":it["id"],"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}));
                let (a, b) = t.split_at(t.len() / 2);
                for d in [a, b] {
                    evs.push(json!({"type":"response.output_text.delta","output_index":i,"item_id":it["id"],"content_index":0,"delta":d}));
                }
                evs.push(json!({"type":"response.output_text.done","output_index":i,"item_id":it["id"],"content_index":0,"text":t}));
                evs.push(json!({"type":"response.content_part.done","output_index":i,"item_id":it["id"],"content_index":0,"part":it["content"][0]}));
            }
            "function_call" => {
                added["arguments"] = json!("");
                added["status"] = json!("in_progress");
                evs.push(json!({"type":"response.output_item.added","output_index":i,"item":added}));
                let s = it["arguments"].as_str().unwrap();
                let (a, b) = s.split_at(s.len() / 2);
                for d in [a, b] {
                    evs.push(json!({"type":"response.function_call_arguments.delta","output_index":i,"item_id":it["id"],"delta":d}));
                }
                evs.push(json!({"type":"response.function_call_arguments.done","output_index":i,"item_id":it["id"],"arguments":s}));
            }
            _ => {
                evs.push(json!({"type":"response.output_item.added","output_index":i,"item":added}));
            }
        }
        evs.push(json!({"type":"response.output_item.done","output_index":i,"item":it}));
    }
    evs.push(json!({"type":"response.completed","response":resp_obj(id, items, inp, out)}));
    for (n, e) in evs.iter_mut().enumerate() {
        e["sequence_number"] = json!(n);
    }
    Reply::Sse(evs)
}

const INSTRUCTIONS: &str = "You are Codex, a coding agent.";

fn env_ctx() -> Value {
    json!({"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>\n  <cwd>/tmp/unimem-test-nonexistent</cwd>\n</environment_context>"}]})
}
fn user(t: &str) -> Value {
    json!({"type":"message","role":"user","content":[{"type":"input_text","text":t}]})
}
fn harness_tools() -> Value {
    json!([{"type":"function","name":"shell","description":"run a command","strict":false,"parameters":{"type":"object","properties":{"command":{"type":"array","items":{"type":"string"}}}}}])
}
fn agent_req(input: Value, stream: bool) -> Value {
    json!({"model":"gpt-x","instructions":INSTRUCTIONS,"input":input,"tools":harness_tools(),"tool_choice":"auto","parallel_tool_calls":false,
           "reasoning":{"effort":"medium"},"store":false,"stream":stream,"include":["reasoning.encrypted_content"],"prompt_cache_key":"pck-1"})
}

struct Stream {
    events: Vec<(String, Value)>,
    raw: String,
}

impl Stream {
    async fn read(r: reqwest::Response) -> Self {
        assert!(r.status().is_success(), "status {}", r.status());
        let bytes = r.bytes().await.unwrap();
        let mut p = sse::Parser::default();
        let events = p.push(&bytes).into_iter().map(|e| (e.event.clone().unwrap_or_default(), e.json().unwrap())).collect();
        Self { events, raw: String::from_utf8_lossy(&bytes).into_owned() }
    }
    fn of(&self, ty: &str) -> Vec<&Value> {
        self.events.iter().filter(|e| e.0 == ty).map(|e| &e.1).collect()
    }
    fn assert_well_formed(&self) {
        for (i, (name, v)) in self.events.iter().enumerate() {
            assert_eq!(name, v["type"].as_str().unwrap());
            assert_eq!(v["sequence_number"].as_u64().unwrap(), i as u64, "sequence numbers contiguous");
        }
        let added: Vec<u64> = self.of("response.output_item.added").iter().map(|v| v["output_index"].as_u64().unwrap()).collect();
        assert_eq!(added, (0..added.len() as u64).collect::<Vec<_>>(), "output indices contiguous");
        for (_, v) in &self.events {
            if let Some(i) = v["output_index"].as_u64() {
                assert!(i < added.len() as u64);
            }
        }
        assert_eq!(self.of("response.created").len(), 1);
        assert_eq!(self.of("response.in_progress").len(), 1);
        assert_eq!(self.of("response.completed").len(), 1);
        assert_eq!(self.events.last().unwrap().0, "response.completed");
    }
    /// The output items as a client would collect them from `output_item.done`.
    fn items(&self) -> Vec<Value> {
        self.of("response.output_item.done").into_iter().map(|v| v["item"].clone()).collect()
    }
    fn completed(&self) -> &Value {
        &self.of("response.completed")[0]["response"]
    }
}

fn mentions_unimem_call(s: &Stream) -> bool {
    s.events.iter().any(|e| e.1.to_string().contains("unimem_"))
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect()
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn stream_intercepts_merges_and_expands_next_request() {
    let remember = json!({"content": "We deploy the alpha service through ops/deploy.sh, never by hand", "kind": "procedure"});
    let e = env(vec![
        sse_script(
            "resp_1",
            vec![
                reasoning_item("rs_1", "ENC-1"),
                msg_item("msg_1", "Let me save that."),
                call_item("fc_1", "call_mem1", "unimem_remember", &remember),
            ],
            100,
            30,
        ),
        sse_script("resp_2", vec![msg_item("msg_2", "Saved it.")], 140, 7),
        Reply::Json(resp_obj("resp_3", vec![msg_item("msg_3", "ok")], 5, 1)),
    ])
    .await;

    let u1 = user("Remember: deploy alpha via ops/deploy.sh");
    let req1 = agent_req(json!([env_ctx(), u1.clone()]), true);
    let s = Stream::read(e.post(&req1).await).await;
    s.assert_well_formed();
    assert!(!mentions_unimem_call(&s), "no unimem item reaches the client");
    let items = s.items();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["type"], "reasoning");
    assert_eq!(items[1]["content"][0]["text"], "Let me save that.");
    assert_eq!(items[2]["content"][0]["text"], "Saved it.");
    let done = s.completed();
    assert_eq!(done["id"], "resp_1", "merged response keeps the first id");
    assert_eq!(done["output"].as_array().unwrap(), &items);
    // honest usage: last round's input side, summed output; true totals in `x_unimem_usage`
    assert_eq!(done["usage"]["input_tokens"], 140);
    assert_eq!(done["usage"]["output_tokens"], 37);
    assert_eq!(done["usage"]["input_tokens_details"]["cached_tokens"], 70);
    assert_eq!(done["usage"]["output_tokens_details"]["reasoning_tokens"], 2);
    assert_eq!(done["usage"]["total_tokens"], 177);
    assert!(s.raw.contains(": x-unimem-usage {"), "true totals as an SSE comment");
    assert!(s.raw.contains("\"input_tokens\":240"));
    // text deltas of round 2 carry the renumbered index
    let deltas: Vec<u64> = s.of("response.output_text.delta").iter().map(|v| v["output_index"].as_u64().unwrap()).collect();
    assert_eq!(deltas, vec![1, 1, 2, 2]);

    let seen = e.seen();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].path, "/v1/responses");
    assert_eq!(seen[0].headers["authorization"], "Bearer sk-test");
    let b1 = seen[0].body();
    let b2 = seen[1].body();
    assert_eq!(b1["instructions"], INSTRUCTIONS, "instructions untouched");
    assert_eq!(tool_names(&b1), vec!["shell", "unimem_recall", "unimem_remember", "unimem_forget", "unimem_project"]);
    assert_eq!(b1["tools"], b2["tools"]);
    // the prompt got the session preamble as an extra input_text part
    let p = b1["input"][1]["content"].as_array().unwrap();
    assert_eq!(p.len(), 2);
    assert_eq!(p[1]["type"], "input_text");
    assert!(p[1]["text"].as_str().unwrap().contains("<unimem-context>"));
    // continuation = input + round-1 items (ids stripped, store=false) + function_call_output
    let in2 = b2["input"].as_array().unwrap();
    assert_eq!(in2.len(), 6, "{}", serde_json::to_string_pretty(in2).unwrap());
    assert_eq!(in2[..2], b1["input"].as_array().unwrap()[..]);
    assert_eq!(in2[2]["type"], "reasoning");
    assert_eq!(in2[2]["encrypted_content"], "ENC-1");
    assert!(in2[2].get("id").is_none());
    assert_eq!(in2[4]["name"], "unimem_remember");
    assert_eq!(in2[5]["type"], "function_call_output");
    assert_eq!(in2[5]["call_id"], "call_mem1");
    assert!(in2[5]["output"].as_str().unwrap().len() > 3);
    assert!(e.engine.store.count_memories().values().sum::<i64>() >= 1, "remember executed in gateway");

    // next request: Codex echoes the visible items, then a new prompt
    let mut input3 = vec![env_ctx(), u1];
    input3.extend(items.clone());
    input3.push(user("thanks, now run the tests"));
    let r = e.post(&agent_req(Value::Array(input3), false)).await;
    assert!(r.status().is_success());
    let b3 = e.seen()[2].body();
    let in3 = b3["input"].as_array().unwrap();
    assert_eq!(in3.len(), 8, "{}", serde_json::to_string_pretty(in3).unwrap());
    assert_eq!(in3[1], b1["input"][1], "augment overlay replayed byte-identically");
    assert_eq!(in3[2], items[0]);
    assert_eq!(in3[3], items[1]);
    assert_eq!(in3[4], in2[4], "hidden function_call restored");
    assert_eq!(in3[5], in2[5], "hidden function_call_output restored");
    assert_eq!(in3[6], items[2]);
    assert_eq!(in3[7]["content"][0]["text"], "thanks, now run the tests");
    assert_eq!(b3["instructions"], INSTRUCTIONS);
    let raw_tools = |c: &Captured| {
        let s = String::from_utf8(c.raw.to_vec()).unwrap();
        let i = s.find("\"tools\"").unwrap();
        s[i..i + 3000].to_string()
    };
    assert_eq!(raw_tools(&seen[0]), raw_tools(&e.seen()[2]), "tool definitions byte-stable");
}

#[tokio::test]
async fn non_stream_merges_rounds_and_sums_usage() {
    let e = env(vec![
        Reply::Json(resp_obj(
            "resp_a",
            vec![msg_item("m1", "Checking memory."), call_item("fc_r", "call_r1", "unimem_recall", &json!({"query":"deploy"}))],
            100,
            10,
        )),
        Reply::Json(resp_obj("resp_b", vec![msg_item("m2", "Nothing recorded.")], 120, 7)),
        Reply::Json(resp_obj("resp_c", vec![msg_item("m3", "ok")], 1, 1)),
    ])
    .await;
    let u1 = user("how do we deploy?");
    let r = e.post(&agent_req(json!([u1.clone()]), false)).await;
    assert!(r.status().is_success());
    let v: Value = r.json().await.unwrap();
    let out = v["output"].as_array().unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["content"][0]["text"], "Checking memory.");
    assert_eq!(out[1]["content"][0]["text"], "Nothing recorded.");
    assert_eq!(v["id"], "resp_a");
    assert_eq!(v["usage"]["input_tokens"], 120);
    assert_eq!(v["usage"]["output_tokens"], 17);
    assert!(!v.to_string().contains("unimem_"));

    let b2 = e.seen()[1].body();
    let in2 = b2["input"].as_array().unwrap();
    assert_eq!(in2.len(), 4);
    assert_eq!(in2[3]["call_id"], "call_r1");
    assert!(in2[3]["output"].as_str().unwrap().contains("sufficient: false"));

    let mut input = vec![u1];
    input.extend(out.iter().cloned());
    input.push(user("ok"));
    e.post(&agent_req(Value::Array(input), false)).await.bytes().await.unwrap();
    let in3 = e.seen()[2].body()["input"].as_array().unwrap().clone();
    assert_eq!(in3.len(), 6);
    assert_eq!(in3[2], in2[2]);
    assert_eq!(in3[3], in2[3]);
    assert_eq!(in3[4]["content"][0]["text"], "Nothing recorded.");
}

#[tokio::test]
async fn mixed_turn_hands_harness_calls_and_replays_hidden_items() {
    let shell_args = json!({"command":["cargo","test"]});
    let e = env(vec![
        sse_script(
            "resp_m",
            vec![
                msg_item("m1", "Looking."),
                call_item("fc_mem", "call_mem", "unimem_recall", &json!({"query":"tests"})),
                call_item("fc_sh", "call_sh", "shell", &shell_args),
            ],
            50,
            20,
        ),
        Reply::Json(resp_obj("resp_n", vec![msg_item("m2", "done")], 1, 1)),
    ])
    .await;
    let u1 = user("run the tests");
    let s = Stream::read(e.post(&agent_req(json!([u1.clone()]), true)).await).await;
    s.assert_well_formed();
    assert!(!mentions_unimem_call(&s));
    let items = s.items();
    assert_eq!(items.len(), 2);
    assert_eq!(items[1]["name"], "shell");
    assert_eq!(items[1]["arguments"], shell_args.to_string());
    let args_deltas: Vec<u64> =
        s.of("response.function_call_arguments.delta").iter().map(|v| v["output_index"].as_u64().unwrap()).collect();
    assert_eq!(args_deltas, vec![1, 1]);
    assert_eq!(s.completed()["usage"]["output_tokens"], 20);
    assert_eq!(e.seen().len(), 1, "no continuation on mixed turns");

    let harness_out = json!({"type":"function_call_output","call_id":"call_sh","output":"ok: 3 passed"});
    let mut input = vec![u1];
    input.extend(items);
    input.push(harness_out.clone());
    let _ = e.post(&agent_req(Value::Array(input), true)).await.bytes().await;
    let in2 = e.seen()[1].body()["input"].as_array().unwrap().clone();
    let shape: Vec<String> =
        in2.iter().map(|i| format!("{}:{}", i["type"].as_str().unwrap(), i["call_id"].as_str().unwrap_or(""))).collect();
    assert_eq!(
        shape,
        vec![
            "message:",
            "message:",
            "function_call:call_sh",
            "function_call:call_mem",
            "function_call_output:call_mem",
            "function_call_output:call_sh"
        ]
    );
    assert_eq!(in2[3]["arguments"], json!({"query":"tests"}).to_string());
    assert_eq!(in2[5], harness_out);
}

#[tokio::test]
async fn requests_without_tools_pass_through_byte_identical() {
    let e = env(vec![Reply::Json(resp_obj("r", vec![msg_item("m", "Title")], 1, 1))]).await;
    let raw = r#"{"model":"gpt-x",  "input":"title for: fix bug","instructions":"be brief"}"#;
    let r = e
        .client
        .post(format!("{}/v1/responses", e.gw))
        .header("authorization", "Bearer k")
        .header("content-type", "application/json")
        .body(raw)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["output"][0]["content"][0]["text"], "Title");
    let seen = e.seen();
    assert_eq!(seen[0].raw.as_ref(), raw.as_bytes());
    assert_eq!(seen[0].headers["authorization"], "Bearer k");
}

#[tokio::test]
async fn compressed_bodies_are_decoded_and_forwarded_plain() {
    let e = env(vec![Reply::Json(resp_obj("r", vec![msg_item("m", "hi")], 1, 1))]).await;
    let body = agent_req(json!([user("hello there")]), false);
    let z = zstd::encode_all(&serde_json::to_vec(&body).unwrap()[..], 3).unwrap();
    let r = e
        .client
        .post(format!("{}/v1/responses", e.gw))
        .header("content-encoding", "zstd")
        .header("content-type", "application/json")
        .body(z)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    let seen = e.seen();
    assert!(seen[0].headers.get("content-encoding").is_none());
    assert_eq!(tool_names(&seen[0].body()).len(), 5);
}

#[tokio::test]
async fn upstream_errors_are_relayed() {
    let err = json!({"error":{"type":"rate_limit_error","message":"slow down","code":"rate_limit_exceeded"}});
    let e = env(vec![Reply::Status(429, err.clone()), Reply::Status(400, err.clone())]).await;
    let r = e.post(&agent_req(json!([user("hi there")]), true)).await;
    assert_eq!(r.status().as_u16(), 429);
    assert_eq!(r.json::<Value>().await.unwrap(), err);
    let r = e.post(&agent_req(json!([user("hi again")]), false)).await;
    assert_eq!(r.status().as_u16(), 400);
    assert_eq!(r.json::<Value>().await.unwrap(), err);
}

#[tokio::test]
async fn continuation_error_ends_stream_cleanly() {
    let e = env(vec![
        sse_script(
            "resp_x",
            vec![msg_item("m1", "Looking it up."), call_item("fc_x", "call_x", "unimem_recall", &json!({"query":"q"}))],
            10,
            4,
        ),
        Reply::Status(500, json!({"error":{"message":"boom"}})),
    ])
    .await;
    let s = Stream::read(e.post(&agent_req(json!([user("look it up")]), true)).await).await;
    s.assert_well_formed();
    assert_eq!(s.items().len(), 1);
    assert_eq!(s.completed()["output"].as_array().unwrap().len(), 1);
    assert_eq!(s.completed()["status"], "completed");
    assert!(!mentions_unimem_call(&s));
}

#[tokio::test]
async fn store_with_previous_response_id_does_not_replay() {
    let e = env(vec![
        Reply::Json(resp_obj(
            "resp_s1",
            vec![
                msg_item("m1", "Saving."),
                call_item("fc_s", "call_s", "unimem_remember", &json!({"content":"The CI runs on Buildkite, not GitHub Actions"})),
            ],
            10,
            2,
        )),
        Reply::Json(resp_obj("resp_s2", vec![msg_item("m2", "Saved.")], 12, 3)),
    ])
    .await;
    let mut req = agent_req(json!([user("note: CI is on Buildkite")]), false);
    req["store"] = json!(true);
    req["previous_response_id"] = json!("resp_prev");
    let v: Value = e.post(&req).await.json().await.unwrap();
    assert_eq!(v["id"], "resp_s2", "chained responses keep the last server-side id");
    assert_eq!(v["output"].as_array().unwrap().len(), 2);
    let seen = e.seen();
    let b1 = seen[0].body();
    assert_eq!(b1["previous_response_id"], "resp_prev");
    assert_eq!(tool_names(&b1).len(), 5, "tools still injected");
    let b2 = seen[1].body();
    assert_eq!(b2["previous_response_id"], "resp_s1");
    assert_eq!(b2["input"].as_array().unwrap().len(), 1);
    assert_eq!(b2["input"][0]["type"], "function_call_output");
}

#[tokio::test]
async fn tool_output_turns_are_not_augmented_and_augment_is_stable_on_retry() {
    let ok = || Reply::Json(resp_obj("r", vec![msg_item("m", "ok")], 1, 1));
    let e = env(vec![ok(), ok(), ok()]).await;
    let input = json!([env_ctx(), user("please fix the parser")]);
    e.post(&agent_req(input.clone(), false)).await.bytes().await.unwrap();
    e.post(&agent_req(input.clone(), false)).await.bytes().await.unwrap();
    let seen = e.seen();
    assert_eq!(seen[0].body()["input"], seen[1].body()["input"], "retry gets identical augmentation");
    let tr = json!([env_ctx(), user("please fix the parser"),
        {"type":"function_call","call_id":"call_b","name":"shell","arguments":"{}"},
        {"type":"function_call_output","call_id":"call_b","output":"x"}]);
    e.post(&agent_req(tr, false)).await.bytes().await.unwrap();
    let b = e.seen()[2].body();
    assert_eq!(b["input"][1], seen[0].body()["input"][1]);
    assert_eq!(b["input"][3], json!({"type":"function_call_output","call_id":"call_b","output":"x"}), "tool output turn not augmented");
    assert_eq!(b["input"].as_array().unwrap().len(), 4);
}
