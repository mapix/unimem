use serde_json::{json, Value};
use std::sync::Arc;

use unimem::config::Config;
use unimem::decide::{ChainDecider, HeuristicDecider};
use unimem::embed::HashEmbedder;
use unimem::engine::Engine;
use unimem::gateway::{router, AppState};
use unimem::store::Store;

async fn start() -> String {
    let engine = Engine::with_parts(
        Config::default(),
        Store::open_memory().unwrap(),
        Box::new(HashEmbedder { dim: 512 }),
        ChainDecider::new(vec![Box::new(HeuristicDecider)]),
    );
    let st = Arc::new(AppState { engine: Arc::new(engine), client: reqwest::Client::new() });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router(st)).await.unwrap() });
    format!("http://{addr}/mcp")
}

async fn post(c: &reqwest::Client, url: &str, sid: Option<&str>, body: Value) -> (u16, Option<String>, Option<Value>) {
    let mut rb = c.post(url).header("accept", "application/json, text/event-stream").json(&body);
    if let Some(s) = sid {
        rb = rb.header("mcp-session-id", s);
    }
    let r = rb.send().await.unwrap();
    let status = r.status().as_u16();
    let sid = r.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()).map(String::from);
    let text = r.text().await.unwrap();
    (status, sid, if text.is_empty() { None } else { Some(serde_json::from_str(&text).unwrap()) })
}

fn call(id: i64, name: &str, args: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": args}})
}

#[tokio::test]
async fn mcp_roundtrip() {
    let url = start().await;
    let c = reqwest::Client::new();

    let (s, sid, v) = post(
        &c,
        &url,
        None,
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "claude-code", "version": "2.0"}}}),
    )
    .await;
    assert_eq!(s, 200);
    let sid = sid.expect("session id header");
    let v = v.unwrap();
    assert_eq!(v["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(v["result"]["serverInfo"]["name"], "unimem");
    assert!(v["result"]["capabilities"]["tools"].is_object());
    assert!(v["result"]["instructions"].as_str().unwrap().contains("unimem_recall"));

    let (s, _, v) = post(&c, &url, Some(&sid), json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await;
    assert_eq!(s, 202);
    assert!(v.is_none());

    let (_, _, v) = post(&c, &url, Some(&sid), json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})).await;
    let tools = v.unwrap()["result"]["tools"].as_array().unwrap().clone();
    assert_eq!(tools.len(), 4);
    assert!(tools.iter().all(|t| t["inputSchema"]["properties"]["cwd"].is_object()));

    let (_, _, v) = post(&c, &url, Some(&sid), call(3, "unimem_remember", json!({"content": "The staging database is reset every Sunday at 03:00 UTC by the cron job db-reset", "scope": "global"}))).await;
    let v = v.unwrap();
    assert_eq!(v["result"]["isError"], false);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("Saved "), "{text}");

    let (_, _, v) = post(&c, &url, Some(&sid), call(4, "unimem_recall", json!({"query": "when is the staging database reset?"}))).await;
    let text = v.unwrap()["result"]["content"][0]["text"].as_str().unwrap().to_string();
    assert!(text.contains("every Sunday"), "{text}");

    let (_, _, v) = post(&c, &url, Some(&sid), json!({"jsonrpc": "2.0", "id": 5, "method": "ping"})).await;
    assert_eq!(v.unwrap()["result"], json!({}));

    // GET is not supported (no server-initiated stream).
    let r = c.get(&url).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 405);
}

#[tokio::test]
async fn mcp_errors_and_batch() {
    let url = start().await;
    let c = reqwest::Client::new();

    let (s, _, v) = post(&c, &url, None, json!({"jsonrpc": "2.0", "id": 9, "method": "no/such"})).await;
    assert_eq!(s, 200);
    let v = v.unwrap();
    assert_eq!(v["id"], 9);
    assert_eq!(v["error"]["code"], -32601);

    let (_, _, v) = post(
        &c,
        &url,
        None,
        json!([
            {"jsonrpc": "2.0", "id": "a", "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": "b", "method": "tools/list"},
            {"jsonrpc": "2.0", "id": "c", "method": "bogus"}
        ]),
    )
    .await;
    let arr = v.unwrap().as_array().unwrap().clone();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["id"], "a");
    assert_eq!(arr[1]["result"]["tools"].as_array().unwrap().len(), 4);
    assert_eq!(arr[2]["error"]["code"], -32601);

    let (_, _, v) = post(&c, &url, None, call(10, "rm_rf", json!({}))).await;
    assert_eq!(v.unwrap()["error"]["code"], -32602);
}
