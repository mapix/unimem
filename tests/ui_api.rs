//! The web UI: the embedded page and every `/ui/api/*` endpoint, over an in-memory engine seeded with the demo data.

use std::sync::Arc;

use serde_json::{json, Value};

use unimem::config::Config;
use unimem::decide::{ChainDecider, HeuristicDecider};
use unimem::embed::HashEmbedder;
use unimem::engine::Engine;
use unimem::gateway::{self, AppState};
use unimem::store::Store;

struct Env {
    base: String,
    http: reqwest::Client,
    engine: Arc<Engine>,
    _dir: tempfile::TempDir,
}

async fn env() -> Env {
    let engine = Arc::new(Engine::with_parts(
        Config::default(),
        Store::open_memory().unwrap(),
        Box::new(HashEmbedder { dim: 512 }),
        ChainDecider::new(vec![Box::new(HeuristicDecider)]),
    ));
    let dir = tempfile::tempdir().unwrap();
    let n = gateway::ui::demo_seed(&engine, dir.path()).await.unwrap();
    assert!(n >= 40, "demo seed wrote {n}");
    let state = Arc::new(AppState { engine: engine.clone(), client: reqwest::Client::new() });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, gateway::router(state)).await.unwrap() });
    Env { base: format!("http://{addr}"), http: reqwest::Client::new(), engine, _dir: dir }
}

impl Env {
    async fn get(&self, p: &str) -> Value {
        let r = self.http.get(format!("{}{p}", self.base)).send().await.unwrap();
        assert!(r.status().is_success(), "GET {p}: {}", r.status());
        r.json().await.unwrap()
    }
    async fn send(&self, m: reqwest::Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut rb = self.http.request(m, format!("{}{p}", self.base));
        if let Some(b) = body {
            rb = rb.json(&b);
        }
        let r = rb.send().await.unwrap();
        let s = r.status().as_u16();
        (s, r.json().await.unwrap_or(Value::Null))
    }
    async fn find(&self, needle: &str) -> Value {
        let all = self.get("/ui/api/memories?include_superseded=true&limit=2000").await;
        all["items"].as_array().unwrap().iter().find(|m| m["content"].as_str().unwrap().contains(needle)).cloned().unwrap_or_else(|| panic!("no memory containing {needle}"))
    }
}

fn pid_of(e: &Env, name: &str) -> String {
    e.engine.store.list_projects().into_iter().find(|p| p.name == name).unwrap().pid
}

#[tokio::test]
async fn index_page_is_served_self_contained() {
    let e = env().await;
    let r = e.http.get(format!("{}/ui", e.base)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/html"));
    let html = r.text().await.unwrap();
    assert!(html.contains("<title>unimem"));
    assert!(!html.contains("https://"), "the page must not load anything from the network");
    for k in ["preference", "decision", "procedure", "gotcha", "fact", "episode"] {
        assert!(html.contains(&format!("R.{k}")) || html.contains(&format!("{k}(m")), "renderer for {k}");
    }
}

#[tokio::test]
async fn overview_counts_everything() {
    let e = env().await;
    let o = e.get("/ui/api/overview").await;
    assert_eq!(o["live"].as_i64().unwrap() + o["superseded"].as_i64().unwrap(), 46);
    assert_eq!(o["superseded"], 6);
    for k in ["preference", "decision", "procedure", "gotcha", "fact", "episode"] {
        assert!(o["by_kind"][k].as_i64().unwrap() > 0, "kind {k}");
    }
    assert_eq!(o["projects"].as_array().unwrap().len(), 3);
    assert!(o["by_scope"]["global"].as_i64().unwrap() >= 9);
    assert!(o["by_harness"]["claude-code"].as_i64().unwrap() > 0 && o["by_harness"]["codex"].as_i64().unwrap() > 0);
    assert!(o["by_rel"]["supersedes"].as_i64().unwrap() == 6 && o["by_rel"]["causal"].as_i64().unwrap() >= 8);
    assert_eq!(o["decider"], "heuristic");
    assert!(o["embedder"].as_str().unwrap().starts_with("hash"));
    assert!(o["episodes"].as_i64().unwrap() > 0);
    assert!(!o["recent"].as_array().unwrap().is_empty() && !o["activity"].as_array().unwrap().is_empty());
    // pinned standing rules come first
    assert_eq!(o["preferences"][0]["pinned"], true);
}

#[tokio::test]
async fn memories_filters_and_search() {
    let e = env().await;
    let all = e.get("/ui/api/memories").await;
    assert_eq!(all["total"], 40, "superseded hidden by default");
    assert!(all["items"].as_array().unwrap().iter().all(|m| m["superseded_by"].is_null() && m["scope_label"].is_string()));
    let with_sup = e.get("/ui/api/memories?include_superseded=true").await;
    assert_eq!(with_sup["total"], 46);

    let g = e.get("/ui/api/memories?scope=global&kind=preference").await;
    assert!(g["total"].as_u64().unwrap() >= 5);
    assert!(g["items"].as_array().unwrap().iter().all(|m| m["scope"] == "global" && m["kind"] == "preference"));

    // scope by project name, several kinds, harness
    let p = e.get("/ui/api/memories?scope=payments-api&kind=gotcha,procedure").await;
    assert!(p["items"].as_array().unwrap().iter().all(|m| m["scope_label"] == "payments-api" && (m["kind"] == "gotcha" || m["kind"] == "procedure")));
    assert_eq!(p["total"], 4);
    let h = e.get("/ui/api/memories?harness=cursor").await;
    assert!(h["total"].as_u64().unwrap() >= 1 && h["items"].as_array().unwrap().iter().all(|m| m["source"]["harness"] == "cursor"));

    // paging
    let pg = e.get("/ui/api/memories?limit=5&offset=5").await;
    assert_eq!(pg["items"].as_array().unwrap().len(), 5);
    assert_eq!(pg["total"], 40);

    // q → recall ranking
    let q = e.get("/ui/api/memories?q=where%20are%20idempotency%20keys%20stored").await;
    assert_eq!(q["mode"], "recall");
    let first = &q["items"][0];
    assert!(first["content"].as_str().unwrap().contains("idempotency"), "{first}");
    assert!(first["score"].is_number() || first["via"] == "text");

    let (s, _) = e.send(reqwest::Method::GET, "/ui/api/memories?scope=nope", None).await;
    assert_eq!(s, 400);
}

#[tokio::test]
async fn memory_detail_has_neighbors_chain_and_source() {
    let e = env().await;
    let mid = e.find("move from Redis to a Postgres").await;
    let id = mid["id"].as_str().unwrap();
    let d = e.get(&format!("/ui/api/memories/{id}")).await;
    assert_eq!(d["memory"]["id"], id);
    let chain: Vec<&str> = d["chain"].as_array().unwrap().iter().map(|c| c["content"].as_str().unwrap()).collect();
    assert_eq!(chain.len(), 3, "{chain:?}");
    assert!(chain[0].contains("Redis with a 24h TTL") && chain[2].contains("kept 72h"), "oldest → newest: {chain:?}");
    let rels: Vec<(String, String)> = d["neighbors"].as_array().unwrap().iter().map(|n| (n["rel"].as_str().unwrap().into(), n["dir"].as_str().unwrap().into())).collect();
    assert!(rels.contains(&("supersedes".into(), "out".into())) && rels.contains(&("supersedes".into(), "in".into())), "{rels:?}");
    assert!(rels.iter().any(|r| r.0 == "causal"), "{rels:?}");
    assert_eq!(d["source"]["harness"], "claude-code");
    assert!(d["source"]["cwd"].as_str().unwrap().ends_with("payments-api"));
    assert_eq!(d["project"]["name"], "payments-api");

    let (s, _) = e.send(reqwest::Method::GET, "/ui/api/memories/m_nope", None).await;
    assert_eq!(s, 404);
}

#[tokio::test]
async fn patch_and_delete_memory() {
    let e = env().await;
    let m = e.find("Prefer `rg` and `fd`").await;
    let id = m["id"].as_str().unwrap().to_string();
    let url = format!("/ui/api/memories/{id}");
    assert_eq!(m["pinned"], false);

    let (s, r) = e.send(reqwest::Method::PATCH, &url, Some(json!({"pinned": true}))).await;
    assert_eq!(s, 200);
    assert_eq!(r["memory"]["pinned"], true);

    // edit → re-embedded and findable under the new wording
    let before = e.engine.store.embedding(&id).unwrap();
    let (s, r) = e.send(reqwest::Method::PATCH, &url, Some(json!({"content": "Prefer ripgrep and fd-find over grep and find in shell suggestions.", "why": "much faster on big monorepos"}))).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["memory"]["why"], "much faster on big monorepos");
    assert_ne!(before, e.engine.store.embedding(&id).unwrap());
    let q = e.get("/ui/api/memories?q=ripgrep").await;
    assert!(q["items"].as_array().unwrap().iter().any(|x| x["id"] == id.as_str()));

    // rejected edits
    let (s, _) = e.send(reqwest::Method::PATCH, &url, Some(json!({"content": "short"}))).await;
    assert_eq!(s, 400);
    let (s, _) = e.send(reqwest::Method::PATCH, &url, Some(json!({"content": "the deploy key is sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"}))).await;
    assert_eq!(s, 400);

    // change scope by project name
    let (s, r) = e.send(reqwest::Method::PATCH, &url, Some(json!({"scope": "tern"}))).await;
    assert_eq!(s, 200);
    assert_eq!(r["memory"]["scope"], format!("project:{}", pid_of(&e, "tern")));
    let (s, _) = e.send(reqwest::Method::PATCH, &url, Some(json!({"scope": "no-such-project"}))).await;
    assert_eq!(s, 400);

    let (s, _) = e.send(reqwest::Method::DELETE, &url, None).await;
    assert_eq!(s, 200);
    let (s, _) = e.send(reqwest::Method::GET, &url, None).await;
    assert_eq!(s, 404);
    let (s, _) = e.send(reqwest::Method::DELETE, &url, None).await;
    assert_eq!(s, 404);
    let t = e.get("/ui/api/traces?limit=10").await;
    let kinds: Vec<&str> = t["traces"].as_array().unwrap().iter().map(|x| x["kind"].as_str().unwrap()).collect();
    assert!(kinds.contains(&"forget") && kinds.contains(&"ui_edit"), "{kinds:?}");
}

#[tokio::test]
async fn projects_list_and_patch() {
    let e = env().await;
    let p = e.get("/ui/api/projects").await;
    let ps = p["projects"].as_array().unwrap();
    assert_eq!(ps.len(), 3);
    let pay = ps.iter().find(|x| x["card"]["name"] == "payments-api").unwrap();
    assert_eq!(pay["counts"]["live"], 13);
    assert_eq!(pay["counts"]["superseded"], 3);
    assert!(pay["counts"]["by_kind"]["decision"].as_i64().unwrap() >= 3);
    assert!(pay["card"]["stack"].as_str().unwrap().contains("rust"));
    assert!(pay["card"]["aliases"].as_array().unwrap().iter().any(|a| a == "payments"));
    assert!(p["global"]["counts"]["live"].as_i64().unwrap() >= 9);

    let pid = pay["card"]["pid"].as_str().unwrap();
    let (s, r) = e.send(reqwest::Method::PATCH, &format!("/ui/api/projects/{pid}"), Some(json!({"summary": "Card payments service.", "stack": "rust, axum, postgres"}))).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["card"]["summary"], "Card payments service.");
    let card = e.engine.store.get_project(pid).unwrap();
    assert_eq!(card.summary, "Card payments service.");
    assert_eq!(card.stack, "rust, axum, postgres");
    assert_eq!(card.summary_by, "agent:ui");
    let (s, r) = e.send(reqwest::Method::PATCH, &format!("/ui/api/projects/{pid}"), Some(json!({"name": "payments"}))).await;
    assert_eq!(s, 200);
    assert_eq!(r["card"]["name"], "payments");
    let (s, _) = e.send(reqwest::Method::PATCH, "/ui/api/projects/github.com/acme/nope", Some(json!({"summary": "x"}))).await;
    assert_eq!(s, 404);
}

#[tokio::test]
async fn graph_and_timeline() {
    let e = env().await;
    let g = e.get("/ui/api/graph").await;
    let nodes = g["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 46);
    let ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
    let edges = g["edges"].as_array().unwrap();
    assert!(edges.len() > 30);
    assert!(edges.iter().all(|x| ids.contains(x["src"].as_str().unwrap()) && ids.contains(x["dst"].as_str().unwrap())));
    for rel in ["supersedes", "causal", "entity", "semantic", "temporal"] {
        assert!(edges.iter().any(|x| x["rel"] == rel), "edge type {rel}");
    }
    let capped = e.get("/ui/api/graph?limit=10").await;
    assert_eq!(capped["nodes"].as_array().unwrap().len(), 10);
    assert_eq!(capped["truncated"], true);
    let gs = e.get("/ui/api/graph?scope=global").await;
    assert!(gs["nodes"].as_array().unwrap().iter().all(|n| n["scope"] == "global"));

    let t = e.get("/ui/api/timeline").await;
    let days = t["days"].as_array().unwrap();
    assert!(days.len() > 10);
    let dates: Vec<&str> = days.iter().map(|d| d["day"].as_str().unwrap()).collect();
    let mut sorted = dates.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(dates, sorted, "newest day first");
    let events: Vec<&Value> = days.iter().flat_map(|d| d["events"].as_array().unwrap()).collect();
    assert_eq!(events.len(), 6);
    assert!(events.iter().all(|ev| ev["type"] == "supersede" && ev["old"]["id"].is_string() && ev["new"]["id"].is_string()));
    let tt = e.get(&format!("/ui/api/timeline?scope=project:{}", pid_of(&e, "tern"))).await;
    assert!(tt["days"].as_array().unwrap().iter().flat_map(|d| d["memories"].as_array().unwrap()).all(|m| m["scope_label"] == "tern"));
}

#[tokio::test]
async fn traces_list() {
    let e = env().await;
    let t = e.get("/ui/api/traces?limit=5").await;
    assert_eq!(t["traces"].as_array().unwrap().len(), 5);
    let all = e.get("/ui/api/traces?limit=500").await;
    let tr = all["traces"].as_array().unwrap();
    assert!(tr.iter().any(|x| x["kind"] == "write" && x["data"]["kind"].is_string()));
    assert!(tr.iter().any(|x| x["kind"] == "recall" && x["data"]["q"].is_string()));
    assert!(!all["episodes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn recall_playground_shows_injection() {
    let e = env().await;
    let (s, r) = e.send(reqwest::Method::POST, "/ui/api/recall", Some(json!({"query": "why did we stop storing idempotency keys in redis?", "project": "payments-api"}))).await;
    assert_eq!(s, 200, "{r}");
    let notes = r["out"]["notes"].as_array().unwrap();
    assert!(!notes.is_empty());
    assert!(notes.iter().any(|n| n["content"].as_str().unwrap().contains("idempotency")));
    assert!(notes.iter().all(|n| n["p"].is_number() && n["via"].is_string()));
    assert!(r["out"]["sufficient"].is_boolean());
    let rendered = r["rendered"].as_str().unwrap();
    assert!(rendered.starts_with("<unimem-recall>") && rendered.trim_end().ends_with("</unimem-recall>"));
    assert!(rendered.contains("this project"));
    assert!(r["preamble"].as_str().unwrap().contains("Current project: payments-api"));
    assert_eq!(r["project"]["name"], "payments-api");
    assert!(r["scopes"].as_array().unwrap().iter().any(|x| x["label"] == "global"));

    // a working directory resolves the project the same way the gateway does
    let cwd = e.engine.store.get_project(&pid_of(&e, "tern")).unwrap().root;
    let (_, r) = e.send(reqwest::Method::POST, "/ui/api/recall", Some(json!({"query": "how do I backfill a partition in tern?", "cwd": cwd, "passive": false}))).await;
    assert_eq!(r["project"]["name"], "tern");
    assert_eq!(r["passive"], false);

    // passive recall of an acknowledgement is skipped and injects nothing
    let (_, r) = e.send(reqwest::Method::POST, "/ui/api/recall", Some(json!({"query": "ok thanks"}))).await;
    assert!(r["out"]["mode"] == "skipped" || r["out"]["mode"] == "gated", "{}", r["out"]);
    assert_eq!(r["rendered"], "");
}
