//! unimem-bench: offline evaluation harness for unimem.
//!
//! ```text
//! unimem-bench [all|e1|e2|e3|e4|e5|summary] [--data <dataset.json>] [--out <results dir>]
//! ```
//!
//! `--data` (default `bench/data/devmem.json`) is the E1 dataset; E2 always uses devmem.json (its scripted
//! conversation is about the ledger-api project). `--out` defaults to `bench/results`.
//!
//! * E1 memory quality on the synthetic developer corpus `bench/data/devmem.json` (unimem vs baselines)
//! * E2 gateway overhead + prompt-cache stability + interception correctness against a local mock Anthropic upstream
//! * E3 project identification on synthetic directory layouts
//! * E4 engine throughput at 1k / 10k notes
//! * E5 a 120-request Claude-Code-like session with tool errors and a forced compaction (budgets, de-dup,
//!   re-injection, tool-error recall, pre-compaction flush)
//!
//! Results go to `bench/results/*.json` and `bench/results/SUMMARY.md`.
//!
//! The System-One decider and the embedder are taken from the normal unimem config so they can be swapped:
//! `UNIMEM_BENCH_CONFIG=path/to/config.toml` (full config), or `UNIMEM_BENCH_DECIDER=jev,heuristic`,
//! `UNIMEM_BENCH_EMBEDDER=openai` on top of the defaults. Without either, the bench runs fully offline with the
//! heuristic decider and the hash embedder. `UNIMEM_BENCH_KEEP=1` keeps the temp directories.

use anyhow::{bail, Context, Result};
use axum::{
    body::{Body, Bytes},
    extract::State,
    response::Response,
    routing::post,
    Router,
};
use futures::StreamExt;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use unimem::config::Config;
use unimem::decide::ChainDecider;
use unimem::embed;
use unimem::engine::{Depth, Engine, RecallReq, WriteOutcome, WriteReq};
use unimem::gateway::{self, sse, AppState};
use unimem::project::{scope_of, Resolver};
use unimem::store::{cosine, Store};
use unimem::util::{hash_hex, tokenize};

// ============================================================================ common

fn bench_config() -> Result<Config> {
    let mut cfg = match std::env::var("UNIMEM_BENCH_CONFIG") {
        Ok(p) => Config::load(Some(Path::new(&p)))?,
        Err(_) => {
            let mut c = Config::default();
            c.decider.chain = vec!["heuristic".into()];
            c
        }
    };
    if let Ok(d) = std::env::var("UNIMEM_BENCH_DECIDER") {
        cfg.decider.chain = d.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    }
    if let Ok(k) = std::env::var("UNIMEM_BENCH_EMBEDDER") {
        cfg.embedder.kind = k;
    }
    if let Ok(m) = std::env::var("UNIMEM_BENCH_EMBEDDER_MODEL") {
        cfg.embedder.model = m;
    }
    Ok(cfg)
}

/// The embedder of a config, built once per distinct embedder config and shared by every engine of the run (a
/// transformer model would otherwise be loaded once per engine).
fn shared_embedder(cfg: &Config) -> Arc<dyn embed::Embedder> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, Arc<dyn embed::Embedder>>>> = std::sync::OnceLock::new();
    let key = serde_json::to_string(&cfg.embedder).unwrap_or_default();
    let mut c = CACHE.get_or_init(Default::default).lock();
    c.entry(key).or_insert_with(|| Arc::from(embed::build(&cfg.embedder))).clone()
}

fn make_engine(cfg: &Config, store: Store) -> Engine {
    Engine::with_parts(cfg.clone(), store, Box::new(shared_embedder(cfg)), ChainDecider::from_config(&cfg.decider))
}

fn config_info(cfg: &Config) -> Value {
    json!({"decider": ChainDecider::from_config(&cfg.decider).describe(), "embedder": shared_embedder(cfg).id()})
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Result<Self> {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.subsec_nanos();
        let p = std::env::temp_dir().join(format!("unimem-bench-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p)?;
        Ok(Self(p.canonicalize()?))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if std::env::var("UNIMEM_BENCH_KEEP").is_err() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let o = std::process::Command::new("git")
        .current_dir(dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .args(["-c", "user.email=bench@unimem", "-c", "user.name=bench", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false"])
        .args(args)
        .output()?;
    if !o.status.success() {
        bail!("git {:?} in {}: {}", args, dir.display(), String::from_utf8_lossy(&o.stderr));
    }
    Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn write_files(dir: &Path, files: &[(&str, &str)]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (f, c) in files {
        let p = dir.join(f);
        if let Some(pp) = p.parent() {
            std::fs::create_dir_all(pp)?;
        }
        std::fs::write(p, c)?;
    }
    Ok(())
}

fn mk_repo(dir: &Path, remotes: &[(&str, &str)], files: &[(&str, &str)]) -> Result<()> {
    write_files(dir, files)?;
    if files.is_empty() {
        write_files(dir, &[("README.md", "# test\n\nA small repository used by the unimem project-identification bench.\n")])?;
    }
    git(dir, &["init", "-q"])?;
    for (n, u) in remotes {
        git(dir, &["remote", "add", n, u])?;
    }
    git(dir, &["add", "-A"])?;
    git(dir, &["commit", "-qm", "init"])?;
    Ok(())
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((p / 100.0) * (s.len() - 1) as f64).round() as usize;
    s[i.min(s.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn lat(v: &[f64]) -> Value {
    json!({"n": v.len(), "p50_ms": r3(pct(v, 50.0)), "p95_ms": r3(pct(v, 95.0)), "mean_ms": r3(mean(v)), "max_ms": r3(pct(v, 100.0))})
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn bench_root() -> PathBuf {
    if let Ok(r) = std::env::var("UNIMEM_BENCH_ROOT") {
        return PathBuf::from(r);
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("bench/data/devmem.json").exists() {
        return cwd;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().pool_max_idle_per_host(16).build().unwrap()
}

/// Serve a router on an ephemeral port exactly as `unimem serve` does (`gateway::serve`).
async fn serve(r: Router) -> Result<String> {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = l.local_addr()?;
    tokio::spawn(async move {
        let _ = gateway::serve(l, r).await;
    });
    Ok(format!("http://{addr}"))
}

/// The mock upstream sets TCP_NODELAY so its many small SSE writes are not delayed by Nagle/delayed-ACK
/// (a real upstream sits behind TLS + network; we only want to measure the gateway).
async fn serve_nodelay(r: Router) -> Result<String> {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = l.local_addr()?;
    tokio::spawn(async move {
        use axum::serve::ListenerExt;
        let l = l.tap_io(|tcp| {
            let _ = tcp.set_nodelay(true);
        });
        let _ = axum::serve(l, r).await;
    });
    Ok(format!("http://{addr}"))
}

// ============================================================================ dataset

#[derive(Deserialize, Clone)]
struct DProject {
    key: String,
    dir: String,
    remote: Option<String>,
    files: BTreeMap<String, String>,
}

#[derive(Deserialize, Clone)]
struct DNote {
    id: String,
    at: String,
    harness: String,
    session: String,
    content: String,
    why: Option<String>,
    kind: String,
    scope: String,
    label: String,
    #[serde(rename = "ref")]
    reff: Option<String>,
    #[allow(dead_code)]
    explains: Option<String>,
    /// Earlier note this one resembles but must NOT supersede (held-out set).
    #[serde(default)]
    not_supersedes: Option<String>,
    /// Legit note that looks like something to reject ("secret" | "injection" | "status"); must be accepted.
    #[serde(default)]
    decoy: Option<String>,
}

#[derive(Deserialize, Clone)]
struct DQuery {
    id: String,
    cat: String,
    at: String,
    harness: String,
    q: String,
    gold: Vec<String>,
    stale: Vec<String>,
    forbidden: Vec<String>,
    #[allow(dead_code)]
    also: Vec<String>,
}

#[derive(Deserialize)]
struct Dataset {
    projects: Vec<DProject>,
    notes: Vec<DNote>,
    queries: Vec<DQuery>,
}

fn load_ds(p: &Path) -> Result<Dataset> {
    serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?)
        .with_context(|| format!("parsing {}", p.display()))
}

struct Corpus {
    dirs: HashMap<String, PathBuf>,
    home: PathBuf,
}

impl Corpus {
    fn build(base: &Path, ds: &Dataset) -> Result<Self> {
        let mut dirs = HashMap::new();
        for p in &ds.projects {
            let d = base.join(&p.dir);
            let files: Vec<(&str, &str)> = p.files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let remotes: Vec<(&str, &str)> = p.remote.iter().map(|r| ("origin", r.as_str())).collect();
            mk_repo(&d, &remotes, &files)?;
            dirs.insert(p.key.clone(), d);
        }
        let home = dirs::home_dir().context("no home dir")?;
        Ok(Self { dirs, home })
    }
    fn cwd(&self, at: &str) -> String {
        if at == "home" {
            self.home.display().to_string()
        } else {
            self.dirs[at].display().to_string()
        }
    }
}

enum Out {
    Stored { scope: String, kind: String, superseded: Vec<String> },
    Dup { of: String },
    Rejected { reason: String },
}

struct WriteLog {
    outs: Vec<(String, Out)>,
    m2n: HashMap<String, String>,
    ms: Vec<f64>,
}

async fn write_notes(e: &Engine, c: &Corpus, notes: &[&DNote]) -> Result<WriteLog> {
    let mut log = WriteLog { outs: vec![], m2n: HashMap::new(), ms: vec![] };
    for n in notes {
        let ctx = e.ctx_for(&n.harness, &n.session, Some(&c.cwd(&n.at)));
        let t = Instant::now();
        let r = e.write(&ctx, WriteReq { content: n.content.clone(), why: n.why.clone(), ..Default::default() }).await?;
        log.ms.push(ms(t));
        let o = match r {
            WriteOutcome::Stored { id, scope, kind, superseded, .. } => {
                log.m2n.insert(id.clone(), n.id.clone());
                Out::Stored { scope, kind, superseded }
            }
            WriteOutcome::Duplicate { of, .. } => Out::Dup { of },
            WriteOutcome::Rejected { reason } => Out::Rejected { reason },
        };
        log.outs.push((n.id.clone(), o));
    }
    Ok(log)
}

// ============================================================================ E1 baselines

struct BDoc {
    id: String,
    at: String,
    text: String,
}

fn bm_toks(s: &str) -> Vec<String> {
    tokenize(s).into_iter().filter(|t| t.chars().count() > 1 || !t.is_ascii()).collect()
}

struct Bm25 {
    tf: Vec<HashMap<String, f64>>,
    len: Vec<f64>,
    df: HashMap<String, f64>,
    avgdl: f64,
}

impl Bm25 {
    fn new(docs: &[BDoc]) -> Self {
        let mut tf = vec![];
        let mut len = vec![];
        let mut df: HashMap<String, f64> = HashMap::new();
        for d in docs {
            let toks = bm_toks(&d.text);
            let mut m: HashMap<String, f64> = HashMap::new();
            for t in &toks {
                *m.entry(t.clone()).or_default() += 1.0;
            }
            for t in m.keys() {
                *df.entry(t.clone()).or_default() += 1.0;
            }
            len.push(toks.len() as f64);
            tf.push(m);
        }
        let avgdl = mean(&len).max(1.0);
        Self { tf, len, df, avgdl }
    }
    fn score(&self, q: &str, i: usize) -> f64 {
        let (k1, b) = (1.2, 0.75);
        let n = self.tf.len() as f64;
        let qt: HashSet<String> = bm_toks(q).into_iter().collect();
        let mut s = 0.0;
        for t in qt {
            let Some(f) = self.tf[i].get(&t) else { continue };
            let df = self.df.get(&t).copied().unwrap_or(0.0);
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            s += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * self.len[i] / self.avgdl));
        }
        s
    }
}

// ============================================================================ E1 evaluation

#[derive(Default, Clone)]
struct SysOut {
    ids: Vec<String>,
    flagged: HashSet<String>,
    sufficient: bool,
    ms: f64,
}

struct Ev<'a> {
    ds: &'a Dataset,
    by_id: HashMap<&'a str, &'a DNote>,
}

impl<'a> Ev<'a> {
    fn canon(&self, id: &str) -> String {
        let mut cur = id.to_string();
        for _ in 0..10 {
            match self.by_id.get(cur.as_str()) {
                Some(n) if n.label == "duplicate" && n.reff.is_some() => cur = n.reff.clone().unwrap(),
                _ => break,
            }
        }
        cur
    }
    fn scope(&self, id: &str) -> String {
        self.by_id.get(id).map(|n| n.scope.clone()).unwrap_or_default()
    }
    /// Is every note equivalent to the gold written by another harness than the asking one?
    fn cross_harness(&self, q: &DQuery) -> bool {
        let gold: HashSet<&String> = q.gold.iter().collect();
        !self.ds.notes.iter().any(|n| gold.contains(&self.canon(&n.id)) && n.harness == q.harness)
    }
}

#[derive(Default, Clone)]
struct QM {
    answerable: bool,
    r1: f64,
    r5: f64,
    rr: f64,
    returned: f64,
    leaked: bool,
    leaked_notes: f64,
    forb_app: bool,
    forb_hit: bool,
    stale_app: bool,
    stale_any: bool,
    stale_unflagged: bool,
    stale_above: bool,
    unans: bool,
    abstain: bool,
    abstain_insuff: bool,
    unsafe_ret: bool,
    insuff: bool,
    hits: f64,
    ms: f64,
}

fn query_metrics(ev: &Ev, q: &DQuery, o: &SysOut) -> (QM, Vec<String>) {
    let mut c: Vec<String> = vec![];
    for id in &o.ids {
        let x = ev.canon(id);
        if !c.contains(&x) {
            c.push(x);
        }
    }
    let gold: HashSet<&String> = q.gold.iter().collect();
    let mut m = QM { answerable: !q.gold.is_empty(), returned: o.ids.len() as f64, ms: o.ms, insuff: !o.sufficient, ..Default::default() };
    m.hits = c.iter().filter(|x| gold.contains(x)).count() as f64;
    let first_gold = c.iter().position(|x| gold.contains(x));
    if m.answerable {
        let g = gold.len() as f64;
        m.r1 = c.iter().take(1).filter(|x| gold.contains(x)).count() as f64 / g;
        m.r5 = c.iter().take(5).filter(|x| gold.contains(x)).count() as f64 / g;
        m.rr = first_gold.map(|p| 1.0 / (p as f64 + 1.0)).unwrap_or(0.0);
    }
    let mut allowed: HashSet<String> = HashSet::from(["global".to_string()]);
    if q.at != "home" {
        allowed.insert(q.at.clone());
    }
    for g in &q.gold {
        allowed.insert(ev.scope(g));
    }
    let leaked = c.iter().filter(|x| !allowed.contains(&ev.scope(x))).count();
    m.leaked = leaked > 0;
    m.leaked_notes = leaked as f64;
    if !q.forbidden.is_empty() {
        m.forb_app = true;
        let f: HashSet<String> = q.forbidden.iter().map(|x| ev.canon(x)).collect();
        m.forb_hit = c.iter().any(|x| f.contains(x));
    }
    if !q.stale.is_empty() {
        m.stale_app = true;
        let s: HashSet<&String> = q.stale.iter().collect();
        m.stale_any = c.iter().any(|x| s.contains(x));
        m.stale_unflagged = o.ids.iter().any(|r| s.contains(&ev.canon(r)) && !o.flagged.contains(r));
        let first_stale = c.iter().position(|x| s.contains(x));
        m.stale_above = match (first_stale, first_gold) {
            (Some(a), Some(b)) => a < b,
            (Some(_), None) => true,
            _ => false,
        };
    }
    if q.cat == "unanswerable" {
        m.unans = true;
        m.abstain = c.is_empty();
        m.abstain_insuff = c.is_empty() || !o.sufficient;
    }
    m.unsafe_ret = c.iter().any(|x| ev.by_id.get(x.as_str()).is_some_and(|n| n.label.starts_with("reject")));
    (m, c)
}

fn agg(ms: &[&QM]) -> Value {
    let f = |p: &dyn Fn(&QM) -> bool| ms.iter().filter(|m| p(m)).count() as f64;
    let ans: Vec<&&QM> = ms.iter().filter(|m| m.answerable).collect();
    let na = ans.len() as f64;
    let n = ms.len() as f64;
    let rate = |num: f64, den: f64| if den > 0.0 { json!(r3(num / den)) } else { Value::Null };
    let sum = |g: &dyn Fn(&QM) -> f64, v: &[&&QM]| v.iter().map(|m| g(m)).sum::<f64>();
    let n_unans = f(&|m| m.unans);
    let n_stale = f(&|m| m.stale_app);
    let n_forb = f(&|m| m.forb_app);
    json!({
        "queries": n as u64,
        "answerable": na as u64,
        "recall@1": rate(sum(&|m| m.r1, &ans), na),
        "recall@5": rate(sum(&|m| m.r5, &ans), na),
        "mrr": rate(sum(&|m| m.rr, &ans), na),
        "notes_per_query": rate(ms.iter().map(|m| m.returned).sum::<f64>(), n),
        "precision_answerable": rate(sum(&|m| m.hits, &ans), sum(&|m| m.returned, &ans)),
        "insufficient_on_answerable_rate": rate(ans.iter().filter(|m| m.insuff).count() as f64, na),
        "false_abstain_rate": rate(ans.iter().filter(|m| m.returned == 0.0).count() as f64, na),
        "scope_leak_rate": rate(f(&|m| m.leaked), n),
        "leaked_notes_per_query": rate(ms.iter().map(|m| m.leaked_notes).sum::<f64>(), n),
        "forbidden_hit_rate": rate(f(&|m| m.forb_hit), n_forb),
        "forbidden_queries": n_forb as u64,
        "stale_queries": n_stale as u64,
        "stale_returned_rate": rate(f(&|m| m.stale_any), n_stale),
        "stale_unflagged_rate": rate(f(&|m| m.stale_unflagged), n_stale),
        "stale_above_gold_rate": rate(f(&|m| m.stale_above), n_stale),
        "unanswerable": n_unans as u64,
        "abstain_rate": rate(f(&|m| m.abstain), n_unans),
        "abstain_or_insufficient_rate": rate(f(&|m| m.abstain_insuff), n_unans),
        "unsafe_note_returned_rate": rate(f(&|m| m.unsafe_ret), n),
        "latency_ms_mean": r3(mean(&ms.iter().map(|m| m.ms).collect::<Vec<_>>())),
    })
}

const CATS: &[&str] = &["single_hop", "paraphrase", "chinese", "cross_project", "temporal", "causal", "scope_isolation", "unanswerable"];

fn evaluate(ev: &Ev, outs: &[SysOut]) -> (Value, Vec<Value>) {
    let mut qms = vec![];
    let mut per = vec![];
    for (q, o) in ev.ds.queries.iter().zip(outs) {
        let (m, c) = query_metrics(ev, q, o);
        per.push(json!({"id": q.id, "ret": c, "rr": r3(m.rr)}));
        qms.push((q, m));
    }
    let all: Vec<&QM> = qms.iter().map(|x| &x.1).collect();
    let mut by_cat = serde_json::Map::new();
    for c in CATS {
        let v: Vec<&QM> = qms.iter().filter(|x| x.0.cat == *c).map(|x| &x.1).collect();
        by_cat.insert(c.to_string(), agg(&v));
    }
    let cross: Vec<&QM> = qms.iter().filter(|x| x.1.answerable && ev.cross_harness(x.0)).map(|x| &x.1).collect();
    let same: Vec<&QM> = qms.iter().filter(|x| x.1.answerable && !ev.cross_harness(x.0)).map(|x| &x.1).collect();
    (json!({"overall": agg(&all), "by_cat": by_cat, "cross_harness": agg(&cross), "same_harness": agg(&same)}), per)
}

async fn run_uni(e: &Engine, m2n: &HashMap<String, String>, c: &Corpus, q: &DQuery, passive: bool) -> Result<SysOut> {
    run_uni_depth(e, m2n, c, q, passive, Depth::Adaptive).await
}

async fn run_uni_depth(e: &Engine, m2n: &HashMap<String, String>, c: &Corpus, q: &DQuery, passive: bool, depth: Depth) -> Result<SysOut> {
    let ctx = e.ctx_for(&q.harness, &format!("q-{}", q.id), Some(&c.cwd(&q.at)));
    let t = Instant::now();
    let out = e.recall(&ctx, RecallReq { query: q.q.clone(), project: None, k: 5, passive, depth }).await?;
    let el = ms(t);
    let mut o = SysOut { sufficient: out.sufficient, ms: el, ..Default::default() };
    for h in &out.notes {
        if let Some(n) = m2n.get(&h.id) {
            o.ids.push(n.clone());
            if h.superseded {
                o.flagged.insert(n.clone());
            }
        }
    }
    Ok(o)
}

fn baseline_visible(d: &BDoc, q: &DQuery, scoped: bool) -> bool {
    !scoped || d.at == q.at || d.at == "home"
}

async fn e1(data: &Path) -> Result<Value> {
    let cfg = bench_config()?;
    let ds = load_ds(data)?;
    let tmp = TempDir::new("e1")?;
    let corpus = Corpus::build(&tmp.0.join("ws"), &ds)?;
    let ev = Ev { ds: &ds, by_id: ds.notes.iter().map(|n| (n.id.as_str(), n)).collect() };
    println!(
        "[e1] {} projects, {} notes, {} queries; decider={} embedder={}",
        ds.projects.len(),
        ds.notes.len(),
        ds.queries.len(),
        config_info(&cfg)["decider"],
        config_info(&cfg)["embedder"]
    );

    // ---- unified unimem
    let full_db = tmp.0.join("full.db");
    let full = make_engine(&cfg, Store::open(&full_db)?);
    let mut scope2key: HashMap<String, String> = HashMap::from([("global".to_string(), "global".to_string())]);
    for p in &ds.projects {
        let ctx = full.ctx_for("bench", "setup", Some(&corpus.cwd(&p.key)));
        let pid = ctx.pid().context("project not resolved")?.to_string();
        scope2key.insert(scope_of(&pid), p.key.clone());
    }
    let all_notes: Vec<&DNote> = ds.notes.iter().collect();
    let log = write_notes(&full, &corpus, &all_notes).await?;

    // ---- lite: same DB, graph edges removed (no expansion possible)
    let lite_db = tmp.0.join("lite.db");
    {
        let c = rusqlite::Connection::open(&full_db)?;
        c.execute_batch(&format!("VACUUM INTO '{}'", lite_db.display()))?;
        let edges: Vec<(String, i64)> = {
            let mut st = c.prepare("SELECT rel, COUNT(*) FROM edges GROUP BY rel ORDER BY rel")?;
            let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.filter_map(|r| r.ok()).collect()
        };
        println!("[e1] edges: {edges:?}");
    }
    let edge_stats: Value = {
        let c = rusqlite::Connection::open(&full_db)?;
        let mut st = c.prepare("SELECT rel, COUNT(*) FROM edges GROUP BY rel ORDER BY rel")?;
        let rows: Vec<(String, i64)> = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.filter_map(|r| r.ok()).collect();
        json!(rows.into_iter().collect::<BTreeMap<_, _>>())
    };
    rusqlite::Connection::open(&lite_db)?.execute_batch("DELETE FROM edges;")?;
    let lite = make_engine(&cfg, Store::open(&lite_db)?);

    // ---- per-harness silos
    let mut silos: HashMap<String, (Engine, HashMap<String, String>)> = HashMap::new();
    let harnesses: std::collections::BTreeSet<&str> =
        ds.notes.iter().map(|n| n.harness.as_str()).chain(ds.queries.iter().map(|q| q.harness.as_str())).collect();
    for h in harnesses {
        let e = make_engine(&cfg, Store::open_memory()?);
        let notes: Vec<&DNote> = ds.notes.iter().filter(|n| n.harness == h).collect();
        let l = write_notes(&e, &corpus, &notes).await?;
        silos.insert(h.to_string(), (e, l.m2n));
    }

    // ---- baselines: store every write verbatim (no screening, no dedupe, no supersede)
    let docs: Vec<BDoc> = ds
        .notes
        .iter()
        .map(|n| BDoc { id: n.id.clone(), at: n.at.clone(), text: format!("{} {}", n.content, n.why.clone().unwrap_or_default()) })
        .collect();
    let embs = full.embedder.embed(&docs.iter().map(|d| d.text.clone()).collect::<Vec<_>>()).await?;
    let bm = Bm25::new(&docs);

    let mut results: BTreeMap<String, Vec<SysOut>> = BTreeMap::new();
    let names = [
        "unimem-full",
        "unimem-top5",
        "unimem-lite",
        "unimem-passive",
        "silo-per-harness",
        "vector-top5",
        "vector-top5-cwd",
        "bm25-top5",
        "bm25-top5-cwd",
    ];
    for n in names {
        results.insert(n.to_string(), vec![]);
    }
    // `UNIMEM_BENCH_EVIDENCE=FILE`: dump the recall candidates' evidence + gold label per query (for fitting the
    // relevance prior on development data, `bench/std/fit_prior.py`)
    let mut evid = match std::env::var("UNIMEM_BENCH_EVIDENCE") {
        Ok(p) => Some(std::fs::File::create(p)?),
        Err(_) => None,
    };
    for q in &ds.queries {
        if let Some(f) = evid.as_mut() {
            use std::io::Write as _;
            let ctx = full.ctx_for(&q.harness, &format!("q-{}", q.id), Some(&corpus.cwd(&q.at)));
            let scopes = full.recall_scopes(&ctx, &q.q, None);
            let cands = full.candidates(&q.q, &scopes, 30).await?;
            let rows: Vec<Value> =
                cands.iter().map(|c| json!({"y": log.m2n.get(&c.id).is_some_and(|n| q.gold.contains(n)) as u8, "ev": c.ev})).collect();
            writeln!(f, "{}", json!({"qid": q.id, "cat": q.cat, "n_gold": q.gold.len(), "cands": rows}))?;
        }
        results.get_mut("unimem-full").unwrap().push(run_uni(&full, &log.m2n, &corpus, q, false).await?);
        results.get_mut("unimem-top5").unwrap().push(run_uni_depth(&full, &log.m2n, &corpus, q, false, Depth::TopK).await?);
        results.get_mut("unimem-lite").unwrap().push(run_uni(&lite, &log.m2n, &corpus, q, false).await?);
        results.get_mut("unimem-passive").unwrap().push(run_uni(&full, &log.m2n, &corpus, q, true).await?);
        let (se, sm) = &silos[&q.harness];
        results.get_mut("silo-per-harness").unwrap().push(run_uni(se, sm, &corpus, q, false).await?);
        for scoped in [false, true] {
            let t = Instant::now();
            let qe = full.embedder.embed_one(&q.q).await?;
            let mut v: Vec<(usize, f32)> =
                docs.iter().enumerate().filter(|(_, d)| baseline_visible(d, q, scoped)).map(|(i, _)| (i, cosine(&qe, &embs[i]))).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let o =
                SysOut { ids: v.iter().take(5).map(|x| docs[x.0].id.clone()).collect(), sufficient: true, ms: ms(t), ..Default::default() };
            results.get_mut(if scoped { "vector-top5-cwd" } else { "vector-top5" }).unwrap().push(o);
            let t = Instant::now();
            let mut v: Vec<(usize, f64)> = docs
                .iter()
                .enumerate()
                .filter(|(_, d)| baseline_visible(d, q, scoped))
                .map(|(i, _)| (i, bm.score(&q.q, i)))
                .filter(|x| x.1 > 0.0)
                .collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let o =
                SysOut { ids: v.iter().take(5).map(|x| docs[x.0].id.clone()).collect(), sufficient: true, ms: ms(t), ..Default::default() };
            results.get_mut(if scoped { "bm25-top5-cwd" } else { "bm25-top5" }).unwrap().push(o);
        }
    }

    let mut systems = serde_json::Map::new();
    let mut per_query: BTreeMap<String, serde_json::Map<String, Value>> = BTreeMap::new();
    for n in names {
        let (v, per) = evaluate(&ev, &results[n]);
        println!(
            "[e1] {n:18} R@1={} R@5={} MRR={} leak={} abstain={}",
            v["overall"]["recall@1"],
            v["overall"]["recall@5"],
            v["overall"]["mrr"],
            v["overall"]["scope_leak_rate"],
            v["overall"]["abstain_rate"]
        );
        systems.insert(n.to_string(), v);
        for (q, p) in ds.queries.iter().zip(per) {
            per_query.entry(q.id.clone()).or_default().insert(n.to_string(), json!({"ret": p["ret"], "rr": p["rr"]}));
        }
    }
    let per_query: Vec<Value> = ds
        .queries
        .iter()
        .map(|q| json!({"id": q.id, "cat": q.cat, "at": q.at, "harness": q.harness, "q": q.q, "gold": q.gold, "systems": per_query.remove(&q.id).unwrap_or_default()}))
        .collect();

    let ws = write_screen(&ev, &log, &scope2key);
    println!("[e1] write screen: {}", ws["summary"]);
    Ok(json!({
        "experiment": "E1 memory quality",
        "config": config_info(&cfg),
        "dataset": {"path": data.display().to_string(), "projects": ds.projects.len(), "notes": ds.notes.len(), "queries": ds.queries.len(),
            "cross_harness_answerable": ds.queries.iter().filter(|q| !q.gold.is_empty() && ev.cross_harness(q)).count(),
            "same_harness_answerable": ds.queries.iter().filter(|q| !q.gold.is_empty() && !ev.cross_harness(q)).count()},
        "systems_description": {
            "unimem-full": "unimem engine, agent-initiated recall (unimem_recall tool, k=5), graph expansion when the query looks multi-hop",
            "unimem-top5": "unimem full in ranking mode (`Depth::TopK`, k=5: the tool with an explicit k), same depth as the top-5 baselines",
            "unimem-lite": "same database with all graph edges deleted (anchors + scoring only)",
            "unimem-passive": "unimem full with the passive (gateway-injected) gate enabled",
            "silo-per-harness": "unimem full, but one store per harness: notes written in claude-code are invisible to codex and vice versa",
            "vector-top5": "every write stored verbatim (no screening/dedupe/supersede), cosine top-5 over all notes, same embedder",
            "vector-top5-cwd": "as vector-top5 but only notes written in the same directory (or home) are visible",
            "bm25-top5": "every write stored verbatim, BM25 (k1=1.2,b=0.75, same tokenizer) top-5 over all notes",
            "bm25-top5-cwd": "as bm25-top5 but only notes written in the same directory (or home) are visible"
        },
        "edges": edge_stats,
        "write_latency": lat(&log.ms),
        "write_screen": ws,
        "systems": systems,
        "per_query": per_query,
    }))
}

fn write_screen(ev: &Ev, log: &WriteLog, scope2key: &HashMap<String, String>) -> Value {
    let label = |id: &str| ev.by_id[id].label.clone();
    let m2n = &log.m2n;
    let mut details = vec![];
    // rejection
    let (mut tp, mut fp, mut fnn) = (0, 0, 0);
    let mut per_type: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut false_rejects = vec![];
    // dup
    let (mut d_tp_strict, mut d_tp_loose, mut d_fp, mut d_fn) = (0, 0, 0, 0);
    let mut sup_as_dup = 0;
    let mut false_dups = vec![];
    // supersede
    let mut gold_pairs: HashSet<(String, String)> = HashSet::new();
    let mut pred_pairs: HashSet<(String, String)> = HashSet::new();
    let mut lost_updates = vec![];
    // scope / kind
    let mut scope_ok = 0;
    let mut scope_n = 0;
    let mut scope_groups: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut confusion: BTreeMap<String, u32> = BTreeMap::new();
    let mut kind_ok = 0;
    let mut scope_errors = vec![];
    // held-out extras
    let mut decoys: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut decoy_rejected = vec![];
    let mut distractor_pairs: Vec<(String, String)> = vec![];

    for (nid, o) in &log.outs {
        let n = ev.by_id[nid.as_str()];
        if let Some(d) = &n.decoy {
            let e = decoys.entry(d.clone()).or_default();
            e.1 += 1;
            if let Out::Rejected { reason } = o {
                e.0 += 1;
                decoy_rejected.push(json!({"id": nid, "decoy": d, "content": n.content, "reason": reason}));
            }
        }
        if let Some(x) = &n.not_supersedes {
            distractor_pairs.push((nid.clone(), ev.canon(x)));
        }
        let is_rej = n.label.starts_with("reject");
        let rejected = matches!(o, Out::Rejected { .. });
        if is_rej {
            let e = per_type.entry(n.label.clone()).or_default();
            e.1 += 1;
            if rejected {
                tp += 1;
                e.0 += 1;
            } else {
                fnn += 1;
                details.push(json!({"missed_reject": nid, "content": n.content}));
            }
        } else if rejected {
            fp += 1;
            if let Out::Rejected { reason } = o {
                false_rejects.push(json!({"id": nid, "label": n.label, "content": n.content, "reason": reason}));
            }
        }
        // duplicates
        if n.label == "duplicate" {
            match o {
                Out::Dup { of } => {
                    d_tp_loose += 1;
                    let of_note = m2n.get(of).map(|x| ev.canon(x)).unwrap_or_default();
                    if Some(of_note) == n.reff.as_ref().map(|r| ev.canon(r)) {
                        d_tp_strict += 1;
                    }
                }
                _ => d_fn += 1,
            }
        } else if let Out::Dup { of } = o {
            d_fp += 1;
            if n.label == "supersede" {
                sup_as_dup += 1;
            }
            false_dups.push(json!({"id": nid, "label": n.label, "content": n.content, "dup_of": m2n.get(of)}));
        }
        // supersede
        if n.label == "supersede" {
            if let Some(r) = &n.reff {
                gold_pairs.insert((nid.clone(), ev.canon(r)));
            }
            if !matches!(o, Out::Stored { .. }) {
                lost_updates.push(json!({"id": nid, "content": n.content, "outcome": match o { Out::Dup{..} => "duplicate", Out::Rejected{..} => "rejected", _ => "" }}));
            }
        }
        if let Out::Stored { scope, kind, superseded, .. } = o {
            for s in superseded {
                if let Some(old) = m2n.get(s) {
                    pred_pairs.insert((nid.clone(), ev.canon(old)));
                }
            }
            let got = scope2key.get(scope).cloned().unwrap_or(scope.clone());
            scope_n += 1;
            let group = if n.scope == "global" {
                if n.at == "home" {
                    "global note written at home"
                } else {
                    "global note written inside a project"
                }
            } else if n.scope == n.at {
                "project note written in that project"
            } else {
                "note about another project (cross-project write)"
            };
            let g = scope_groups.entry(group.to_string()).or_default();
            g.1 += 1;
            if got == n.scope {
                scope_ok += 1;
                g.0 += 1;
            } else {
                scope_errors.push(json!({"id": nid, "at": n.at, "gold": n.scope, "got": got, "content": n.content}));
            }
            let gk = if n.scope == "global" {
                "global"
            } else if n.scope == n.at {
                "current"
            } else {
                "other"
            };
            let pk = if got == "global" {
                "global"
            } else if got == n.at {
                "current"
            } else {
                "other"
            };
            *confusion.entry(format!("gold={gk} → got={pk}")).or_default() += 1;
            if *kind == n.kind {
                kind_ok += 1;
            }
        }
    }
    let _ = label;
    // distractor pairs: did the distractor supersede its look-alike? was the look-alike superseded by anything
    // that is not a gold superseding update of it?
    let mut ds_pair_hits = vec![];
    let mut ds_target_hits = vec![];
    for (a, b) in &distractor_pairs {
        if pred_pairs.contains(&(a.clone(), b.clone())) {
            ds_pair_hits.push(json!({"new": a, "superseded": b}));
        }
        let by: Vec<&String> =
            pred_pairs.iter().filter(|(x, y)| y == b && !gold_pairs.contains(&(x.clone(), y.clone()))).map(|(x, _)| x).collect();
        if !by.is_empty() {
            ds_target_hits.push(json!({"target": b, "wrongly_superseded_by": by}));
        }
    }
    let n_decoy: u32 = decoys.values().map(|v| v.1).sum();
    let n_decoy_rej: u32 = decoys.values().map(|v| v.0).sum();
    let tp_pairs = pred_pairs.intersection(&gold_pairs).count() as f64;
    let wrong_pairs: Vec<Value> = pred_pairs
        .difference(&gold_pairs)
        .map(|(a, b)| json!({"new": a, "superseded": b, "new_label": ev.by_id[a.as_str()].label}))
        .collect();
    let div = |a: f64, b: f64| if b > 0.0 { json!(r3(a / b)) } else { Value::Null };
    let n_dup_gold = (d_tp_loose + d_fn) as f64;
    let n_dup_gold_i = d_tp_loose + d_fn;
    json!({
        "summary": {
            "reject_precision": div(tp as f64, (tp + fp) as f64), "reject_recall": div(tp as f64, (tp + fnn) as f64),
            "dup_precision": div(d_tp_loose as f64, (d_tp_loose + d_fp) as f64), "dup_recall": div(d_tp_loose as f64, n_dup_gold),
            "dup_recall_correct_original": div(d_tp_strict as f64, n_dup_gold),
            "supersede_precision": div(tp_pairs, pred_pairs.len() as f64), "supersede_recall": div(tp_pairs, gold_pairs.len() as f64),
            "scope_accuracy": div(scope_ok as f64, scope_n as f64), "kind_accuracy": div(kind_ok as f64, scope_n as f64),
            "decoy_false_reject_rate": div(n_decoy_rej as f64, n_decoy as f64),
            "distractor_false_supersede_rate": div(ds_pair_hits.len() as f64, distractor_pairs.len() as f64),
            "distractor_target_wrongly_superseded_rate": div(ds_target_hits.len() as f64, distractor_pairs.len() as f64),
        },
        "decoys": {"n": n_decoy, "rejected": n_decoy_rej, "per_type_rejected": decoys.iter().map(|(k, v)| (k.clone(), json!(format!("{}/{}", v.0, v.1)))).collect::<serde_json::Map<_, _>>(), "rejected_notes": decoy_rejected},
        "distractors": {"pairs": distractor_pairs.len(), "false_supersede_pairs": ds_pair_hits, "targets_wrongly_superseded": ds_target_hits},
        "rejection": {"tp": tp, "fp": fp, "fn": fnn, "per_type_recall": per_type.iter().map(|(k, v)| (k.clone(), json!(format!("{}/{}", v.0, v.1)))).collect::<serde_json::Map<_, _>>(), "false_rejects": false_rejects},
        "duplicates": {"gold": n_dup_gold_i, "detected": d_tp_loose, "detected_with_correct_original": d_tp_strict, "false_positive": d_fp, "supersede_swallowed_as_duplicate": sup_as_dup, "false_dups": false_dups},
        "supersede": {"gold_pairs": gold_pairs.len(), "predicted_pairs": pred_pairs.len(), "correct_pairs": tp_pairs as u64, "wrong_pairs": wrong_pairs, "lost_updates": lost_updates},
        "scope": {"stored": scope_n, "correct": scope_ok, "by_group": scope_groups.iter().map(|(k, v)| (k.clone(), json!({"correct": v.0, "n": v.1, "acc": div(v.0 as f64, v.1 as f64)}))).collect::<serde_json::Map<_, _>>(), "confusion": confusion, "errors": scope_errors},
        "missed_rejects": details,
    })
}

// ============================================================================ E2 gateway

#[derive(Default)]
struct Mock {
    record: AtomicBool,
    seen: Mutex<Vec<Value>>,
    ctr: AtomicU64,
}

const FILLER: &str = "Here is the plan: I read the relevant module, checked how the call sites use it, and the change is local. \
I will keep the public interface stable, add a regression test that reproduces the failure, and run the full suite before committing. ";

fn tool_use(id: &str, name: &str, input: Value) -> Value {
    json!({"type": "tool_use", "id": id, "name": name, "input": input})
}

fn text(t: &str) -> Value {
    json!({"type": "text", "text": t})
}

fn blocks_of(c: &Value) -> Vec<Value> {
    match c {
        Value::String(s) => vec![text(s)],
        Value::Array(a) => a.clone(),
        _ => vec![],
    }
}

fn mock_content(v: &Value, n: u64) -> (Vec<Value>, &'static str) {
    let msgs = v["messages"].as_array().cloned().unwrap_or_default();
    let ours = v["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["name"].as_str().is_some_and(|s| s.starts_with("unimem_"))));
    let last = msgs.last().cloned().unwrap_or(Value::Null);
    let blocks = blocks_of(&last["content"]);
    let results: Vec<&Value> = blocks.iter().filter(|b| b["type"] == "tool_result").collect();
    if !results.is_empty() {
        let bash = results.iter().any(|b| b["tool_use_id"].as_str().is_some_and(|s| s.starts_with("toolu_bash")));
        let fixed =
            msgs.iter().any(|m| blocks_of(&m["content"]).iter().any(|b| b["text"].as_str().is_some_and(|t| t.contains("[[fixed]]"))));
        let uniq = if fixed { String::new() } else { format!(" (r{n})") };
        let t = if bash {
            format!("The test run finished: 42 passed, 0 failed{uniq}. {FILLER}")
        } else {
            format!("Memory says deploys go through ops/deploy.sh; proceeding with that{uniq}. {FILLER}")
        };
        return (vec![text(&t)], "end_turn");
    }
    let prompt: String = blocks.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("\n");
    let first_line = prompt.lines().next().unwrap_or("");
    let tag = &hash_hex(&[first_line])[..8];
    let bash = tool_use(&format!("toolu_bash{n}"), "Bash", json!({"command": "cargo test"}));
    let rec = tool_use(&format!("toolu_um{n}"), "unimem_recall", json!({"query": "how do we deploy and run the integration tests"}));
    let rem = tool_use(
        &format!("toolu_um{n}"),
        "unimem_remember",
        json!({"content": format!("ledger-api integration tests need `docker compose up -d db` before `uv run pytest -m slow` (bench note {n})")}),
    );
    if ours && prompt.contains("[[mixed]]") {
        (vec![text("Let me check memory and run the tests."), rec, bash], "tool_use")
    } else if ours && prompt.contains("[[recall]]") {
        (vec![text("Let me check memory first."), rec], "tool_use")
    } else if ours && prompt.contains("[[remember]]") {
        (vec![text("Saving that for later."), rem], "tool_use")
    } else if prompt.contains("[[bash]]") || prompt.contains("[[mixed]]") {
        (vec![text("Running the tests."), bash], "tool_use")
    } else {
        (vec![text(&format!("Answer {tag}: {FILLER}{FILLER}"))], "end_turn")
    }
}

fn sse_events(id: &str, content: &[Value], stop: &str) -> Vec<Value> {
    let mut ev = vec![
        json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":"claude-mock","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1200,"output_tokens":1,"cache_read_input_tokens":1000}}}),
        json!({"type":"ping"}),
    ];
    for (i, b) in content.iter().enumerate() {
        match b["type"].as_str().unwrap_or("") {
            "text" => {
                ev.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}}));
                let t: Vec<char> = b["text"].as_str().unwrap_or("").chars().collect();
                for ch in t.chunks(48) {
                    ev.push(
                        json!({"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text": ch.iter().collect::<String>()}}),
                    );
                }
            }
            _ => {
                ev.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id":b["id"],"name":b["name"],"input":{}}}));
                ev.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json": b["input"].to_string()}}));
            }
        }
        ev.push(json!({"type":"content_block_stop","index":i}));
    }
    ev.push(json!({"type":"message_delta","delta":{"stop_reason":stop,"stop_sequence":null},"usage":{"output_tokens":60}}));
    ev.push(json!({"type":"message_stop"}));
    ev
}

async fn mock_handler(State(m): State<Arc<Mock>>, body: Bytes) -> Response {
    let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let n = m.ctr.fetch_add(1, Ordering::SeqCst);
    let (content, stop) = mock_content(&v, n);
    let stream = v["stream"].as_bool().unwrap_or(false);
    if m.record.load(Ordering::SeqCst) {
        m.seen.lock().push(v);
    }
    if stream {
        let chunks: Vec<Result<Bytes, std::convert::Infallible>> = sse_events(&format!("msg_{n}"), &content, stop)
            .into_iter()
            .map(|e| Ok(sse::encode_json(Some(e["type"].as_str().unwrap_or("")), &e)))
            .collect();
        Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(futures::stream::iter(chunks))).unwrap()
    } else {
        let msg = json!({"id": format!("msg_{n}"), "type":"message","role":"assistant","model":"claude-mock","content":content,"stop_reason":stop,"stop_sequence":null,"usage":{"input_tokens":1200,"output_tokens":60,"cache_read_input_tokens":1000}});
        Response::builder().header("content-type", "application/json").body(Body::from(msg.to_string())).unwrap()
    }
}

fn filler(n: usize, seed: &str) -> String {
    let mut s = String::new();
    let mut i = 0;
    while s.len() < n {
        s.push_str(&format!("{seed} guideline {i}: prefer small, reviewable changes; read files before editing them; never invent APIs; keep the user informed. "));
        i += 1;
    }
    s
}

fn cc_system(cwd: &str) -> Value {
    json!([{"type":"text","text": format!("You are Claude Code, Anthropic's official CLI for Claude.\n{}\n<env>\nWorking directory: {cwd}\nIs directory a git repo: Yes\nPlatform: linux\n</env>", filler(11000, "System")), "cache_control": {"type":"ephemeral"}}])
}

fn cc_tools() -> Value {
    let names =
        ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task", "TodoWrite", "WebFetch", "WebSearch", "NotebookEdit", "ExitPlanMode"];
    Value::Array(
        names
            .iter()
            .map(|n| json!({"name": n, "description": filler(1400, n), "input_schema": {"type":"object","properties":{"command":{"type":"string","description":"input"},"path":{"type":"string"}},"required":[]}}))
            .collect(),
    )
}

fn agent_body(system: &Value, tools: &Value, messages: &[Value], stream: bool) -> Value {
    json!({"model":"claude-sonnet-x","max_tokens":4096,"stream":stream,"system":system,"tools":tools,"messages":messages,"metadata":{"user_id":"bench"}})
}

fn session_uuid(tag: u32, i: u64) -> String {
    format!("{tag:08x}-0000-4000-8000-{i:012x}")
}

struct Posted {
    status: u16,
    ttfb: f64,
    total: f64,
    bytes: Bytes,
}

async fn post_timed(client: &reqwest::Client, url: &str, session: &str, body: &Value) -> Result<Posted> {
    let raw = serde_json::to_vec(body)?;
    let t = Instant::now();
    let r = client
        .post(format!("{url}/v1/messages"))
        .header("x-api-key", "sk-bench")
        .header("anthropic-version", "2023-06-01")
        .header("user-agent", "claude-cli/2.0.0 (external, cli)")
        .header("x-claude-code-session-id", session)
        .header("content-type", "application/json")
        .body(raw)
        .send()
        .await?;
    let status = r.status().as_u16();
    let mut s = r.bytes_stream();
    let mut ttfb = None;
    let mut buf = Vec::new();
    while let Some(c) = s.next().await {
        let c = c?;
        if ttfb.is_none() {
            ttfb = Some(ms(t));
        }
        buf.extend_from_slice(&c);
    }
    let total = ms(t);
    Ok(Posted { status, ttfb: ttfb.unwrap_or(total), total, bytes: Bytes::from(buf) })
}

/// The assistant message as a harness reconstructs it, plus framing diagnostics.
struct Visible {
    content: Vec<Value>,
    stop: String,
    framing_ok: bool,
    framing_err: String,
}

fn parse_visible(stream: bool, b: &[u8]) -> Visible {
    if !stream {
        let v: Value = serde_json::from_slice(b).unwrap_or(Value::Null);
        return Visible {
            content: v["content"].as_array().cloned().unwrap_or_default(),
            stop: v["stop_reason"].as_str().unwrap_or("").into(),
            framing_ok: v["content"].is_array(),
            framing_err: String::new(),
        };
    }
    let mut p = sse::Parser::default();
    let mut evs = p.push(b);
    if let Some(e) = p.finish() {
        evs.push(e);
    }
    let mut blocks: Vec<Value> = vec![];
    let mut jb: Vec<String> = vec![];
    let mut open: Vec<bool> = vec![];
    let (mut starts, mut stops, mut stop) = (0, 0, String::new());
    let mut err = String::new();
    let n = evs.len();
    for (k, e) in evs.iter().enumerate() {
        let Some(v) = e.json() else { continue };
        match v["type"].as_str().unwrap_or("") {
            "message_start" => starts += 1,
            "message_stop" => {
                stops += 1;
                if k != n - 1 {
                    err.push_str("message_stop not last; ");
                }
            }
            "content_block_start" => {
                let i = v["index"].as_u64().unwrap_or(999) as usize;
                if i != blocks.len() {
                    err.push_str(&format!("non-contiguous index {i}; "));
                }
                blocks.push(v["content_block"].clone());
                jb.push(String::new());
                open.push(true);
            }
            "content_block_delta" => {
                let i = v["index"].as_u64().unwrap_or(999) as usize;
                if i >= blocks.len() {
                    err.push_str("delta for unknown block; ");
                    continue;
                }
                let d = &v["delta"];
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        let t = blocks[i]["text"].as_str().unwrap_or("").to_string() + d["text"].as_str().unwrap_or("");
                        blocks[i]["text"] = json!(t);
                    }
                    "input_json_delta" => jb[i].push_str(d["partial_json"].as_str().unwrap_or("")),
                    _ => {}
                }
            }
            "content_block_stop" => {
                let i = v["index"].as_u64().unwrap_or(999) as usize;
                if i < open.len() {
                    open[i] = false;
                } else {
                    err.push_str("stop for unknown block; ");
                }
            }
            "message_delta" => stop = v["delta"]["stop_reason"].as_str().unwrap_or("").to_string(),
            _ => {}
        }
    }
    for (b, j) in blocks.iter_mut().zip(&jb) {
        if !j.is_empty() {
            b["input"] = serde_json::from_str(j).unwrap_or(json!({}));
        }
    }
    if starts != 1 || stops != 1 {
        err.push_str(&format!("message_start={starts} message_stop={stops}; "));
    }
    if open.iter().any(|o| *o) {
        err.push_str("unclosed block; ");
    }
    Visible { content: blocks, stop, framing_ok: err.is_empty(), framing_err: err }
}

#[derive(Default)]
struct Checks {
    pass: u64,
    fail: Vec<String>,
    by_name: BTreeMap<String, (u64, u64)>,
}

impl Checks {
    fn check(&mut self, ok: bool, name: &str, detail: &str) {
        let e = self.by_name.entry(name.to_string()).or_default();
        e.1 += 1;
        if ok {
            self.pass += 1;
            e.0 += 1;
        } else {
            self.fail.push(format!("{name}: {detail}"));
        }
    }
    fn json(&self) -> Value {
        json!({"passed": self.pass, "failed": self.fail.len(), "total": self.pass + self.fail.len() as u64,
            "by_assertion": self.by_name.iter().map(|(k, v)| (k.clone(), json!(format!("{}/{}", v.0, v.1)))).collect::<serde_json::Map<_, _>>(),
            "failures": self.fail.iter().take(40).collect::<Vec<_>>()})
    }
}

fn leak_checks(ck: &mut Checks, p: &Posted, vis: &Visible, ctx: &str) {
    let s = String::from_utf8_lossy(&p.bytes);
    ck.check(p.status == 200, "status_200", ctx);
    ck.check(!s.contains("unimem_"), "client_never_sees_unimem_tool", ctx);
    ck.check(!s.contains("<unimem"), "client_never_sees_unimem_context", ctx);
    ck.check(vis.framing_ok, "framing_valid", &format!("{ctx}: {}", vis.framing_err));
}

fn names_of(tools: &Value) -> Vec<String> {
    tools.as_array().map(|a| a.iter().filter_map(|t| t["name"].as_str().map(String::from)).collect()).unwrap_or_default()
}

fn has_unimem_use(m: &Value) -> bool {
    blocks_of(&m["content"]).iter().any(|b| b["type"] == "tool_use" && b["name"].as_str().is_some_and(|s| s.starts_with("unimem_")))
}

/// Concatenation in cache-prefix order: tools → system → messages.
fn cache_bytes(v: &Value) -> Vec<u8> {
    let mut s = String::new();
    s.push_str(&v["tools"].to_string());
    s.push('\u{1}');
    s.push_str(&v["system"].to_string());
    s.push('\u{1}');
    for m in v["messages"].as_array().cloned().unwrap_or_default() {
        s.push_str(&m.to_string());
        s.push('\u{1}');
    }
    s.into_bytes()
}

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

async fn gateway_env(
    cfg: &Config,
    ds: &Dataset,
    corpus: &Corpus,
    db: &Path,
    upstream: &str,
    recall: bool,
) -> Result<(String, String, Arc<Engine>)> {
    let mut c = cfg.clone();
    c.upstream.anthropic = upstream.to_string();
    c.inject.recall = recall;
    let engine = Arc::new(make_engine(&c, Store::open(db)?));
    let notes: Vec<&DNote> = ds.notes.iter().collect();
    write_notes(&engine, corpus, &notes).await?;
    // the upstream client `unimem serve` builds
    let state = Arc::new(AppState { engine: engine.clone(), client: gateway::upstream_client(c.upstream.timeout_secs)? });
    let a = serve(gateway::router(state.clone())).await?;
    let b = serve_nodelay(gateway::router(state)).await?;
    Ok((a, b, engine))
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Passthrough,
    Unimem,
    NaiveSystem,
    NaiveTail,
}

fn last_prompt(msgs: &[Value]) -> Option<String> {
    msgs.iter().rev().filter(|m| m["role"] == "user").find_map(|m| {
        let t: Vec<String> =
            blocks_of(&m["content"]).iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str().map(String::from)).collect();
        (!t.is_empty()).then(|| t.join("\n"))
    })
}

async fn naive_recall(e: &Engine, cwd: &str, session: &str, prompt: &str) -> Result<Option<String>> {
    let ctx = e.ctx_for("claude-code", session, Some(cwd));
    let out = e.recall(&ctx, RecallReq { query: prompt.to_string(), project: None, k: 5, passive: true, ..Default::default() }).await?;
    Ok((!out.notes.is_empty())
        .then(|| format!("<memory>\nRelevant notes from long-term memory:\n{}</memory>", e.render_hits(&out.notes, &ctx))))
}

const CONV: &[&str] = &[
    "I'm picking up the ledger-api billing work again. Remind me how this service is deployed.",
    "Ok. Open app/billing/invoices.py and explain how invoice totals are computed.",
    "[[bash]] Run the unit tests to see where we are.",
    "Why do we keep money amounts as integers here?",
    "[[recall]] Check memory: is there anything about the Stripe webhook signature gotcha?",
    "Add a failing test for the rounding bug in multi-currency invoices.",
    "[[remember]] Remember: integration tests need the db container started first.",
    "[[bash]] Run the slow integration suite now.",
    "What port does the API listen on in local dev?",
    "Refactor compute_totals into a pure function, keep the signature.",
    "[[recall]] What did we decide about SQLAlchemy upgrades?",
    "[[mixed]] Check memory for the test command and run the tests.",
    "Write the migration for the new credit_notes table.",
    "Who should review changes to the money paths?",
    "[[bash]] Run alembic upgrade head against the test DB.",
    "Explain the retry policy for failed Stripe charges.",
    "[[recall]] How does the nightly invoice job get scheduled?",
    "Update the README section about local setup.",
    "好的，再帮我看一下 tenants 表的索引是否合理。",
    "[[remember]] Remember that credit notes must never be negative; validate at the API layer.",
    "[[bash]] Run the linters.",
    "What's the per-tenant rate limit on the invoices endpoint now?",
    "Draft the PR description for these changes.",
    "[[mixed]] Recall the branch naming convention and then run the full test suite.",
    "How do I tail the staging logs?",
    "[[recall]] Anything in memory about the Postgres version we target?",
    "Fix the typo in the error message for invalid currency codes.",
    "[[bash]] Run the tests one last time.",
    "Summarize everything we changed today in five bullets.",
    "Thanks, that's all for today.",
];

#[allow(clippy::too_many_arguments)]
async fn converse(
    mode: Mode,
    client: &reqwest::Client,
    mock_url: &str,
    gw_url: &str,
    mock: &Mock,
    naive: &Engine,
    cwd: &str,
    ck: &mut Checks,
) -> Result<Value> {
    let system = cc_system(cwd);
    let tools = cc_tools();
    let session = session_uuid(0xC0FFEE00 + mode as u32, 1);
    mock.seen.lock().clear();
    mock.record.store(true, Ordering::SeqCst);
    let mut msgs: Vec<Value> = vec![];
    let mut client_reqs = 0;
    for (t, p) in CONV.iter().enumerate() {
        msgs.push(json!({"role":"user","content":[{"type":"text","text": p}]}));
        for _round in 0..4 {
            let body = agent_body(&system, &tools, &msgs, true);
            client_reqs += 1;
            let posted = match mode {
                Mode::Unimem => post_timed(client, gw_url, &session, &body).await?,
                Mode::Passthrough => post_timed(client, mock_url, &session, &body).await?,
                Mode::NaiveSystem => {
                    let mut b = body.clone();
                    if let Some(r) = naive_recall(naive, cwd, &session, &last_prompt(&msgs).unwrap_or_default()).await? {
                        b["system"].as_array_mut().unwrap().push(json!({"type":"text","text": r}));
                    }
                    post_timed(client, mock_url, &session, &b).await?
                }
                Mode::NaiveTail => {
                    let mut b = body.clone();
                    let last_is_prompt = msgs.last().is_some_and(|m| blocks_of(&m["content"]).iter().any(|x| x["type"] == "text"));
                    if last_is_prompt {
                        if let Some(r) = naive_recall(naive, cwd, &session, &last_prompt(&msgs).unwrap_or_default()).await? {
                            let m = b["messages"].as_array_mut().unwrap().last_mut().unwrap();
                            m["content"].as_array_mut().unwrap().push(json!({"type":"text","text": r}));
                        }
                    }
                    post_timed(client, mock_url, &session, &b).await?
                }
            };
            let vis = parse_visible(true, &posted.bytes);
            if mode == Mode::Unimem {
                leak_checks(ck, &posted, &vis, &format!("conversation turn {t}"));
            }
            msgs.push(json!({"role":"assistant","content": vis.content.clone()}));
            let bash: Vec<String> = vis
                .content
                .iter()
                .filter(|b| b["type"] == "tool_use" && b["name"] == "Bash")
                .filter_map(|b| b["id"].as_str().map(String::from))
                .collect();
            if bash.is_empty() {
                break;
            }
            msgs.push(json!({"role":"user","content": bash.iter().map(|id| json!({"type":"tool_result","tool_use_id": id, "content": "ok: 42 passed"})).collect::<Vec<_>>()}));
        }
    }
    mock.record.store(false, Ordering::SeqCst);
    let seen = std::mem::take(&mut *mock.seen.lock());
    let keys: Vec<Vec<u8>> = seen.iter().map(cache_bytes).collect();
    let mut fracs = vec![];
    let (mut common_sum, mut next_sum) = (0usize, 0usize);
    let mut stable = 0;
    for w in keys.windows(2) {
        let c = lcp(&w[0], &w[1]);
        fracs.push(c as f64 / w[0].len() as f64);
        common_sum += c;
        next_sum += w[1].len();
        if c == w[0].len() {
            stable += 1;
        }
    }
    let sys_stable = seen.windows(2).all(|w| w[0]["system"] == w[1]["system"]);
    let tools_stable = seen.windows(2).all(|w| w[0]["tools"] == w[1]["tools"]);
    Ok(json!({
        "mode": format!("{mode:?}"),
        "turns": CONV.len(),
        "client_requests": client_reqs,
        "upstream_requests": seen.len(),
        "continuation_rounds": seen.len() as i64 - client_reqs as i64,
        "transitions": fracs.len(),
        "prefix_fraction_mean": r3(mean(&fracs)),
        "prefix_fraction_min": r3(fracs.iter().cloned().fold(f64::INFINITY, f64::min)),
        "fully_prefix_stable_transitions": stable,
        "cacheable_byte_ratio": r3(common_sum as f64 / next_sum.max(1) as f64),
        "upstream_bytes_total": keys.iter().map(|k| k.len()).sum::<usize>(),
        "system_byte_stable": sys_stable,
        "tools_byte_stable": tools_stable,
    }))
}

async fn e2(root: &Path) -> Result<Value> {
    let cfg = bench_config()?;
    let ds = load_ds(&root.join("bench/data/devmem.json"))?;
    let tmp = TempDir::new("e2")?;
    let corpus = Corpus::build(&tmp.0.join("ws"), &ds)?;
    let cwd = corpus.cwd("ledger-api");
    let mock = Arc::new(Mock::default());
    let mock_url = serve_nodelay(Router::new().route("/v1/messages", post(mock_handler)).with_state(mock.clone())).await?;
    let (gw, gw_nd, _e1) = gateway_env(&cfg, &ds, &corpus, &tmp.0.join("gw.db"), &mock_url, true).await?;
    let (gw_nr, _gw_nr_nd, _e2) = gateway_env(&cfg, &ds, &corpus, &tmp.0.join("gw-norecall.db"), &mock_url, false).await?;
    let naive = make_engine(&cfg, Store::open(&tmp.0.join("naive.db"))?);
    write_notes(&naive, &corpus, &ds.notes.iter().collect::<Vec<_>>()).await?;
    let client = http_client();
    let system = cc_system(&cwd);
    let tools = cc_tools();

    // ---------------- latency
    let n_req: u64 = std::env::var("UNIMEM_BENCH_E2_N").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let warm = 20;
    // "gateway-nodelay": the same gateway state served with TCP_NODELAY on its client-facing sockets (diagnostic;
    // `unimem serve` does not set it).
    let targets = [
        ("direct", mock_url.as_str()),
        ("gateway", gw.as_str()),
        ("gateway-norecall", gw_nr.as_str()),
        ("gateway-nodelay", gw_nd.as_str()),
    ];
    let mut lat_s: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut body_bytes = vec![];
    for i in 0..(n_req + warm) {
        let q = &ds.queries[(i as usize * 7) % ds.queries.len()].q;
        let hist = (i % 6) as usize;
        let mut msgs = vec![];
        for h in 0..hist {
            msgs.push(json!({"role":"user","content":[{"type":"text","text": format!("Earlier request {h} in this session: look at module {h}.")}]}));
            msgs.push(json!({"role":"assistant","content":[{"type":"text","text": format!("Done with module {h}. {FILLER}")}]}));
        }
        msgs.push(json!({"role":"user","content":[{"type":"text","text": format!("{q} (request {i})")}]}));
        for stream in [true, false] {
            let body = agent_body(&system, &tools, &msgs, stream);
            if i == 0 {
                body_bytes.push(serde_json::to_vec(&body)?.len());
            }
            let rot = (i as usize) % targets.len();
            for k in 0..targets.len() {
                let ti = (k + rot) % targets.len();
                let (name, url) = targets[ti];
                let session = session_uuid(0xA0 + ti as u32, i / 10);
                let p = post_timed(&client, url, &session, &body).await?;
                if p.status != 200 {
                    bail!("{name} returned {}", p.status);
                }
                if i >= warm {
                    let e = lat_s.entry(format!("{name}/{}", if stream { "stream" } else { "json" })).or_default();
                    e.0.push(p.ttfb);
                    e.1.push(p.total);
                }
            }
        }
    }
    let mut latency = serde_json::Map::new();
    for (k, (ttfb, total)) in &lat_s {
        latency.insert(k.clone(), json!({"ttfb": lat(ttfb), "total": lat(total)}));
    }
    let mut overhead = serde_json::Map::new();
    for via in ["gateway", "gateway-norecall", "gateway-nodelay"] {
        for kind in ["stream", "json"] {
            let d = &lat_s[&format!("direct/{kind}")];
            let g = &lat_s[&format!("{via}/{kind}")];
            let dt: Vec<f64> = g.0.iter().zip(&d.0).map(|(a, b)| a - b).collect();
            let dtot: Vec<f64> = g.1.iter().zip(&d.1).map(|(a, b)| a - b).collect();
            overhead.insert(format!("{via}/{kind}"), json!({
                "added_ttfb_p50_ms": r3(pct(&g.0, 50.0) - pct(&d.0, 50.0)), "added_ttfb_p95_ms": r3(pct(&g.0, 95.0) - pct(&d.0, 95.0)),
                "added_total_p50_ms": r3(pct(&g.1, 50.0) - pct(&d.1, 50.0)), "added_total_p95_ms": r3(pct(&g.1, 95.0) - pct(&d.1, 95.0)),
                "paired_diff_ttfb": lat(&dt), "paired_diff_total": lat(&dtot),
            }));
        }
    }
    println!("[e2] latency done: {}", Value::Object(overhead.clone()));

    // ---------------- interception correctness
    let mut ck = Checks::default();
    let markers = ["recall", "remember", "mixed", "bash", "plain"];
    let (mut client_reqs, mut upstream_reqs, mut continuations) = (0u64, 0u64, 0u64);
    mock.record.store(true, Ordering::SeqCst);
    for rep in 0..6 {
        for m in markers {
            for stream in [true, false] {
                let ctxs = format!("{m}/{}/rep{rep}", if stream { "stream" } else { "json" });
                let session = session_uuid(0xB0, (rep * 100 + markers.iter().position(|x| *x == m).unwrap() * 2 + stream as usize) as u64);
                let prompt = if m == "plain" {
                    format!("Explain the invoices module, part {rep}.")
                } else {
                    format!("[[{m}]] please help with the deploy step {rep}")
                };
                let mut msgs = vec![json!({"role":"user","content":[{"type":"text","text": prompt}]})];
                let before = mock.seen.lock().len();
                let body = agent_body(&system, &tools, &msgs, stream);
                let p = post_timed(&client, &gw, &session, &body).await?;
                client_reqs += 1;
                let vis = parse_visible(stream, &p.bytes);
                leak_checks(&mut ck, &p, &vis, &ctxs);
                let ups: Vec<Value> = mock.seen.lock()[before..].to_vec();
                upstream_reqs += ups.len() as u64;
                continuations += ups.len().saturating_sub(1) as u64;
                let ours_only = m == "recall" || m == "remember";
                ck.check(
                    !ups.is_empty()
                        && names_of(&ups[0]["tools"])
                            == [
                                names_of(&tools),
                                vec!["unimem_recall".into(), "unimem_remember".into(), "unimem_forget".into(), "unimem_project".into()],
                            ]
                            .concat(),
                    "tools_appended_in_order",
                    &ctxs,
                );
                ck.check(!ups.is_empty() && ups[0]["system"] == system, "system_untouched", &ctxs);
                ck.check(ups.len() == if ours_only { 2 } else { 1 }, "continuation_count", &format!("{ctxs}: upstream={}", ups.len()));
                let want_stop = if m == "mixed" || m == "bash" { "tool_use" } else { "end_turn" };
                ck.check(vis.stop == want_stop, "visible_stop_reason", &format!("{ctxs}: {}", vis.stop));
                if m == "mixed" || m == "bash" {
                    ck.check(
                        vis.content.iter().any(|b| b["name"] == "Bash" && b["input"] == json!({"command":"cargo test"})),
                        "harness_tool_call_delivered",
                        &ctxs,
                    );
                }
                if ours_only {
                    ck.check(vis.content.iter().filter(|b| b["type"] == "text").count() == 2, "rounds_merged_into_one_message", &ctxs);
                }
                // follow-up request (harness echoes the visible message)
                msgs.push(json!({"role":"assistant","content": vis.content.clone()}));
                let bash: Vec<String> =
                    vis.content.iter().filter(|b| b["name"] == "Bash").filter_map(|b| b["id"].as_str().map(String::from)).collect();
                if bash.is_empty() {
                    msgs.push(json!({"role":"user","content":[{"type":"text","text":"thanks, continue"}]}));
                } else {
                    msgs.push(json!({"role":"user","content": bash.iter().map(|id| json!({"type":"tool_result","tool_use_id": id,"content":"ok"})).collect::<Vec<_>>()}));
                }
                let before = mock.seen.lock().len();
                let p2 = post_timed(&client, &gw, &session, &agent_body(&system, &tools, &msgs, stream)).await?;
                client_reqs += 1;
                let vis2 = parse_visible(stream, &p2.bytes);
                leak_checks(&mut ck, &p2, &vis2, &format!("{ctxs}/followup"));
                let ups2: Vec<Value> = mock.seen.lock()[before..].to_vec();
                upstream_reqs += ups2.len() as u64;
                continuations += ups2.len().saturating_sub(1) as u64;
                let Some(u) = ups2.first() else {
                    ck.check(false, "followup_reached_upstream", &ctxs);
                    continue;
                };
                let um = u["messages"].as_array().cloned().unwrap_or_default();
                match m {
                    "recall" | "remember" => {
                        ck.check(
                            um.len() == msgs.len() + 2 && has_unimem_use(&um[1]) && um[2]["content"][0]["type"] == "tool_result",
                            "hidden_rounds_replayed",
                            &format!("{ctxs}: {} msgs", um.len()),
                        );
                        let same = ups.len() == 2 && um.len() > 2 && um[1] == ups[1]["messages"][1] && um[2] == ups[1]["messages"][2];
                        if !same && std::env::var("UNIMEM_BENCH_DEBUG").is_ok() {
                            eprintln!(
                                "REPLAY {ctxs}\n cont: {}\n {}\n replay: {}\n {}",
                                ups.get(1).map(|u| u["messages"][1].to_string()).unwrap_or_default(),
                                ups.get(1).map(|u| u["messages"][2].to_string()).unwrap_or_default(),
                                um[1],
                                um[2]
                            );
                        }
                        ck.check(same, "replay_byte_identical_to_continuation", &ctxs);
                    }
                    "mixed" => {
                        let first_user = blocks_of(&um.get(2).cloned().unwrap_or_default()["content"]);
                        ck.check(
                            um.len() == msgs.len()
                                && has_unimem_use(&um[1])
                                && first_user.len() == 2
                                && first_user[0]["tool_use_id"].as_str().is_some_and(|s| s.starts_with("toolu_um")),
                            "mixed_hidden_call_and_result_replayed",
                            &format!(
                                "{ctxs}: {}",
                                serde_json::to_string(&um.get(2)).unwrap_or_default().chars().take(200).collect::<String>()
                            ),
                        );
                    }
                    _ => {
                        ck.check(um.len() == msgs.len() && um[1] == msgs[1], "plain_history_unchanged", &ctxs);
                    }
                }
                ck.check(ups.first().is_some_and(|u0| u0["messages"][0] == um[0]), "augmented_prompt_replayed_byte_identical", &ctxs);
            }
        }
    }
    // Anchor-collision probe: two different sessions, identical prompt, identical (deterministic) model reply.
    // Expansion overlays are keyed on (previous message, visible message) only, so session B may get session A's
    // hidden rounds replayed. Counted separately from the correctness assertions.
    let mut probe = (0u64, 0u64);
    for i in 0..10u64 {
        let prompt = format!("[[recall]] [[fixed]] where is the deploy script? probe {i}");
        let mut own = vec![];
        let mut replayed = vec![];
        for sess in 0..2u64 {
            let session = session_uuid(0xD0, i * 2 + sess);
            let mut msgs = vec![json!({"role":"user","content":[{"type":"text","text": prompt}]})];
            let before = mock.seen.lock().len();
            let p = post_timed(&client, &gw, &session, &agent_body(&system, &tools, &msgs, false)).await?;
            let vis = parse_visible(false, &p.bytes);
            let ups: Vec<Value> = mock.seen.lock()[before..].to_vec();
            own.push(ups.get(1).map(|u| u["messages"][1].clone()).unwrap_or_default());
            msgs.push(json!({"role":"assistant","content": vis.content}));
            msgs.push(json!({"role":"user","content":[{"type":"text","text":"ok thanks"}]}));
            let before = mock.seen.lock().len();
            post_timed(&client, &gw, &session, &agent_body(&system, &tools, &msgs, false)).await?;
            replayed.push(mock.seen.lock().get(before).map(|u| u["messages"][1].clone()).unwrap_or_default());
        }
        probe.1 += 1;
        if replayed[1] != own[1] && replayed[1] == own[0] {
            probe.0 += 1;
        }
    }
    mock.record.store(false, Ordering::SeqCst);
    println!(
        "[e2] correctness: {} passed, {} failed; anchor-collision probe: {}/{} sessions got another session's hidden rounds",
        ck.pass,
        ck.fail.len(),
        probe.0,
        probe.1
    );

    // ---------------- cache stability over a 30-turn conversation
    let mut conv = serde_json::Map::new();
    for mode in [Mode::Passthrough, Mode::Unimem, Mode::NaiveSystem, Mode::NaiveTail] {
        let r = converse(mode, &client, &mock_url, &gw, &mock, &naive, &cwd, &mut ck).await?;
        println!("[e2] conversation {mode:?}: prefix mean {} cacheable {}", r["prefix_fraction_mean"], r["cacheable_byte_ratio"]);
        conv.insert(format!("{mode:?}"), r);
    }

    Ok(json!({
        "experiment": "E2 gateway overhead & correctness",
        "config": config_info(&cfg),
        "setup": {
            "requests_per_cell": n_req, "warmup": warm, "request_body_bytes": {"stream": body_bytes.first(), "json": body_bytes.get(1)},
            "history_messages": "0-10 (cycled)", "memory_notes_loaded": ds.notes.len(),
            "notes": "mock upstream (TCP_NODELAY) on 127.0.0.1 answers instantly (no model latency), so differences are pure gateway cost; sessions rotate every 10 requests (preamble once per session); every request is a new prompt (passive recall runs each time)."
        },
        "latency": latency,
        "overhead": overhead,
        "correctness": {"client_requests": client_reqs, "upstream_requests": upstream_reqs, "continuation_rounds": continuations, "assertions": ck.json()},
        "anchor_collision_probe": {"pairs": probe.1, "second_session_got_first_sessions_hidden_rounds": probe.0,
            "description": "two sessions send the same prompt and get the same deterministic reply that used unimem_recall; on the follow-up, does session 2's replay contain session 1's tool_use ids/results?"},
        "cache_stability": conv,
    }))
}

// ============================================================================ E5 long session

#[derive(Default)]
struct Mock5 {
    seen: Mutex<Vec<Value>>,
    ctr: AtomicU64,
}

/// E5 upstream: a scripted "model". A prompt or tool result carrying `[[run:<cmd>]]` gets a Bash call; a prompt with
/// `[[recall]]` gets an `unimem_recall` call (a hidden round); everything else a short answer. Usage reports the
/// request size (bytes / 4) as input tokens, so the gateway's context tracking sees realistic numbers.
async fn mock5_handler(State(m): State<Arc<Mock5>>, body: Bytes) -> Response {
    let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let n = m.ctr.fetch_add(1, Ordering::SeqCst);
    m.seen.lock().push(v.clone());
    let last = v["messages"].as_array().and_then(|a| a.last()).cloned().unwrap_or(Value::Null);
    let blocks = blocks_of(&last["content"]);
    let text: String = blocks
        .iter()
        .map(|b| match b["type"].as_str() {
            Some("text") => b["text"].as_str().unwrap_or("").to_string(),
            Some("tool_result") => b["content"].as_str().unwrap_or("").to_string(),
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let ours_result = blocks.iter().any(|b| b["tool_use_id"].as_str().is_some_and(|s| s.starts_with("toolu_um")));
    let has_ours = v["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["name"] == "unimem_recall"));
    let (content, stop) = if let Some(cmd) = text.split("[[run:").nth(1).and_then(|r| r.split("]]").next()) {
        (vec![text_block_e5("Running it."), tool_use(&format!("toolu_e5_{n}"), "Bash", json!({"command": cmd}))], "tool_use")
    } else if text.contains("[[recall]]") && has_ours && !ours_result {
        (
            vec![
                text_block_e5("Let me check memory."),
                tool_use(
                    &format!("toolu_um{n}"),
                    "unimem_recall",
                    json!({"query": "ledger-api conventions for tests, deploys and migrations", "k": 8}),
                ),
            ],
            "tool_use",
        )
    } else {
        (vec![text_block_e5(&format!("Done (r{n}). {FILLER}"))], "end_turn")
    };
    let msg = json!({"id": format!("msg_{n}"), "type":"message","role":"assistant","model":"claude-mock","content":content,"stop_reason":stop,"stop_sequence":null,
        "usage":{"input_tokens": body.len() / 4, "output_tokens": 60}});
    Response::builder().header("content-type", "application/json").body(Body::from(msg.to_string())).unwrap()
}

fn text_block_e5(t: &str) -> Value {
    json!({"type":"text","text":t})
}

/// A long, benign tool output (build / test log) of about `chars` characters.
fn log_output(seed: usize, chars: usize) -> String {
    let mut s = String::new();
    let mut i = 0;
    while s.len() < chars {
        s.push_str(&format!("tests/unit/test_mod{seed}_{i}.py::test_case_{i} PASSED [{:>3}%]\n", (i * 7) % 100));
        i += 1;
    }
    s.push_str("==== 42 passed in 3.21s ====\n");
    s
}

/// Tool error outputs whose root cause is a stored ledger-api note (devmem id).
const E5_ERRORS: &[(&str, &str, &str)] = &[
    ("n053", "uv run pytest tests/test_webhooks.py", "FAILED tests/test_webhooks.py::test_stripe_event - stripe.error.SignatureVerificationError: No signatures found matching the expected signature for payload\n  File \"app/billing/webhooks.py\", line 41, in stripe_webhook\n    event = stripe.Webhook.construct_event(payload, sig_header, secret)"),
    ("n050", "uv run python -m ledger.jobs.invoice_batch --dry-run", "sqlalchemy.exc.TimeoutError: QueuePool limit of size 50 overflow 0 reached, connection timed out, timeout 30.00\npgbouncer: no more connections allowed (max_client_conn) while running ledger.jobs.invoice_batch with AsyncSession"),
    ("n013", "uv run pytest -m slow", "E   psycopg.OperationalError: connection to server at \"localhost\" (127.0.0.1), port 5432 failed: Connection refused\nFAILED tests/integration/test_invoices.py::test_totals - psycopg.OperationalError\n==== 1 failed, 12 errors in 2.03s ===="),
    ("n014", "make dev", "uvicorn: ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use\nmake: *** [dev] Error 1"),
    ("n123", "uv run python -m ledger.jobs.invoice_batch", "invoice_batch ERROR: priced 312 multi-currency invoices with stale ECB FX rates (rates dated yesterday); FX import had not landed yet"),
    ("n126", "uv run alembic check", "alembic ERROR: autogenerate did not detect the server_default change on invoices.status; migration 0042 is incomplete\nFAILED: Target database is not up to date."),
];

/// Facts the user states during the session (new: not in the store). The pre-compaction flush should capture them.
const E5_PLANTED: &[(&str, &str)] = &[
    ("Remember that the staging database snapshot job runs at 04:30 UTC and takes about 20 minutes.", "04:30"),
    ("Remember: the invoice PDF renderer is pinned to WeasyPrint 62 because 63 broke page breaks.", "WeasyPrint"),
    ("From now on, name feature branches lg/<ticket>-<slug> in ledger-api.", "lg/<ticket>"),
    ("Keep in mind that the tenants table is partitioned by region since the March migration.", "partitioned by region"),
    ("Remember that credit notes are numbered CN-<year>-<seq> and the sequence resets every January.", "CN-<year>"),
    ("Note that the ledger-api staging environment uses the EU Stripe sandbox account.", "EU Stripe sandbox"),
];

const E5_CORRECTIONS: &[&str] = &[
    "No, don't use pip for this, use uv add instead.",
    "That's wrong: amounts are stored in cents, not Decimal.",
    "不对，部署要用 ops/deploy.sh，不要直接 kubectl apply。",
];

enum Step {
    Prompt(String),
    /// a task: prompt with a chain of tool outputs (each may be an error with a gold note)
    Task(String, Vec<(String, String, Option<&'static str>)>),
}

/// The 120-request script (requests are counted, not steps). Deterministic.
fn e5_script(queries: &[String]) -> Vec<Step> {
    let mut v = vec![];
    let mut err_i = 0usize;
    let mut q = 0usize;
    for t in 0..80 {
        // every 3rd step a task with 2-3 tool calls; errors on every other task
        if t % 6 == 2 && t / 6 < E5_PLANTED.len() {
            v.push(Step::Prompt(E5_PLANTED[t / 6].0.to_string()));
        } else if t % 3 == 1 {
            let mut chain = vec![];
            let len = 2 + (t / 3) % 2;
            for k in 0..len {
                let err = (t / 3) % 2 == 0 && k == 1;
                if err {
                    let (gold, cmd, out) = E5_ERRORS[err_i % E5_ERRORS.len()];
                    err_i += 1;
                    chain.push((cmd.to_string(), format!("{}\n{out}", log_output(t * 10 + k, 2_000)), Some(gold)));
                } else {
                    chain.push((format!("uv run pytest tests/unit -k part{t}_{k}"), log_output(t * 10 + k, 24_000), None));
                }
            }
            let first = format!("Work item {t}: run the checks for this change and fix what fails. [[run:{}]]", chain[0].0);
            v.push(Step::Task(first, chain));
        } else if t % 16 == 5 {
            v.push(Step::Prompt(E5_CORRECTIONS[(t / 16) % E5_CORRECTIONS.len()].to_string()));
        } else if t % 12 == 9 {
            v.push(Step::Prompt(format!(
                "[[recall]] Before you continue, check what memory says about our test and migration conventions (step {t})."
            )));
        } else {
            v.push(Step::Prompt(queries[q % queries.len()].clone()));
            q += 1;
        }
    }
    v
}

struct E5Cfg {
    name: &'static str,
    budgets: bool,
    dedup: bool,
    flush: bool,
    /// add 40 long ledger-api notes (≈2 kchar content + ≈1 kchar why each): the review's "realistic heavy" regime
    heavy: bool,
}

/// Long, relevant notes for the heavy variant.
fn heavy_notes() -> Vec<(String, String)> {
    let topics = [
        "tests",
        "deploys",
        "migrations",
        "webhooks",
        "the invoice batch",
        "rate limits",
        "money columns",
        "SQLAlchemy",
        "Postgres",
        "reviews",
    ];
    (0..40)
        .map(|i| {
            let t = topics[i % topics.len()];
            let body = format!("ledger-api runbook {i} on {t}: ") + &format!("when working on {t} in ledger-api check the {t} checklist item {i}, run the {t} verification step, compare against the {t} baseline and record the outcome in the {t} log; ").repeat(9);
            let why = format!("learned the hard way during incident {i} on {t}: ") + &"the previous approach silently produced wrong results under load and took a day to diagnose; ".repeat(10);
            (body, why)
        })
        .collect()
}

async fn e5_run(root: &Path, vc: &E5Cfg) -> Result<Value> {
    let mut cfg = bench_config()?;
    cfg.inject.budgets = vc.budgets;
    cfg.inject.dedup = vc.dedup;
    if !vc.flush {
        cfg.inject.flush_at = 0.0;
    }
    cfg.distill.enabled = true; // heuristic distiller: only the flush / session-end triggers run it here
    let ds = load_ds(&root.join("bench/data/devmem.json"))?;
    let tmp = TempDir::new("e5")?;
    let corpus = Corpus::build(&tmp.0.join("ws"), &ds)?;
    let cwd = corpus.cwd("ledger-api");
    let mock = Arc::new(Mock5::default());
    let mock_url = serve_nodelay(Router::new().route("/v1/messages", post(mock5_handler)).with_state(mock.clone())).await?;
    let (gw, _nd, engine) = gateway_env(&cfg, &ds, &corpus, &tmp.0.join("e5.db"), &mock_url, true).await?;
    if vc.heavy {
        let ctx = engine.ctx_for("claude-code", "seed", Some(&cwd));
        for (c, w) in heavy_notes() {
            engine
                .write(
                    &ctx,
                    WriteReq {
                        content: c,
                        why: Some(w),
                        kind: Some("procedure".into()),
                        scope: Some("project".into()),
                        ..Default::default()
                    },
                )
                .await?;
        }
    }
    let m2n: HashMap<String, String> = {
        // map store ids back to dataset ids by content
        let mut m = HashMap::new();
        for mem in engine.store.list_memories(None, true, 100_000) {
            if let Some(n) = ds.notes.iter().find(|n| n.content.trim() == mem.content) {
                m.insert(mem.id.clone(), n.id.clone());
            }
        }
        m
    };
    let queries: Vec<String> = ds.queries.iter().filter(|q| q.at == "ledger-api" && !q.gold.is_empty()).map(|q| q.q.clone()).collect();
    let script = e5_script(&queries);
    let client = http_client();
    let system = cc_system(&cwd);
    let tools = cc_tools();
    let session = session_uuid(0xE5000000, hash_hex(&[vc.name]).len() as u64);
    let window = 200_000f64;
    let id_re = regex::Regex::new(r"m_[0-9a-z]{26}").unwrap();
    let est = |v: &Value| unimem::budget::estimate_tokens(&v.to_string());

    let mut msgs: Vec<Value> = vec![];
    let mut rows: Vec<Value> = vec![];
    let mut req = 0usize;
    let (mut dup_new, mut new_total, mut cum_new) = (0usize, 0usize, 0usize);
    let (mut err_turns, mut err_hits, mut err_avail, mut benign_tool_turns, mut benign_injections) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut err_detail = vec![];
    let mut compaction: Value = Value::Null;
    let (mut max_live_share, mut max_ctx_share) = (0f64, 0f64);
    let mut planted_at_compaction = 0usize;
    let mut planted_before = 0usize;
    let mut compacted = false;
    let mut post_compaction_preamble: Option<bool> = None;
    let mut corrections_hinted = 0usize;
    let mut flush_turn: Option<usize> = None;

    let post = |client: &reqwest::Client, body: Value| {
        let gw = gw.clone();
        let session = session.clone();
        let client = client.clone();
        async move {
            let r = client
                .post(format!("{gw}/v1/messages"))
                .header("x-api-key", "sk-bench")
                .header("anthropic-version", "2023-06-01")
                .header("user-agent", "claude-cli/2.0.0 (external, cli)")
                .header("x-claude-code-session-id", &session)
                .json(&body)
                .send()
                .await?;
            anyhow::ensure!(r.status().is_success(), "status {}", r.status());
            Ok::<Value, anyhow::Error>(r.json().await?)
        }
    };
    let planted_count = |e: &Engine| -> usize {
        let all = e.store.list_memories(None, false, 100_000);
        E5_PLANTED.iter().filter(|(_, key)| all.iter().any(|m| m.content.contains(key))).count()
    };

    let mut steps = script.into_iter();
    'outer: while req < 120 {
        // forced compaction at request 60: the harness asks for a summary, then replaces the history with it
        if req >= 60 && !compacted {
            compacted = true;
            let mut cm = msgs.clone();
            cm.push(json!({"role":"user","content":[{"type":"text","text":"Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions."}]}));
            let body = agent_body(&system, &tools, &cm, false);
            let before = mock.seen.lock().len();
            post(&client, body.clone()).await?;
            let up = mock.seen.lock().get(before).cloned().unwrap_or(Value::Null);
            // let a background distillation (pre-compaction flush) finish
            for _ in 0..100 {
                if engine.store.undistilled_episodes(&session).iter().all(|e| e.role != "user") {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            planted_at_compaction = planted_count(&engine);
            compaction = json!({"at_request": req, "request_passed_through_untouched": up == body, "messages_before": msgs.len()});
            msgs = vec![
                json!({"role":"user","content":[{"type":"text","text":"This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:\nThe user is working on ledger-api billing: tests, webhooks, the invoice batch and migrations."}]}),
                json!({"role":"assistant","content":[{"type":"text","text":"Understood. Continuing from the summary."}]}),
            ];
        }
        let Some(step) = steps.next() else { break };
        let (prompt, chain) = match step {
            Step::Prompt(p) => (p, vec![]),
            Step::Task(p, c) => (p, c),
        };
        if !compacted && E5_PLANTED.iter().any(|x| x.0 == prompt) {
            planted_before += 1;
        }
        msgs.push(json!({"role":"user","content":[{"type":"text","text": prompt}]}));
        let mut tool_i = 0usize;
        let mut expect: Option<&'static str> = None;
        loop {
            if req >= 120 {
                break 'outer;
            }
            let body = agent_body(&system, &tools, &msgs, false);
            let before = mock.seen.lock().len();
            let vis = post(&client, body.clone()).await?;
            req += 1;
            let ups: Vec<Value> = mock.seen.lock()[before..].to_vec();
            let up = ups.first().cloned().unwrap_or(Value::Null);
            // what unimem added to the live context = upstream messages − harness messages (incl. replayed hidden rounds)
            let live = est(&up["messages"]).saturating_sub(est(&body["messages"]));
            let ctx_tokens = est(&up);
            let last_up = up["messages"].as_array().and_then(|a| a.last()).cloned().unwrap_or(Value::Null);
            let new_blocks: Vec<String> = blocks_of(&last_up["content"])
                .iter()
                .filter_map(|b| b["text"].as_str())
                .filter(|t| t.contains("<unimem"))
                .map(String::from)
                .collect();
            let new_text = new_blocks.join("\n");
            let new_tokens = unimem::budget::estimate_tokens(&new_text);
            let earlier = {
                let mut m = up["messages"].as_array().cloned().unwrap_or_default();
                m.pop();
                Value::Array(m).to_string()
            };
            let new_ids: Vec<String> = id_re.find_iter(&new_text).map(|m| m.as_str().to_string()).collect();
            let dups = new_ids.iter().filter(|id| earlier.contains(id.as_str())).count();
            dup_new += dups;
            new_total += new_ids.len();
            cum_new += new_tokens;
            max_live_share = max_live_share.max(live as f64 / window);
            max_ctx_share = max_ctx_share.max(ctx_tokens as f64 / window);
            if new_text.contains("Context is nearly full") && flush_turn.is_none() {
                flush_turn = Some(req);
            }
            if new_text.contains("The user just corrected you") {
                corrections_hinted += 1;
            }
            let kind = if tool_i == 0 { "prompt" } else { "tool" };
            if compacted && post_compaction_preamble.is_none() {
                post_compaction_preamble = Some(new_text.contains("<unimem-context>"));
            }
            if kind == "tool" {
                let tool_recall_ids: Vec<String> = new_blocks
                    .iter()
                    .filter(|b| b.contains("tool-error"))
                    .flat_map(|b| id_re.find_iter(b).map(|m| m.as_str().to_string()).collect::<Vec<_>>())
                    .collect();
                match expect {
                    Some(gold) => {
                        err_turns += 1;
                        let hit = tool_recall_ids.iter().any(|id| m2n.get(id).map(String::as_str) == Some(gold));
                        // the gold note was already in the live context (preamble, earlier recall): no injection needed
                        let live_already = id_re.find_iter(&earlier).any(|m| m2n.get(m.as_str()).map(String::as_str) == Some(gold));
                        err_hits += hit as usize;
                        err_avail += (hit || live_already) as usize;
                        err_detail.push(json!({"request": req, "gold": gold, "hit": hit, "already_live": live_already, "injected": tool_recall_ids.iter().map(|i| m2n.get(i).cloned().unwrap_or(i.clone())).collect::<Vec<_>>()}));
                    }
                    None => {
                        benign_tool_turns += 1;
                        benign_injections += (!tool_recall_ids.is_empty()) as usize;
                    }
                }
            }
            rows.push(json!({"request": req, "kind": kind, "context_tokens": ctx_tokens, "unimem_live_tokens": live, "new_injected_tokens": new_tokens, "cumulative_new_tokens": cum_new, "new_note_ids": new_ids.len(), "duplicate_new_ids": dups, "upstream_rounds": ups.len()}));
            let content = vis["content"].as_array().cloned().unwrap_or_default();
            msgs.push(json!({"role":"assistant","content": content.clone()}));
            let bash: Vec<String> = content
                .iter()
                .filter(|b| b["type"] == "tool_use" && b["name"] == "Bash")
                .filter_map(|b| b["id"].as_str().map(String::from))
                .collect();
            if bash.is_empty() {
                break;
            }
            // the harness runs the command: scripted output (+ the next command marker for chains)
            let (_, out, gold) = chain.get(tool_i).cloned().unwrap_or_default();
            tool_i += 1;
            expect = gold;
            let next = chain.get(tool_i).map(|c| format!("\n[[run:{}]]", c.0)).unwrap_or_default();
            msgs.push(json!({"role":"user","content": bash.iter().map(|id| json!({"type":"tool_result","tool_use_id": id, "content": format!("{out}{next}")})).collect::<Vec<_>>()}));
        }
    }
    let planted_end = planted_count(&engine);
    let st_rows: Vec<&Value> = rows.iter().collect();
    let live_series: Vec<f64> = st_rows.iter().map(|r| r["unimem_live_tokens"].as_f64().unwrap_or(0.0)).collect();
    Ok(json!({
        "variant": vc.name, "budgets": vc.budgets, "dedup": vc.dedup, "flush": vc.flush,
        "requests": req, "window": window,
        "summary": {
            "max_unimem_live_tokens": live_series.iter().cloned().fold(0.0, f64::max),
            "max_unimem_context_share": r3(max_live_share),
            "max_total_context_share": r3(max_ctx_share),
            "cumulative_new_injected_tokens": cum_new,
            "new_note_injections": new_total,
            "duplicate_injections": dup_new,
            "compaction": compaction,
            "preamble_reinjected_after_compaction": post_compaction_preamble,
            "tool_error_turns": err_turns, "tool_error_recall_hits": err_hits, "tool_error_gold_in_context": err_avail,
            "benign_tool_turns": benign_tool_turns, "benign_tool_turn_injections": benign_injections,
            "corrections_hinted": corrections_hinted,
            "flush_nudge_at_request": flush_turn,
            "planted_facts": E5_PLANTED.len(), "planted_before_compaction": planted_before, "planted_captured_by_compaction": planted_at_compaction, "planted_captured_by_end": planted_end,
        },
        "tool_error_detail": err_detail,
        "per_request": rows,
    }))
}

async fn e5(root: &Path) -> Result<Value> {
    let variants = [
        E5Cfg { name: "v0.4", budgets: true, dedup: true, flush: true, heavy: false },
        E5Cfg { name: "no-flush", budgets: true, dedup: true, flush: false, heavy: false },
        E5Cfg { name: "no-budget", budgets: false, dedup: true, flush: true, heavy: false },
        E5Cfg { name: "no-budget-no-dedup", budgets: false, dedup: false, flush: true, heavy: false },
        E5Cfg { name: "heavy/v0.4", budgets: true, dedup: true, flush: true, heavy: true },
        E5Cfg { name: "heavy/no-budget", budgets: false, dedup: true, flush: true, heavy: true },
        E5Cfg { name: "heavy/no-budget-no-dedup", budgets: false, dedup: false, flush: true, heavy: true },
    ];
    let mut out = serde_json::Map::new();
    for v in &variants {
        let r = e5_run(root, v).await?;
        println!("[e5] {:20} {}", v.name, r["summary"]);
        out.insert(v.name.to_string(), r);
    }
    Ok(json!({
        "experiment": "E5 long session",
        "config": config_info(&bench_config()?),
        "setup": "120 client requests of a Claude-Code-like session on ledger-api (devmem corpus loaded; 200k window): prompts from the ledger-api queries, 6 planted user facts, 3 corrections, tasks with 2-3 Bash calls each (24k-char logs; every other task has one error whose root cause is a stored note), 4 prompts that make the model call unimem_recall (hidden rounds). At request 60 the harness sends its compaction request and replaces the history with a summary. 'heavy' variants add 40 long relevant notes (≈2 kchar + ≈1 kchar why). Token counts are the gateway's estimator (budget.rs). 'unimem_live_tokens' = upstream messages minus harness messages (replayed augmentations + hidden rounds).",
        "variants": out,
    }))
}

// ============================================================================ E3 project identification

struct Case {
    name: String,
    cwd: PathBuf,
    expect: Option<String>,
    got: Option<String>,
}

async fn e3(_root: &Path) -> Result<Value> {
    let tmp = TempDir::new("e3")?;
    let b = tmp.0.clone();
    let old_home = std::env::var("HOME").ok();
    let home = b.join("home");
    std::fs::create_dir_all(&home)?;
    std::env::set_var("HOME", &home);
    let r = e3_inner(&b, &home).await;
    match old_home {
        Some(h) => std::env::set_var("HOME", h),
        None => std::env::remove_var("HOME"),
    }
    r
}

async fn e3_inner(b: &Path, home: &Path) -> Result<Value> {
    let code = home.join("code");
    let a = "github.com/acme/alpha".to_string();
    let local = |p: &Path| Some(format!("local{}", p.display()));
    let mut cases: Vec<Case> = vec![];
    let mut add = |name: &str, cwd: PathBuf, expect: Option<String>| cases.push(Case { name: name.into(), cwd, expect, got: None });

    let alpha = code.join("alpha");
    mk_repo(
        &alpha,
        &[("origin", "git@github.com:acme/alpha.git")],
        &[("README.md", "# alpha\n\nAlpha is the main web service.\n"), ("src/deep/nested/mod.rs", "// x\n")],
    )?;
    add("repo root, scp-style ssh remote", alpha.clone(), Some(a.clone()));
    add("nested subdir (3 levels)", alpha.join("src/deep/nested"), Some(a.clone()));
    let variants = [
        ("clone elsewhere, https remote", "https://github.com/acme/alpha.git"),
        ("https remote with token userinfo", "https://x-access-token:abc123@github.com/acme/alpha"),
        ("ssh:// remote", "ssh://git@github.com/acme/alpha.git"),
        ("uppercase owner/repo", "git@github.com:Acme/Alpha.git"),
        ("https remote, trailing slash", "https://github.com/acme/alpha/"),
        ("ssh:// remote with port 22", "ssh://git@github.com:22/acme/alpha.git"),
        ("git:// remote", "git://github.com/acme/alpha.git"),
        ("http remote without .git", "http://github.com/acme/alpha"),
    ];
    for (i, (n, u)) in variants.iter().enumerate() {
        let d = b.join(format!("clones/c{i}/alpha-copy"));
        mk_repo(&d, &[("origin", u)], &[])?;
        add(n, d, Some(a.clone()));
    }
    // worktrees
    let wt = code.join("alpha-feature");
    git(&alpha, &["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "feature"])?;
    add("git worktree (git worktree add)", wt.clone(), Some(a.clone()));
    std::fs::create_dir_all(wt.join("src/deep"))?;
    add("subdir of a git worktree", wt.join("src/deep"), Some(a.clone()));
    {
        // hand-made worktree with a relative gitdir
        let wtd = alpha.join(".git/worktrees/manual");
        std::fs::create_dir_all(&wtd)?;
        std::fs::write(wtd.join("HEAD"), "ref: refs/heads/main\n")?;
        std::fs::write(wtd.join("commondir"), "../..\n")?;
        let m = code.join("alpha-manual");
        std::fs::create_dir_all(&m)?;
        std::fs::write(wtd.join("gitdir"), format!("{}/.git\n", m.display()))?;
        std::fs::write(m.join(".git"), "gitdir: ../alpha/.git/worktrees/manual\n")?;
        add("worktree with relative gitdir + commondir", m, Some(a.clone()));
    }
    let detached = b.join("clones/detached/alpha");
    mk_repo(&detached, &[("origin", "git@github.com:acme/alpha.git")], &[])?;
    git(&detached, &["checkout", "-q", "--detach"])?;
    add("detached HEAD clone", detached, Some(a.clone()));
    // distinct repos
    let beta = code.join("beta");
    mk_repo(&beta, &[("origin", "git@github.com:acme/beta.git")], &[])?;
    add("different repo, same owner", beta.clone(), Some("github.com/acme/beta".into()));
    let ghe = b.join("work/alpha");
    mk_repo(&ghe, &[("origin", "git@github.acme.corp:acme/alpha.git")], &[])?;
    add("same owner/repo on another host (GHE)", ghe, Some("github.acme.corp/acme/alpha".into()));
    let gl = code.join("tool");
    mk_repo(&gl, &[("origin", "git@gitlab.com:grp/sub/tool.git")], &[])?;
    add("gitlab nested groups", gl, Some("gitlab.com/grp/sub/tool".into()));
    let infra = code.join("infra");
    mk_repo(&infra, &[("origin", "ssh://git@git.acme.dev:2222/platform/infra.git")], &[])?;
    add("ssh:// with non-standard port", infra, Some("git.acme.dev/platform/infra".into()));
    let fork = code.join("fork");
    mk_repo(&fork, &[("upstream", "git@github.com:big/project.git"), ("origin", "git@github.com:me/project.git")], &[])?;
    add("fork: upstream listed before origin", fork, Some("github.com/me/project".into()));
    let up_only = code.join("uponly");
    mk_repo(&up_only, &[("upstream", "https://github.com/big/uponly.git")], &[])?;
    add("only a non-origin remote", up_only, Some("github.com/big/uponly".into()));
    // local repos
    let loc = code.join("scratch");
    mk_repo(&loc, &[], &[("notes/a.txt", "x\n")])?;
    add("git repo without remote (root)", loc.clone(), local(&loc));
    add("git repo without remote (subdir)", loc.join("notes"), local(&loc));
    let lwt = code.join("scratch-wt");
    git(&loc, &["worktree", "add", "-q", lwt.to_str().unwrap(), "-b", "wt"])?;
    add("worktree of a repo without remote", lwt, local(&loc));
    // submodule-like and nested repos
    {
        let sm = alpha.join("libs/sublib");
        std::fs::create_dir_all(&sm)?;
        let md = alpha.join(".git/modules/sublib");
        std::fs::create_dir_all(&md)?;
        std::fs::write(md.join("HEAD"), "ref: refs/heads/main\n")?;
        std::fs::write(md.join("config"), "[core]\n\tbare = false\n[remote \"origin\"]\n\turl = git@github.com:acme/sublib.git\n")?;
        std::fs::write(sm.join(".git"), "gitdir: ../../.git/modules/sublib\n")?;
        add("git submodule (gitdir in .git/modules)", sm, Some("github.com/acme/sublib".into()));
    }
    let inner = alpha.join("vendor/inner");
    mk_repo(&inner, &[("origin", "https://github.com/other/inner.git")], &[])?;
    add("independent repo nested in another repo", inner, Some("github.com/other/inner".into()));
    let mono = alpha.join("packages/web");
    write_files(&mono, &[("package.json", "{\"name\":\"@acme/web\",\"description\":\"web package\"}")])?;
    add("monorepo package with its own manifest", mono, Some(a.clone()));
    // non-git
    let cargo = b.join("plain/rusty");
    write_files(&cargo, &[("Cargo.toml", "[package]\nname = \"rusty\"\ndescription = \"plain crate\"\n")])?;
    add("non-git dir with Cargo.toml", cargo.clone(), local(&cargo));
    let readme = b.join("plain/docs-only");
    write_files(&readme, &[("README.md", "# Docs\n\nJust some documentation for a side project, not under git yet.\n")])?;
    add("non-git dir with README only", readme, local(&b.join("plain/docs-only")));
    let pkg = b.join("plain/nodey");
    write_files(&pkg, &[("package.json", "{\"name\":\"nodey\",\"description\":\"node app\"}"), ("src/lib/x.js", "1\n")])?;
    add("subdir of non-git manifest dir", pkg.join("src/lib"), local(&pkg));
    let bare = b.join("plain/empty");
    std::fs::create_dir_all(&bare)?;
    add("empty non-project dir", bare, None);
    // home
    add("home dir", home.to_path_buf(), None);
    mk_repo(home, &[("origin", "git@github.com:me/dotfiles.git")], &[(".bashrc", "x\n")])?;
    let dl = home.join("Downloads");
    std::fs::create_dir_all(&dl)?;
    add("home dir that is a dotfiles repo", home.to_path_buf(), None);
    add("plain subdir of dotfiles-repo home", dl, None);
    add("filesystem root", PathBuf::from("/"), None);
    // symlinks
    let ln_a = b.join("links/alpha-link");
    std::fs::create_dir_all(b.join("links"))?;
    std::os::unix::fs::symlink(&alpha, &ln_a)?;
    add("symlink to a repo with remote", ln_a, Some(a.clone()));
    let ln_l = b.join("links/scratch-link");
    std::os::unix::fs::symlink(&loc, &ln_l)?;
    add("symlink to a repo without remote", ln_l, local(&loc));

    let store = Store::open_memory()?;
    for c in cases.iter_mut() {
        let r = Resolver::default();
        c.got = r.resolve(&store, c.cwd.to_str().unwrap()).map(|x| x.project.pid);
    }
    // legacy: first seen without remote, remote added later → identity kept
    let leg = code.join("legacy");
    mk_repo(&leg, &[], &[])?;
    let first = Resolver::default().resolve(&store, leg.to_str().unwrap()).map(|x| x.project.pid);
    git(&leg, &["remote", "add", "origin", "git@github.com:me/legacy.git"])?;
    let later = Resolver::default().resolve(&store, leg.to_str().unwrap()).map(|x| x.project.pid);
    cases.push(Case {
        name: "repo gains a remote after first sight (identity kept)".into(),
        cwd: leg.clone(),
        expect: first.clone(),
        got: later,
    });

    let ok = cases.iter().filter(|c| c.got == c.expect).count();
    let rows: Vec<Value> = cases
        .iter()
        .map(|c| json!({"case": c.name, "cwd": c.cwd.display().to_string().replace(&b.display().to_string(), "$TMP"), "expect": c.expect.as_ref().map(|s| s.replace(&b.display().to_string(), "$TMP")), "got": c.got.as_ref().map(|s| s.replace(&b.display().to_string(), "$TMP")), "ok": c.got == c.expect}))
        .collect();
    println!("[e3] {ok}/{} correct", cases.len());
    for r in rows.iter().filter(|r| r["ok"] == false) {
        println!("[e3] FAIL {}: expect {} got {}", r["case"], r["expect"], r["got"]);
    }
    Ok(
        json!({"experiment": "E3 project identification", "cases": cases.len(), "correct": ok, "accuracy": r3(ok as f64 / cases.len() as f64), "rows": rows}),
    )
}

// ============================================================================ E4 throughput

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn n(&mut self, k: u64) -> u64 {
        self.next() % k
    }
    fn pick<'a>(&mut self, v: &'a [&'a str]) -> &'a str {
        v[self.n(v.len() as u64) as usize]
    }
}

const SVC: &[&str] = &[
    "auth",
    "billing",
    "search",
    "gateway",
    "ingest",
    "worker",
    "scheduler",
    "notifier",
    "reports",
    "exporter",
    "importer",
    "ledger",
    "catalog",
    "media",
    "thumbnails",
    "webhooks",
    "metrics",
    "audit",
    "payments",
    "profiles",
];
const TECH: &[&str] = &[
    "Postgres",
    "Redis",
    "Kafka",
    "SQLite",
    "ClickHouse",
    "NATS",
    "gRPC",
    "GraphQL",
    "FastAPI",
    "Axum",
    "Express",
    "Celery",
    "Temporal",
    "Terraform",
    "Helm",
    "Docker",
    "Nix",
    "pnpm",
    "uv",
    "Bazel",
];
const PEOPLE: &[&str] = &["Alice", "Bob", "Chen Wei", "Dmitri", "Priya", "Sam", "Yuki", "Lena", "Omar", "Ines"];
const ENVS: &[&str] = &["dev", "staging", "prod", "qa", "canary"];
const REASONS: &[&str] = &[
    "the old client leaked connections under load",
    "it halved p95 latency in the load test",
    "the vendor dropped support",
    "the team already knows it",
    "it removed a whole deploy step",
    "licensing costs went up",
    "it made local dev reproducible",
];

fn synth_note(r: &mut Rng) -> (String, Option<String>) {
    let svc = format!("{}-{}", r.pick(SVC), r.n(60));
    let t = r.pick(TECH);
    let t2 = r.pick(TECH);
    let env = r.pick(ENVS);
    let p = r.pick(PEOPLE);
    match r.n(10) {
        0 => (format!("The {svc} service listens on port {} in {env}.", 3000 + r.n(6000)), None),
        1 => (format!("The {svc} nightly job runs at {:02}:{:02} UTC on the {env} cluster.", r.n(24), r.n(4) * 15), None),
        2 => (format!("To deploy {svc} run `make deploy-{svc} ENV={env}` from the repo root, never by hand."), None),
        3 => (format!("Gotcha: {svc} fails to start when {}_URL is unset; set it in .env.{env}.", r.pick(SVC).to_uppercase()), None),
        4 => (format!("We decided to use {t} for {svc} instead of {t2}."), Some(r.pick(REASONS).to_string())),
        5 => (format!("{p} owns the {svc} on-call rotation and reviews its pull requests."), None),
        6 => (format!("{svc} 服务的配置在 deploy/{env}/{svc}.yaml，修改后需要重启 {env} 环境的 pod。"), None),
        7 => (format!("I prefer {t} over {t2} for {} work, use it by default.", r.pick(SVC)), None),
        8 => (format!("{svc} keeps its schema migrations in migrations/{svc}; run them with `{t} migrate up` before tests."), None),
        _ => (format!("{svc} 的告警阈值是 p95 {} 毫秒，超过后通知 {p}。", 100 + r.n(900)), None),
    }
}

fn synth_query(r: &mut Rng) -> String {
    let svc = format!("{}-{}", r.pick(SVC), r.n(60));
    match r.n(6) {
        0 => format!("which port does {svc} use in {}?", r.pick(ENVS)),
        1 => format!("how do I deploy {svc}?"),
        2 => format!("who is on call for {svc}?"),
        3 => format!("{svc} 的配置文件在哪里？"),
        4 => format!("why did we pick {} for {svc}?", r.pick(TECH)),
        _ => format!("when does the {svc} nightly job run?"),
    }
}

fn db_size(p: &Path) -> u64 {
    ["", "-wal", "-shm"].iter().map(|s| std::fs::metadata(format!("{}{s}", p.display())).map(|m| m.len()).unwrap_or(0)).sum()
}

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|x| x.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

async fn e4(_root: &Path) -> Result<Value> {
    let cfg = bench_config()?;
    let tmp = TempDir::new("e4")?;
    let mut dirs = vec![];
    for i in 0..6 {
        let d = tmp.0.join(format!("ws/proj{i}"));
        mk_repo(
            &d,
            &[("origin", &format!("git@github.com:bench/proj{i}.git"))],
            &[("README.md", &format!("# proj{i}\n\nSynthetic project number {i} for the throughput bench.\n"))],
        )?;
        dirs.push(d.display().to_string());
    }
    let db = tmp.0.join("e4.db");
    let e = make_engine(&cfg, Store::open(&db)?);
    let target: usize = std::env::var("UNIMEM_BENCH_E4_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let checkpoints = [1_000usize, target];
    let mut r = Rng(0x9E3779B97F4A7C15);
    let mut stored = 0usize;
    let mut attempts = 0usize;
    let mut window: Vec<f64> = vec![];
    let mut outcome_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut tiers = vec![];
    let t_all = Instant::now();
    for &cp in &checkpoints {
        while stored < cp && attempts < target * 3 {
            let (c, why) = synth_note(&mut r);
            let at = r.n(7) as usize;
            let cwd = if at == 6 { None } else { Some(dirs[at].as_str()) };
            let h = if r.n(2) == 0 { "claude-code" } else { "codex" };
            let ctx = e.ctx_for(h, &format!("s{}", attempts / 20), cwd);
            let t = Instant::now();
            let o = e.write(&ctx, WriteReq { content: c, why, ..Default::default() }).await?;
            window.push(ms(t));
            if window.len() > 500 {
                window.remove(0);
            }
            attempts += 1;
            let k = match o {
                WriteOutcome::Stored { superseded, .. } => {
                    stored += 1;
                    if superseded.is_empty() {
                        "stored"
                    } else {
                        "stored_superseding"
                    }
                }
                WriteOutcome::Duplicate { .. } => "duplicate",
                WriteOutcome::Rejected { .. } => "rejected",
            };
            *outcome_counts.entry(k).or_default() += 1;
        }
        let write_lat = lat(&window);
        let mut rec = vec![];
        let mut rec_p = vec![];
        let mut returned = 0usize;
        for i in 0..300 {
            let q = synth_query(&mut r);
            let at = i % 7;
            let ctx = e.ctx_for("codex", "q", if at == 6 { None } else { Some(dirs[at].as_str()) });
            let t = Instant::now();
            let out = e.recall(&ctx, RecallReq { query: q.clone(), project: None, k: 5, passive: false, ..Default::default() }).await?;
            rec.push(ms(t));
            returned += out.notes.len();
            let t = Instant::now();
            let _ = e.recall(&ctx, RecallReq { query: q, project: None, k: 5, passive: true, ..Default::default() }).await?;
            rec_p.push(ms(t));
        }
        let size_live = db_size(&db);
        {
            let c = rusqlite::Connection::open(&db)?;
            c.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        }
        let size_main = std::fs::metadata(&db).map(|m| m.len()).unwrap_or(0);
        let live: i64 = e.store.count_memories().values().sum();
        println!(
            "[e4] {stored} stored ({attempts} writes): write p50 {} p95 {}; recall p50 {} p95 {}; db {:.1} MB",
            write_lat["p50_ms"],
            write_lat["p95_ms"],
            r3(pct(&rec, 50.0)),
            r3(pct(&rec, 95.0)),
            size_main as f64 / 1e6
        );
        tiers.push(json!({
            "stored_notes": stored, "live_notes": live, "write_attempts": attempts,
            "write_latency_last_500": write_lat, "recall_latency": lat(&rec), "recall_passive_latency": lat(&rec_p),
            "notes_returned_per_recall": r3(returned as f64 / 300.0),
            "db_bytes_with_wal": size_live, "db_bytes_after_checkpoint": size_main, "bytes_per_note": size_main / stored.max(1) as u64,
            "rss_mb": r3(rss_mb()), "elapsed_s": r3(t_all.elapsed().as_secs_f64()),
        }));
    }
    Ok(json!({"experiment": "E4 engine throughput", "config": config_info(&cfg), "outcomes": outcome_counts, "tiers": tiers,
        "notes": "synthetic templated notes over 20 service families × 60 instances, 6 git projects + global; single-threaded sequential writes and recalls; exact (brute-force) vector search."}))
}

// ============================================================================ summary

fn env_info() -> Value {
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
    let cpuinfo = read("/proc/cpuinfo");
    let model = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name") || l.starts_with("Model") || l.starts_with("CPU part"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mem_kb: f64 = read("/proc/meminfo")
        .lines()
        .find(|l| l.starts_with("MemTotal"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(0.0);
    let lscpu =
        std::process::Command::new("lscpu").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let ls = |k: &str| {
        lscpu
            .lines()
            .find(|l| l.trim_start().starts_with(k))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    let model = if model.is_empty() || model.starts_with("0x") {
        format!("{} {}", ls("Vendor ID"), ls("Model name")).trim().to_string()
    } else {
        model
    };
    let os = read("/etc/os-release")
        .lines()
        .find(|l| l.starts_with("PRETTY_NAME="))
        .map(|l| l.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string())
        .unwrap_or_default();
    let cmd = |c: &str, a: &[&str]| {
        std::process::Command::new(c)
            .args(a)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    json!({
        "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "cpu_model": model, "arch": std::env::consts::ARCH,
        "ram_gb": r3(mem_kb / 1024.0 / 1024.0), "os": os, "kernel": cmd("uname", &["-r"]),
        "rustc": cmd("rustc", &["--version"]), "git_commit": cmd("git", &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--short", "HEAD"]),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
    })
}

fn f(v: &Value) -> String {
    match v {
        Value::Null => "–".into(),
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            if n.is_i64() || n.is_u64() {
                format!("{x}")
            } else {
                format!("{x:.3}")
            }
        }
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn pctf(v: &Value) -> String {
    match v.as_f64() {
        Some(x) => format!("{:.1}%", x * 100.0),
        None => "–".into(),
    }
}

fn summary(out: &Path, data: &Path) -> Result<()> {
    let load = |n: &str| std::fs::read_to_string(out.join(n)).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let env = load("env.json").unwrap_or_else(env_info);
    let mut s = String::new();
    s.push_str("# unimem evaluation — results\n\n");
    s.push_str(&format!("Generated by `cargo run --release --bin unimem-bench -- <exp> --data {} --out {}`. Raw numbers: `*.json` next to this file. E1 dataset: `{}`; E2 dataset: `bench/data/devmem.json`.\n\n", data.display(), out.display(), data.display()));
    s.push_str("## Environment\n\n| | |\n|---|---|\n");
    for k in ["cpus", "cpu_model", "arch", "ram_gb", "os", "kernel", "rustc", "git_commit", "profile"] {
        s.push_str(&format!("| {k} | {} |\n", f(&env[k])));
    }
    if let Some(e1) = load("e1_memory.json") {
        s.push_str(&format!("\nDecider: `{}` · embedder: `{}` (offline; swap via `UNIMEM_BENCH_CONFIG` / `UNIMEM_BENCH_DECIDER` / `UNIMEM_BENCH_EMBEDDER`).\n", f(&e1["config"]["decider"]), f(&e1["config"]["embedder"])));
        let d = &e1["dataset"];
        s.push_str(&format!("\n## E1 — memory quality\n\nCorpus: {} projects + global, {} notes written in order, {} queries ({} answerable cross-harness, {} same-harness).\n\n", f(&d["projects"]), f(&d["notes"]), f(&d["queries"]), f(&d["cross_harness_answerable"]), f(&d["same_harness_answerable"])));
        s.push_str("### Overall\n\nPrecision = gold notes among returned notes (answerable queries). Baselines have no abstention mechanism and no `sufficient` signal (always top-5).\n\n| system | R@1 | R@5 | MRR | notes/query | precision | scope-leak (queries) | forbidden hit | stale returned | stale unflagged | stale above gold | abstain (unanswerable) | abstain or `sufficient=false` (unanswerable) | false abstain (answerable) | rejected-note returned | ms/query |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        let names: Vec<String> = e1["systems"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        for n in &names {
            let o = &e1["systems"][n]["overall"];
            s.push_str(&format!(
                "| {n} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                f(&o["recall@1"]),
                f(&o["recall@5"]),
                f(&o["mrr"]),
                f(&o["notes_per_query"]),
                f(&o["precision_answerable"]),
                pctf(&o["scope_leak_rate"]),
                pctf(&o["forbidden_hit_rate"]),
                pctf(&o["stale_returned_rate"]),
                pctf(&o["stale_unflagged_rate"]),
                pctf(&o["stale_above_gold_rate"]),
                pctf(&o["abstain_rate"]),
                pctf(&o["abstain_or_insufficient_rate"]),
                pctf(&o["false_abstain_rate"]),
                pctf(&o["unsafe_note_returned_rate"]),
                f(&o["latency_ms_mean"])
            ));
        }
        s.push_str("\n`sufficient=false` on answerable queries (the 'I don't know' signal misfiring): ");
        for n in ["unimem-full", "unimem-passive"] {
            s.push_str(&format!("{n} {} ", pctf(&e1["systems"][n]["overall"]["insufficient_on_answerable_rate"])));
        }
        s.push('\n');
        s.push_str("\n### Recall@5 by category (MRR in parentheses; unanswerable → abstain rate)\n\n| system |");
        for c in CATS {
            s.push_str(&format!(" {c} |"));
        }
        s.push_str("\n|---|");
        for _ in CATS {
            s.push_str("---|");
        }
        s.push('\n');
        for n in &names {
            s.push_str(&format!("| {n} |"));
            for c in CATS {
                let o = &e1["systems"][n]["by_cat"][c];
                if *c == "unanswerable" {
                    s.push_str(&format!(" {} |", pctf(&o["abstain_rate"])));
                } else {
                    s.push_str(&format!(" {} ({}) |", f(&o["recall@5"]), f(&o["mrr"])));
                }
            }
            s.push('\n');
        }
        s.push_str("\n### Cross-harness: unified vs per-harness silo (answerable queries)\n\n| system | cross-harness R@5 | cross-harness MRR | same-harness R@5 | same-harness MRR |\n|---|---|---|---|---|\n");
        for n in &names {
            let x = &e1["systems"][n];
            s.push_str(&format!(
                "| {n} | {} | {} | {} | {} |\n",
                f(&x["cross_harness"]["recall@5"]),
                f(&x["cross_harness"]["mrr"]),
                f(&x["same_harness"]["recall@5"]),
                f(&x["same_harness"]["mrr"])
            ));
        }
        s.push_str("\n### Scope isolation detail\n\n| system | scope_isolation R@5 | forbidden hit (isolation queries) | leak rate (isolation queries) |\n|---|---|---|---|\n");
        for n in &names {
            let o = &e1["systems"][n]["by_cat"]["scope_isolation"];
            s.push_str(&format!(
                "| {n} | {} | {} | {} |\n",
                f(&o["recall@5"]),
                pctf(&o["forbidden_hit_rate"]),
                pctf(&o["scope_leak_rate"])
            ));
        }
        let w = &e1["write_screen"];
        let sm = &w["summary"];
        s.push_str("\n### Write path (unimem, unified store)\n\n| metric | value |\n|---|---|\n");
        for (k, v) in [
            ("rejection precision", &sm["reject_precision"]),
            ("rejection recall", &sm["reject_recall"]),
            ("duplicate precision", &sm["dup_precision"]),
            ("duplicate recall", &sm["dup_recall"]),
            ("duplicate recall (correct original)", &sm["dup_recall_correct_original"]),
            ("supersede precision (pairs)", &sm["supersede_precision"]),
            ("supersede recall (pairs)", &sm["supersede_recall"]),
            ("scope-assignment accuracy", &sm["scope_accuracy"]),
            ("kind accuracy", &sm["kind_accuracy"]),
            ("decoy false-reject rate", &sm["decoy_false_reject_rate"]),
            ("distractor false-supersede rate (pairs)", &sm["distractor_false_supersede_rate"]),
            ("distractor target wrongly superseded (by anything)", &sm["distractor_target_wrongly_superseded_rate"]),
        ] {
            s.push_str(&format!("| {k} | {} |\n", f(v)));
        }
        s.push_str(&format!("\nRejections per type: {}. False rejects: {}. Duplicates: {} gold, {} detected, {} false positives ({} of them were superseding updates swallowed as duplicates). Supersede pairs: {} gold, {} predicted, {} correct; lost updates (supersede notes not stored): {}.\n",
            w["rejection"]["per_type_recall"], w["rejection"]["fp"], f(&w["duplicates"]["gold"]), f(&w["duplicates"]["detected"]), f(&w["duplicates"]["false_positive"]), f(&w["duplicates"]["supersede_swallowed_as_duplicate"]),
            f(&w["supersede"]["gold_pairs"]), f(&w["supersede"]["predicted_pairs"]), f(&w["supersede"]["correct_pairs"]), w["supersede"]["lost_updates"].as_array().map(|a| a.len()).unwrap_or(0)));
        s.push_str(&format!(
            "\nDecoys rejected per type: {} (of {}). Distractor pairs: {}, false supersedes: {}.\n",
            w["decoys"]["per_type_rejected"],
            f(&w["decoys"]["n"]),
            f(&w["distractors"]["pairs"]),
            w["distractors"]["false_supersede_pairs"]
        ));
        s.push_str("\nScope assignment by group:\n\n| group | correct / n | accuracy |\n|---|---|---|\n");
        if let Some(o) = w["scope"]["by_group"].as_object() {
            for (k, v) in o {
                s.push_str(&format!("| {k} | {}/{} | {} |\n", f(&v["correct"]), f(&v["n"]), f(&v["acc"])));
            }
        }
        s.push_str(&format!(
            "\nScope confusion: {}. Graph edges created: {}. Write latency: p50 {} ms, p95 {} ms.\n",
            w["scope"]["confusion"],
            e1["edges"],
            f(&e1["write_latency"]["p50_ms"]),
            f(&e1["write_latency"]["p95_ms"])
        ));
    }
    if let Some(e2) = load("e2_gateway.json") {
        s.push_str(&format!("\n## E2 — gateway overhead & correctness\n\n{} requests per cell (+{} warm-up), request body ≈ {} bytes (Claude-Code-like: 12 tools, ~11 KB system prompt, 0–10 history messages). Mock upstream responds instantly, so numbers are pure gateway cost.\n\n", f(&e2["setup"]["requests_per_cell"]), f(&e2["setup"]["warmup"]), f(&e2["setup"]["request_body_bytes"]["stream"])));
        s.push_str("| path | TTFB p50 | TTFB p95 | total p50 | total p95 |\n|---|---|---|---|---|\n");
        if let Some(o) = e2["latency"].as_object() {
            for (k, v) in o {
                s.push_str(&format!(
                    "| {k} | {} | {} | {} | {} |\n",
                    f(&v["ttfb"]["p50_ms"]),
                    f(&v["ttfb"]["p95_ms"]),
                    f(&v["total"]["p50_ms"]),
                    f(&v["total"]["p95_ms"])
                ));
            }
        }
        s.push_str("\nAdded latency vs direct (ms; difference of percentiles, and paired per-request difference p50/p95):\n\n| via | +TTFB p50 | +TTFB p95 | +total p50 | +total p95 | paired ΔTTFB p50/p95 |\n|---|---|---|---|---|---|\n");
        if let Some(o) = e2["overhead"].as_object() {
            for (k, v) in o {
                s.push_str(&format!(
                    "| {k} | {} | {} | {} | {} | {} / {} |\n",
                    f(&v["added_ttfb_p50_ms"]),
                    f(&v["added_ttfb_p95_ms"]),
                    f(&v["added_total_p50_ms"]),
                    f(&v["added_total_p95_ms"]),
                    f(&v["paired_diff_ttfb"]["p50_ms"]),
                    f(&v["paired_diff_ttfb"]["p95_ms"])
                ));
            }
        }
        s.push_str("\n### Prompt-cache stability (30-turn conversation, streaming)\n\nPrefix fraction = byte-identical common prefix of consecutive upstream requests (tools → system → messages) divided by the earlier request's length; 1.0 means the whole previous request is a cacheable prefix.\n\n| mode | client req | upstream req | continuation rounds | mean prefix fraction | min | fully stable transitions | cacheable byte ratio | system stable | tools stable |\n|---|---|---|---|---|---|---|---|---|---|\n");
        if let Some(o) = e2["cache_stability"].as_object() {
            for (k, v) in o {
                s.push_str(&format!(
                    "| {k} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} |\n",
                    f(&v["client_requests"]),
                    f(&v["upstream_requests"]),
                    f(&v["continuation_rounds"]),
                    f(&v["prefix_fraction_mean"]),
                    f(&v["prefix_fraction_min"]),
                    f(&v["fully_prefix_stable_transitions"]),
                    f(&v["transitions"]),
                    f(&v["cacheable_byte_ratio"]),
                    f(&v["system_byte_stable"]),
                    f(&v["tools_byte_stable"])
                ));
            }
        }
        let c = &e2["correctness"];
        s.push_str(&format!("\n### Interception correctness\n\nScripted turns (recall-only, remember-only, mixed unimem+Bash, Bash-only, plain; streaming and JSON; each with a follow-up request that echoes the visible message): {} client requests → {} upstream requests ({} continuation rounds). Leak/framing assertions additionally cover the 37 client requests of the 30-turn conversation. Assertions: **{} / {} passed**.\n\n| assertion | passed |\n|---|---|\n", f(&c["client_requests"]), f(&c["upstream_requests"]), f(&c["continuation_rounds"]), f(&c["assertions"]["passed"]), f(&c["assertions"]["total"])));
        if let Some(o) = c["assertions"]["by_assertion"].as_object() {
            for (k, v) in o {
                s.push_str(&format!("| {k} | {} |\n", f(v)));
            }
        }
        let pr = &e2["anchor_collision_probe"];
        s.push_str(&format!("\nAnchor-collision probe (identical prompt + identical reply in two different sessions): **{} / {}** second sessions had the *first* session's hidden tool rounds replayed into their upstream request.\n", f(&pr["second_session_got_first_sessions_hidden_rounds"]), f(&pr["pairs"])));
        if let Some(a) = c["assertions"]["failures"].as_array().filter(|a| !a.is_empty()) {
            s.push_str("\nFailures (first 40):\n\n");
            for x in a {
                s.push_str(&format!("- {}\n", f(x)));
            }
        }
    }
    if let Some(e3) = load("e3_project.json") {
        s.push_str(&format!("\n## E3 — project identification\n\n**{} / {} layouts correct ({}).**\n\n| case | expected pid | got | ok |\n|---|---|---|---|\n", f(&e3["correct"]), f(&e3["cases"]), pctf(&e3["accuracy"])));
        for r in e3["rows"].as_array().cloned().unwrap_or_default() {
            s.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                f(&r["case"]),
                f(&r["expect"]).replace('|', "\\|"),
                f(&r["got"]),
                if r["ok"] == true { "✓" } else { "✗" }
            ));
        }
    }
    if let Some(e4) = load("e4_throughput.json") {
        s.push_str(&format!("\n## E4 — engine throughput\n\nOutcomes of all writes: {}.\n\n| stored notes | live (not superseded) | write attempts | write p50 | write p95 | recall p50 | recall p95 | passive recall p50 / p95 | notes/recall | DB size | bytes/note | RSS MB |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n", e4["outcomes"]));
        for t in e4["tiers"].as_array().cloned().unwrap_or_default() {
            s.push_str(&format!(
                "| {} | {} | {} | {} ms | {} ms | {} ms | {} ms | {} / {} ms | {} | {:.2} MB | {} | {} |\n",
                f(&t["stored_notes"]),
                f(&t["live_notes"]),
                f(&t["write_attempts"]),
                f(&t["write_latency_last_500"]["p50_ms"]),
                f(&t["write_latency_last_500"]["p95_ms"]),
                f(&t["recall_latency"]["p50_ms"]),
                f(&t["recall_latency"]["p95_ms"]),
                f(&t["recall_passive_latency"]["p50_ms"]),
                f(&t["recall_passive_latency"]["p95_ms"]),
                f(&t["notes_returned_per_recall"]),
                t["db_bytes_after_checkpoint"].as_f64().unwrap_or(0.0) / 1e6,
                f(&t["bytes_per_note"]),
                f(&t["rss_mb"])
            ));
        }
    }
    if let Some(e5) = load("e5_long_session.json") {
        s.push_str(&format!("\n## E5 — long session (budgets, de-dup, compaction, tool-error recall, flush)\n\n{}\n\n", f(&e5["setup"])));
        s.push_str("| variant | max unimem live tokens | max unimem share of window | max total context share | cumulative new injected tokens | note injections | duplicate injections | preamble after compaction | compaction request untouched | tool-error turns | recall hits | gold in context | benign tool turns w/ injection | corrections hinted | flush nudge at req | planted facts stored by the compaction (of those stated before it) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        if let Some(o) = e5["variants"].as_object() {
            for (k, v) in o {
                let x = &v["summary"];
                s.push_str(&format!(
                    "| {k} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {}/{} |\n",
                    f(&x["max_unimem_live_tokens"]),
                    pctf(&x["max_unimem_context_share"]),
                    pctf(&x["max_total_context_share"]),
                    f(&x["cumulative_new_injected_tokens"]),
                    f(&x["new_note_injections"]),
                    f(&x["duplicate_injections"]),
                    f(&x["preamble_reinjected_after_compaction"]),
                    f(&x["compaction"]["request_passed_through_untouched"]),
                    f(&x["tool_error_turns"]),
                    f(&x["tool_error_recall_hits"]),
                    f(&x["tool_error_gold_in_context"]),
                    f(&x["benign_tool_turn_injections"]),
                    f(&x["benign_tool_turns"]),
                    f(&x["corrections_hinted"]),
                    f(&x["flush_nudge_at_request"]),
                    f(&x["planted_captured_by_compaction"]),
                    f(&x["planted_before_compaction"])
                ));
            }
        }
        s.push_str("\nPer-request series (context tokens, unimem live tokens, new injected tokens, cumulative) are in `e5_long_session.json` → `variants.<name>.per_request`.\n");
    }
    if let Ok(obs) = std::fs::read_to_string(out.join("OBSERVATIONS.md")) {
        s.push('\n');
        s.push_str(&obs);
    }
    std::fs::write(out.join("SUMMARY.md"), s)?;
    Ok(())
}

// ============================================================================ main

#[tokio::main]
async fn main() -> Result<()> {
    let root = bench_root();
    let (mut what, mut data, mut out) = (String::from("all"), root.join("bench/data/devmem.json"), root.join("bench/results"));
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--data" => data = PathBuf::from(args.next().context("--data needs a path")?),
            "--out" => out = PathBuf::from(args.next().context("--out needs a dir")?),
            _ => what = a,
        }
    }
    std::fs::create_dir_all(&out)?;
    let save = |name: &str, v: &Value| -> Result<()> {
        std::fs::write(out.join(name), serde_json::to_string_pretty(v)?)?;
        Ok(())
    };
    let run = |x: &str| what == "all" || what == x;
    if what != "summary" {
        save("env.json", &env_info())?;
    }
    if run("e3") {
        save("e3_project.json", &e3(&root).await?)?;
    }
    if run("e1") {
        save("e1_memory.json", &e1(&data).await?)?;
    }
    if run("e2") {
        save("e2_gateway.json", &e2(&root).await?)?;
    }
    if run("e4") {
        save("e4_throughput.json", &e4(&root).await?)?;
    }
    if run("e5") {
        save("e5_long_session.json", &e5(&root).await?)?;
    }
    if !["all", "e1", "e2", "e3", "e4", "e5", "summary"].contains(&what.as_str()) {
        bail!("usage: unimem-bench [all|e1|e2|e3|e4|e5|summary] [--data <dataset.json>] [--out <dir>]");
    }
    summary(&out, &data)?;
    println!("wrote {}", out.join("SUMMARY.md").display());
    Ok(())
}
