//! unimem-std-bench: retrieval evaluation of unimem and standard baselines on recognised long-term-memory
//! benchmarks — LongMemEval_S (Wu et al., ICLR 2025) and LoCoMo (Maharana et al., ACL 2024).
//!
//! ```text
//! unimem-std-bench run    --cache DIR --dataset lme|locomo --gran turn|session --systems a,b,c [--budget SECS] [--limit N]
//! unimem-std-bench report --cache DIR --out bench/std/results [--split all|dev|test]
//! unimem-std-bench evidence --cache DIR --dataset lme --gran turn|session --embedder hash|e5 --out FILE.jsonl (dev split)
//! unimem-std-bench split  --cache DIR                       (LongMemEval question id → dev|test)
//! unimem-std-bench dump   --cache DIR --dataset lme|locomo --gran turn|session --system S [--k 50] --out FILE.jsonl
//! unimem-std-bench verify-onnx --cache DIR --model DIR      (needs `--features onnx` and ORT_DYLIB_PATH)
//! ```
//!
//! `DIR/data` holds the datasets (`bench/std/download.sh`), `DIR/emb` the multilingual-e5-small embedding cache
//! (`bench/std/embed_e5.py`, or the Rust `onnx` embedder). `run` is resumable: per-question results are appended to
//! `DIR/runs/<dataset>-<gran>-<system>.jsonl` and already-done questions are skipped, so it can be called repeatedly
//! with a time budget. `report` aggregates every run file into `results/*.json` and `results/tables.md`.
//!
//! Protocol (LongMemEval `src/retrieval/run_retrieval.py` + `eval_utils.py`, reproduced exactly):
//! * one index per question over its own haystack; keys = values = user-side utterances only;
//!   turn granularity: one unit per user turn, id `<session>_<turn#>` (1-based in the session), gold = `has_answer`;
//!   session granularity: one unit per session (user turns joined by " "), gold = evidence session with ≥1
//!   `has_answer` user turn;
//! * abstention questions (`_abs`) and questions without any user-side evidence turn are skipped (419 of 500 remain);
//! * `recall_any@k`, `recall_all@k` (the paper's "Recall") and `ndcg_any@k` with the official DCG; turn-level runs
//!   also report session metrics through the official turn→session conversion.
//!
//! LoCoMo: one index per conversation, unit = dialog turn (`dia_id`, text `"<speaker>: <text>"` plus the image
//! caption if any), gold = the QA's `evidence` ids; recall = fraction of evidence ids retrieved (the LoCoMo paper's
//! "recall accuracy"), per category (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop; 5 adversarial separately).
//!
//! unimem systems write every unit straight into the engine's store (`Store::insert_memory`, scope `global`,
//! `created_at` = the session timestamp), bypassing the write screens and dedup, then call `Engine::recall` with
//! k = 10 and no project scope. `unimem-write-*` instead runs the real write path (`Engine::write`).

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use unimem::config::Config;
use unimem::decide::ChainDecider;
use unimem::embed::{Embedder, HashEmbedder};
use unimem::engine::{Ctx, Depth, Engine, RecallReq, WriteOutcome, WriteReq};
use unimem::store::{cosine, Memory, Source, Store};

const KS: [usize; 7] = [1, 3, 5, 10, 25, 30, 50];
const RECALL_K: usize = 10;
/// Depth of the `unimem-topk-*` systems (ranking mode): enough for every reported k.
const TOPK_K: usize = 50;

/// Development / test split of LongMemEval_S, fixed before any fitting: a question is in the development split when
/// the first byte of sha256(question_id) is even, else in the test split. Only the development split was used to fit
/// the relevance prior (`bench/std/fit_prior.py`); headline numbers are on the test split. LoCoMo was not used for
/// fitting at all, so all of it is test data.
pub fn split_of(dataset: &str, qid: &str) -> &'static str {
    if dataset != "lme" {
        return "test";
    }
    if Sha256::digest(qid.as_bytes())[0] % 2 == 0 {
        "dev"
    } else {
        "test"
    }
}
const E5_DIM: usize = 384;

// ============================================================================ data

#[derive(Deserialize)]
struct LmeTurn {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: Option<bool>,
}

#[derive(Deserialize)]
struct LmeEntry {
    question_id: String,
    question_type: String,
    question: String,
    #[allow(dead_code)]
    question_date: String,
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<LmeTurn>>,
}

#[derive(Clone)]
struct Unit {
    id: String,
    /// session id (for turn → session conversion)
    group: String,
    /// session id as LongMemEval's official turn→session conversion sees it: non-evidence turns of an evidence
    /// session carry the id `noans_…` (run_retrieval.py renames them), so they count as a different session
    ogroup: String,
    text: String,
    ts: i64,
}

struct Case {
    qid: String,
    cat: String,
    query: String,
    gold: HashSet<String>,
}

struct Corpus {
    units: Vec<Unit>,
    cases: Vec<Case>,
}

/// "2023/05/20 (Sat) 02:21" → unix seconds (UTC)
fn parse_lme_date(s: &str) -> i64 {
    let p: Vec<&str> = s.split(['/', ' ', ':']).filter(|x| !x.is_empty()).collect();
    // [2023, 05, 20, (Sat), 02, 21]
    let n = |i: usize| p.get(i).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0);
    ymd_hm(n(0), n(1), n(2), n(4), n(5))
}

/// "1:56 pm on 8 May, 2023" → unix seconds (UTC)
fn parse_locomo_date(s: &str) -> i64 {
    let s = s.to_lowercase().replace(',', "");
    let p: Vec<&str> = s.split_whitespace().collect();
    if p.len() < 6 {
        return 0;
    }
    let (h, mi) = p[0].split_once(':').map(|(a, b)| (a.parse::<i64>().unwrap_or(0), b.parse::<i64>().unwrap_or(0))).unwrap_or((0, 0));
    let h = match (p[1], h) {
        ("pm", h) if h < 12 => h + 12,
        ("am", 12) => 0,
        (_, h) => h,
    };
    const M: [&str; 12] =
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
    let mon = M.iter().position(|m| *m == p[4]).map(|i| i as i64 + 1).unwrap_or(1);
    ymd_hm(p[5].parse().unwrap_or(1970), mon, p[3].parse().unwrap_or(1), h, mi)
}

fn ymd_hm(y: i64, m: i64, d: i64, h: i64, mi: i64) -> i64 {
    let month = time::Month::try_from(m.clamp(1, 12) as u8).unwrap_or(time::Month::January);
    let date = time::Date::from_calendar_date(y as i32, month, d.clamp(1, 31) as u8)
        .or_else(|_| time::Date::from_calendar_date(y as i32, month, 28))
        .unwrap_or(time::Date::MIN);
    let t = time::Time::from_hms(h.clamp(0, 23) as u8, mi.clamp(0, 59) as u8, 0).unwrap_or(time::Time::MIDNIGHT);
    time::PrimitiveDateTime::new(date, t).assume_utc().unix_timestamp()
}

/// LongMemEval_S → one corpus per evaluated question (`all`: also abstention / no-user-evidence questions, for QA).
fn load_lme(cache: &Path, gran: &str, all: bool) -> Result<(Vec<Corpus>, usize)> {
    let path = cache.join("data/longmemeval_s_cleaned.json");
    let f = std::io::BufReader::new(std::fs::File::open(&path).with_context(|| format!("{} (run bench/std/download.sh)", path.display()))?);
    let data: Vec<LmeEntry> = serde_json::from_reader(f)?;
    let total = data.len();
    let mut out = vec![];
    for e in data {
        let has_user_target = e.haystack_sessions.iter().flatten().any(|t| t.role == "user" && t.has_answer == Some(true));
        if !all && (e.question_id.contains("_abs") || !has_user_target) {
            continue;
        }
        let mut units = vec![];
        let mut gold = HashSet::new();
        for ((sid, sess), date) in e.haystack_session_ids.iter().zip(&e.haystack_sessions).zip(&e.haystack_dates) {
            let ts = parse_lme_date(date);
            let is_ans = sid.contains("answer");
            if gran == "session" {
                let users: Vec<&str> = sess.iter().filter(|t| t.role == "user").map(|t| t.content.as_str()).collect();
                if is_ans && sess.iter().any(|t| t.role == "user" && t.has_answer == Some(true)) {
                    gold.insert(sid.clone());
                }
                units.push(Unit { id: sid.clone(), group: sid.clone(), ogroup: sid.clone(), text: users.join(" "), ts });
            } else {
                for (i, t) in sess.iter().enumerate() {
                    if t.role != "user" {
                        continue;
                    }
                    let id = format!("{sid}_{}", i + 1);
                    let evidence = is_ans && t.has_answer == Some(true);
                    if evidence {
                        gold.insert(id.clone());
                    }
                    let ogroup = if is_ans && !evidence { sid.replace("answer", "noans") } else { sid.clone() };
                    units.push(Unit { id, group: sid.clone(), ogroup, text: t.content.clone(), ts });
                }
            }
        }
        out.push(Corpus { units, cases: vec![Case { qid: e.question_id, cat: e.question_type, query: e.question, gold }] });
    }
    Ok((out, total))
}

pub fn locomo_turn_text(t: &Value) -> String {
    let mut s = format!("{}: {}", t["speaker"].as_str().unwrap_or(""), t["text"].as_str().unwrap_or(""));
    if let Some(c) = t["blip_caption"].as_str().filter(|c| !c.is_empty()) {
        s.push_str(&format!(" [shares {c}]"));
    }
    s
}

/// Normalise LoCoMo evidence ids ("D8:6; D9:17", "D:11:26", "D30:05") into valid dia_ids.
fn locomo_evidence(raw: &[Value], valid: &HashSet<String>) -> (HashSet<String>, usize) {
    let mut out = HashSet::new();
    let mut bad = 0;
    for e in raw {
        let s = e.as_str().unwrap_or("");
        for part in s.split(|c: char| c == ';' || c.is_whitespace()).filter(|p| !p.is_empty()) {
            let p = part.replace("D:", "D");
            let fixed = match p.trim_start_matches('D').split_once(':') {
                Some((a, b)) => format!("D{}:{}", a.trim_start_matches('0'), b.trim_start_matches('0')),
                None => p.clone(),
            };
            if valid.contains(&fixed) {
                out.insert(fixed);
            } else {
                bad += 1;
            }
        }
    }
    (out, bad)
}

fn load_locomo(cache: &Path, all: bool) -> Result<(Vec<Corpus>, Value)> {
    let path = cache.join("data/locomo10.json");
    let data: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?)?;
    let mut out = vec![];
    let (mut n_q, mut n_no_ev, mut n_bad) = (0, 0, 0);
    for conv in &data {
        let c = &conv["conversation"];
        let sample = conv["sample_id"].as_str().unwrap_or("conv").to_string();
        let mut sess_keys: Vec<(usize, String)> = c
            .as_object()
            .map(|o| {
                o.keys().filter_map(|k| k.strip_prefix("session_").and_then(|n| n.parse::<usize>().ok()).map(|n| (n, k.clone()))).collect()
            })
            .unwrap_or_default();
        sess_keys.sort();
        let mut units = vec![];
        for (n, k) in &sess_keys {
            let Some(turns) = c[k].as_array() else { continue };
            let ts = parse_locomo_date(c[format!("session_{n}_date_time")].as_str().unwrap_or(""));
            for t in turns {
                let id = t["dia_id"].as_str().unwrap_or("").to_string();
                units.push(Unit { id, group: format!("S{n}"), ogroup: format!("S{n}"), text: locomo_turn_text(t), ts });
            }
        }
        let valid: HashSet<String> = units.iter().map(|u| u.id.clone()).collect();
        let mut cases = vec![];
        for (i, q) in conv["qa"].as_array().into_iter().flatten().enumerate() {
            n_q += 1;
            let (gold, bad) = locomo_evidence(q["evidence"].as_array().map(|v| v.as_slice()).unwrap_or(&[]), &valid);
            n_bad += bad;
            if gold.is_empty() {
                n_no_ev += 1;
                if !all {
                    continue;
                }
            }
            cases.push(Case {
                qid: format!("{sample}#{i}"),
                cat: q["category"].as_i64().map(|c| c.to_string()).unwrap_or_default(),
                query: q["question"].as_str().unwrap_or("").to_string(),
                gold,
            });
        }
        out.push(Corpus { units, cases });
    }
    Ok((out, json!({"questions": n_q, "skipped_no_valid_evidence": n_no_ev, "unparseable_evidence_ids": n_bad})))
}

// ============================================================================ embedding cache

/// Text → row of an `embed_e5.py` cache (`<set>.texts.jsonl` + `<set>.e5s.f32`).
struct EmbCache {
    idx: HashMap<String, u32>,
    /// the cached texts in row order (exactly as embedded; used by `verify-onnx`)
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    texts: Vec<String>,
    data: Vec<f32>,
}

impl EmbCache {
    fn load(cache: &Path, set: &str) -> Result<Self> {
        let tp = cache.join(format!("emb/{set}.texts.jsonl"));
        let vp = cache.join(format!("emb/{set}.e5s.f32"));
        let f = std::io::BufReader::new(
            std::fs::File::open(&tp).with_context(|| format!("{} (run bench/std/embed_e5.py --set {set})", tp.display()))?,
        );
        let mut idx = HashMap::new();
        let mut texts = vec![];
        for (i, l) in f.lines().enumerate() {
            let s: String = serde_json::from_str(&l?)?;
            // the engine trims queries before embedding them: also index the trimmed form
            idx.entry(s.trim().to_string()).or_insert(i as u32);
            idx.insert(s.clone(), i as u32);
            texts.push(s);
        }
        let bytes = std::fs::read(&vp)?;
        let data: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let rows = texts.len();
        if data.len() != rows * E5_DIM {
            bail!("{set}: embedding cache incomplete ({} of {rows} rows); rerun embed_e5.py", data.len() / E5_DIM);
        }
        Ok(Self { idx, texts, data })
    }
    fn get(&self, text: &str) -> Option<&[f32]> {
        self.idx.get(text).map(|&i| &self.data[i as usize * E5_DIM..(i as usize + 1) * E5_DIM])
    }
}

/// An `Embedder` serving cached E5 vectors: queries from the query cache ("query: " prefix), anything else from the
/// passage cache ("passage: "). The engine only embeds the query at recall time (units are inserted with their
/// vectors), so the asymmetric prefixes are respected.
struct CachedE5 {
    q: Arc<EmbCache>,
    p: Arc<EmbCache>,
}

#[async_trait]
impl Embedder for CachedE5 {
    fn id(&self) -> String {
        "e5-small-cached".into()
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| {
                self.q
                    .get(t)
                    .or_else(|| self.p.get(t))
                    .or_else(|| self.q.get(t.trim()))
                    .or_else(|| self.p.get(t.trim()))
                    .map(|v| v.to_vec())
                    .ok_or_else(|| anyhow!("text not in e5 cache: {}", t.chars().take(60).collect::<String>()))
            })
            .collect()
    }
}

// ============================================================================ retrievers

/// `rank_bm25.BM25Okapi` exactly as LongMemEval's `flat-bm25` uses it: `str.split(" ")` tokens (case-sensitive,
/// punctuation attached), k1 = 1.5, b = 0.75, negative idf floored at 0.25 × mean idf.
struct Okapi {
    tf: Vec<HashMap<String, f64>>,
    len: Vec<f64>,
    avgdl: f64,
    idf: HashMap<String, f64>,
}

impl Okapi {
    fn new(docs: &[&str]) -> Self {
        let mut tf = vec![];
        let mut len = vec![];
        let mut df: HashMap<String, f64> = HashMap::new();
        for d in docs {
            let mut m: HashMap<String, f64> = HashMap::new();
            let toks: Vec<&str> = d.split(' ').collect();
            for t in &toks {
                *m.entry(t.to_string()).or_default() += 1.0;
            }
            for k in m.keys() {
                *df.entry(k.clone()).or_default() += 1.0;
            }
            len.push(toks.len() as f64);
            tf.push(m);
        }
        let n = docs.len() as f64;
        let avgdl = len.iter().sum::<f64>() / n.max(1.0);
        let mut idf: HashMap<String, f64> = HashMap::new();
        let mut sum = 0.0;
        let mut neg = vec![];
        for (w, f) in &df {
            let v = (n - f + 0.5).ln() - (f + 0.5).ln();
            sum += v;
            if v < 0.0 {
                neg.push(w.clone());
            }
            idf.insert(w.clone(), v);
        }
        let eps = 0.25 * sum / (idf.len().max(1) as f64);
        for w in neg {
            idf.insert(w, eps);
        }
        Self { tf, len, avgdl, idf }
    }
    fn scores(&self, q: &str) -> Vec<f64> {
        let (k1, b) = (1.5, 0.75);
        let mut s = vec![0.0; self.tf.len()];
        for t in q.split(' ') {
            let i = self.idf.get(t).copied().unwrap_or(0.0);
            for (d, m) in self.tf.iter().enumerate() {
                let f = m.get(t).copied().unwrap_or(0.0);
                s[d] += i * (f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * self.len[d] / self.avgdl)));
            }
        }
        s
    }
}

const LUCENE_STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
];

/// Lucene-style analyzer: lowercase, split on non-alphanumerics, English stop set, Snowball English stemmer.
fn analyze(stem: &rust_stemmers::Stemmer, s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !LUCENE_STOP.contains(w))
        .map(|w| stem.stem(w).to_string())
        .collect()
}

/// Standard BM25 (k1 = 1.2, b = 0.75, Lucene idf `ln(1 + (N − df + 0.5)/(df + 0.5))`) over the analyzer above.
struct Bm25 {
    post: HashMap<String, Vec<(u32, f32)>>,
    len: Vec<f64>,
    avgdl: f64,
    stem: rust_stemmers::Stemmer,
}

impl Bm25 {
    fn new(docs: &[&str]) -> Self {
        let stem = rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English);
        let mut post: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        let mut len = vec![];
        for (i, d) in docs.iter().enumerate() {
            let toks = analyze(&stem, d);
            let mut m: HashMap<String, f32> = HashMap::new();
            for t in &toks {
                *m.entry(t.clone()).or_default() += 1.0;
            }
            for (t, f) in m {
                post.entry(t).or_default().push((i as u32, f));
            }
            len.push(toks.len() as f64);
        }
        let avgdl = len.iter().sum::<f64>() / (len.len().max(1) as f64);
        Self { post, len, avgdl: avgdl.max(1e-9), stem }
    }
    fn scores(&self, q: &str) -> Vec<f64> {
        let (k1, b) = (1.2, 0.75);
        let n = self.len.len() as f64;
        let mut s = vec![0.0; self.len.len()];
        for t in analyze(&self.stem, q) {
            let Some(p) = self.post.get(&t) else { continue };
            let idf = (1.0 + (n - p.len() as f64 + 0.5) / (p.len() as f64 + 0.5)).ln();
            for (d, f) in p {
                let f = *f as f64;
                s[*d as usize] += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * self.len[*d as usize] / self.avgdl));
            }
        }
        s
    }
}

/// Descending by score; ties: higher index first (what `np.argsort(scores)[::-1]` does for a stable sort).
fn argsort_desc(s: &[f64]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..s.len()).collect();
    v.sort_by(|a, b| s[*a].partial_cmp(&s[*b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    v.reverse();
    v
}

fn rrf(lists: &[&[usize]], n: usize) -> Vec<usize> {
    let mut s = vec![0.0; n];
    for l in lists {
        for (r, &i) in l.iter().enumerate() {
            s[i] += 1.0 / (60.0 + r as f64 + 1.0);
        }
    }
    let mut v: Vec<usize> = (0..n).collect();
    v.sort_by(|a, b| s[*b].partial_cmp(&s[*a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    v
}

struct Shared {
    hash: HashEmbedder,
    e5p: Option<Arc<EmbCache>>,
    e5q: Option<Arc<EmbCache>>,
}

impl Shared {
    fn e5(&self) -> Result<(&EmbCache, &EmbCache)> {
        match (&self.e5p, &self.e5q) {
            (Some(p), Some(q)) => Ok((p, q)),
            _ => bail!("e5 caches not loaded"),
        }
    }
    fn e5_doc(&self, text: &str) -> Result<Vec<f32>> {
        let (p, _) = self.e5()?;
        p.get(text).map(|v| v.to_vec()).ok_or_else(|| anyhow!("passage not in e5 cache: {}", text.chars().take(60).collect::<String>()))
    }
    fn e5_query(&self, text: &str) -> Result<Vec<f32>> {
        let (_, q) = self.e5()?;
        q.get(text).map(|v| v.to_vec()).ok_or_else(|| anyhow!("query not in e5 cache: {text}"))
    }
}

enum Index {
    Okapi(Okapi),
    Bm25(Bm25),
    Dense {
        docs: Vec<Vec<f32>>,
        e5: bool,
    },
    Hybrid(Bm25, Vec<Vec<f32>>),
    /// `ids` maps memory id → unit indices (several when the write path merged duplicates)
    Engine {
        engine: Box<Engine>,
        ids: HashMap<String, Vec<usize>>,
        mode: EngineMode,
        stats: Value,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum EngineMode {
    /// `unimem-prior-*`: the engine's anchor ranking (`Engine::candidates`), no decider, no cut-off
    Rank,
    /// `unimem-topk-*`: `Engine::recall` with `Depth::TopK` (k = 50)
    TopK,
    /// `unimem-*`: `Engine::recall` with the adaptive cut-off (k = 10), as injected
    Adaptive,
}

fn mem_for(u: &Unit, id: String) -> Memory {
    Memory {
        id,
        scope: "global".into(),
        kind: "fact".into(),
        content: u.text.clone(),
        why: None,
        entities: unimem::screen::extract_entities(&u.text),
        source: Source { harness: "std-bench".into(), session: u.group.clone(), ..Default::default() },
        created_at: u.ts,
        updated_at: u.ts,
        superseded_by: None,
        salience: 0.5,
        hits: 0,
        last_hit_at: None,
        pinned: false,
    }
}

fn bench_cfg() -> Config {
    let mut c = Config::default();
    c.decider.chain = vec!["heuristic".into()];
    c
}

async fn build(sys: &str, corpus: &Corpus, sh: &Shared) -> Result<Index> {
    let docs: Vec<&str> = corpus.units.iter().map(|u| u.text.as_str()).collect();
    Ok(match sys {
        "bm25-okapi-ref" => Index::Okapi(Okapi::new(&docs)),
        "bm25" => Index::Bm25(Bm25::new(&docs)),
        "e5-small" => Index::Dense { docs: docs.iter().map(|d| sh.e5_doc(d)).collect::<Result<_>>()?, e5: true },
        "hash-dense" => Index::Dense { docs: docs.iter().map(|d| sh.hash.embed_sync(d)).collect(), e5: false },
        "hybrid-rrf" => Index::Hybrid(Bm25::new(&docs), docs.iter().map(|d| sh.e5_doc(d)).collect::<Result<_>>()?),
        s if s.starts_with("unimem-") => {
            let e5 = s.ends_with("-e5");
            let write = s.starts_with("unimem-write-");
            let mode = if s.starts_with("unimem-prior-") {
                EngineMode::Rank
            } else if s.starts_with("unimem-topk-") {
                EngineMode::TopK
            } else {
                EngineMode::Adaptive
            };
            let store = Store::open_memory()?;
            let embedder: Box<dyn Embedder> = if e5 {
                let (p, q) = (sh.e5p.clone().ok_or_else(|| anyhow!("no e5"))?, sh.e5q.clone().ok_or_else(|| anyhow!("no e5"))?);
                Box::new(CachedE5 { q, p })
            } else {
                Box::new(HashEmbedder { dim: 512 })
            };
            let cfg = bench_cfg();
            let engine = Engine::with_parts(cfg.clone(), store, embedder, ChainDecider::from_config(&cfg.decider));
            let mut ids: HashMap<String, Vec<usize>> = HashMap::new();
            let mut stats = json!({});
            if write {
                let (mut stored, mut dup, mut rej) = (0, 0, 0);
                let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
                for (i, u) in corpus.units.iter().enumerate() {
                    let ctx = Ctx { harness: "std-bench".into(), session: u.group.clone(), ..Default::default() };
                    let req = WriteReq { content: u.text.clone(), scope: Some("global".into()), ..Default::default() };
                    match engine.write(&ctx, req).await? {
                        WriteOutcome::Stored { id, .. } => {
                            stored += 1;
                            ids.entry(id).or_default().push(i);
                        }
                        WriteOutcome::Duplicate { of, .. } => {
                            dup += 1;
                            ids.entry(of).or_default().push(i);
                        }
                        WriteOutcome::Rejected { reason } => {
                            rej += 1;
                            *reasons.entry(reason.chars().take(40).collect()).or_default() += 1;
                        }
                    }
                }
                stats = json!({"stored": stored, "duplicate": dup, "rejected": rej, "reasons": reasons});
            } else {
                for (i, u) in corpus.units.iter().enumerate() {
                    let emb = if e5 { sh.e5_doc(&u.text)? } else { sh.hash.embed_sync(&u.text) };
                    let id = format!("u{i:06}");
                    engine.store.insert_memory(&mem_for(u, id.clone()), &emb)?;
                    ids.insert(id, vec![i]);
                }
            }
            Index::Engine { engine: Box::new(engine), ids, mode, stats }
        }
        _ => bail!("unknown system {sys}"),
    })
}

/// The engine's own candidate ranking (anchors = BM25 top-n ∪ cosine top-n, `relevance_prior`) without the
/// decider round-trip and the cut-off, so that Recall@k is measured on a full ranked list.
async fn engine_prior_rank(engine: &Engine, q: &str) -> Result<Vec<String>> {
    Ok(engine.candidates(q, &["global".to_string()], TOPK_K).await?.into_iter().map(|c| c.id).collect())
}

async fn search(ix: &Index, q: &str, sh: &Shared, n: usize) -> Result<(Vec<usize>, Value)> {
    Ok(match ix {
        Index::Okapi(o) => (argsort_desc(&o.scores(q)), Value::Null),
        Index::Bm25(b) => (argsort_desc(&b.scores(q)), Value::Null),
        Index::Dense { docs, e5 } => {
            let qv = if *e5 { sh.e5_query(q)? } else { sh.hash.embed_sync(q) };
            let s: Vec<f64> = docs.iter().map(|d| cosine(&qv, d) as f64).collect();
            (argsort_desc(&s), Value::Null)
        }
        Index::Hybrid(b, docs) => {
            let qv = sh.e5_query(q)?;
            let d: Vec<f64> = docs.iter().map(|d| cosine(&qv, d) as f64).collect();
            let (r1, r2) = (argsort_desc(&b.scores(q)), argsort_desc(&d));
            (rrf(&[&r1, &r2], n), Value::Null)
        }
        Index::Engine { engine, ids, mode, stats } => {
            let (mids, extra) = if *mode == EngineMode::Rank {
                (engine_prior_rank(engine, q).await?, Value::Null)
            } else {
                let (k, depth) = if *mode == EngineMode::TopK { (TOPK_K, Depth::TopK) } else { (RECALL_K, Depth::Adaptive) };
                let out = engine.recall(&Ctx::default(), RecallReq { query: q.to_string(), k, depth, ..Default::default() }).await?;
                let via: Vec<String> = out.notes.iter().map(|h| h.via.clone()).collect();
                (
                    out.notes.iter().map(|h| h.id.clone()).collect(),
                    json!({"returned": out.notes.len(), "mode": out.mode, "sufficient": out.sufficient, "via": via}),
                )
            };
            let mut seen = HashSet::new();
            let mut r = vec![];
            for m in mids {
                for &i in ids.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if seen.insert(i) {
                        r.push(i);
                    }
                }
            }
            let mut extra = extra;
            if !stats.is_null() && stats.as_object().is_some_and(|o| !o.is_empty()) {
                extra["write"] = stats.clone();
            }
            (r, extra)
        }
    })
}

// ============================================================================ metrics (official LongMemEval eval_utils.py)

fn dcg(rel: &[f64]) -> f64 {
    rel.iter().enumerate().map(|(i, r)| if i == 0 { *r } else { r / ((i + 1) as f64).log2() }).sum()
}

/// (recall_any, recall_all, ndcg_any) of `rank` (indices into `ids`) against `gold` doc ids at k.
fn eval_retrieval(rank: &[usize], gold: &HashSet<&str>, ids: &[&str], k: usize) -> (f64, f64, f64) {
    let top: HashSet<&str> = rank.iter().take(k).map(|&i| ids[i]).collect();
    let any = gold.iter().any(|g| top.contains(g)) as u8 as f64;
    let all = gold.iter().all(|g| top.contains(g)) as u8 as f64;
    let rels: Vec<f64> = ids.iter().map(|d| gold.contains(d) as u8 as f64).collect();
    let actual: Vec<f64> = rank.iter().take(k).map(|&i| rels[i]).collect();
    let mut ideal = rels.clone();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap());
    ideal.truncate(k);
    let idcg = dcg(&ideal);
    (any, all, if idcg == 0.0 { 0.0 } else { dcg(&actual) / idcg })
}

fn metrics(rank: &[usize], corpus: &Corpus, case: &Case, dataset: &str, gran: &str) -> Value {
    let ids: Vec<&str> = corpus.units.iter().map(|u| u.id.as_str()).collect();
    let gold: HashSet<&str> = case.gold.iter().map(|s| s.as_str()).collect();
    let mut unit = serde_json::Map::new();
    let mut sess = serde_json::Map::new();
    let mut strue = serde_json::Map::new();
    for k in KS {
        let (a, l, n) = eval_retrieval(rank, &gold, &ids, k);
        unit.insert(format!("recall_any@{k}"), json!(a));
        unit.insert(format!("recall_all@{k}"), json!(l));
        unit.insert(format!("ndcg_any@{k}"), json!(n));
        if dataset == "locomo" {
            let top: HashSet<&str> = rank.iter().take(k).map(|&i| ids[i]).collect();
            let frac = gold.iter().filter(|g| top.contains(*g)).count() as f64 / gold.len() as f64;
            unit.insert(format!("recall_frac@{k}"), json!(frac));
        }
        if gran == "turn" {
            // official evaluate_retrieval_turn2session (`session`): grow k until k distinct session ids are covered,
            // with the official `noans_` relabelling; `session_true` uses the real session ids instead.
            for real in [false, true] {
                let groups: Vec<&str> = corpus.units.iter().map(|u| if real { u.group.as_str() } else { u.ogroup.as_str() }).collect();
                let gold_s: HashSet<&str> =
                    corpus.units.iter().zip(&groups).filter(|(u, _)| gold.contains(u.id.as_str())).map(|(_, g)| *g).collect();
                let mut ek = k;
                let uniq = |ek: usize| rank.iter().take(ek).map(|&i| groups[i]).collect::<HashSet<_>>().len();
                while ek <= groups.len() && uniq(ek) < k {
                    ek += 1;
                }
                let (a, l, n) = eval_retrieval(rank, &gold_s, &groups, ek);
                let m = if real { &mut strue } else { &mut sess };
                m.insert(format!("recall_any@{k}"), json!(a));
                m.insert(format!("recall_all@{k}"), json!(l));
                m.insert(format!("ndcg_any@{k}"), json!(n));
            }
        }
    }
    let mut m = json!({});
    if gran == "turn" {
        m["turn"] = Value::Object(unit);
        m["session"] = Value::Object(sess);
        m["session_true"] = Value::Object(strue);
    } else {
        m["session"] = Value::Object(unit);
    }
    m
}

// ============================================================================ run / report

fn runs_dir(cache: &Path) -> PathBuf {
    cache.join(std::env::var("UNIMEM_STD_RUNS").unwrap_or("runs".into()))
}

fn run_path(cache: &Path, dataset: &str, gran: &str, sys: &str) -> PathBuf {
    runs_dir(cache).join(format!("{dataset}-{gran}-{sys}.jsonl"))
}

fn done_set(p: &Path) -> HashSet<String> {
    let Ok(f) = std::fs::File::open(p) else { return HashSet::new() };
    std::io::BufReader::new(f)
        .lines()
        .map_while(|l| l.ok())
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .filter_map(|v| v["qid"].as_str().map(String::from))
        .collect()
}

fn load(cache: &Path, dataset: &str, gran: &str, all: bool) -> Result<(Vec<Corpus>, Value)> {
    match dataset {
        "lme" => {
            let (c, total) = load_lme(cache, gran, all)?;
            Ok((c, json!({"total": total})))
        }
        "locomo" => {
            if gran != "turn" {
                bail!("LoCoMo is evaluated at turn (dia_id) granularity");
            }
            load_locomo(cache, all)
        }
        _ => bail!("dataset must be lme or locomo"),
    }
}

fn shared_for(cache: &Path, dataset: &str, gran: &str, systems: &[String]) -> Result<Shared> {
    let needs_e5 = systems.iter().any(|s| s.contains("e5") || s == "hybrid-rrf");
    let (e5p, e5q) = if needs_e5 {
        let (p, q) = match dataset {
            "lme" => (if gran == "turn" { "lme_turn" } else { "lme_session" }, "lme_query"),
            _ => ("locomo_turn", "locomo_query"),
        };
        (Some(Arc::new(EmbCache::load(cache, p)?)), Some(Arc::new(EmbCache::load(cache, q)?)))
    } else {
        (None, None)
    };
    Ok(Shared { hash: HashEmbedder { dim: 512 }, e5p, e5q })
}

/// Retrieval output for downstream QA (`bench/std/qa_eval.py`): every question incl. abstention / adversarial ones,
/// one JSON line per question with the ranked unit ids (top `k`) in rank order.
async fn cmd_dump(cache: &Path, dataset: &str, gran: &str, sys: &str, k: usize, out: &Path) -> Result<()> {
    let (corpora, _) = load(cache, dataset, gran, true)?;
    let sh = shared_for(cache, dataset, gran, &[sys.to_string()])?;
    let mut f = std::fs::File::create(out)?;
    let mut n = 0;
    for corpus in &corpora {
        let ix = build(sys, corpus, &sh).await?;
        for case in &corpus.cases {
            let (rank, _) = search(&ix, &case.query, &sh, corpus.units.len()).await?;
            let ids: Vec<&str> = rank.iter().take(k).map(|&i| corpus.units[i].id.as_str()).collect();
            writeln!(f, "{}", json!({"qid": case.qid, "question": case.query, "system": sys, "granularity": gran, "ranked": ids}))?;
            n += 1;
        }
    }
    eprintln!("wrote {n} retrieval results to {}", out.display());
    Ok(())
}

async fn cmd_run(cache: &Path, dataset: &str, gran: &str, systems: &[String], budget: f64, limit: usize) -> Result<()> {
    let t0 = Instant::now();
    let (corpora, info) = load(cache, dataset, gran, false)?;
    let sh = shared_for(cache, dataset, gran, systems)?;
    eprintln!(
        "loaded {dataset}/{gran}: {} corpora, {} cases ({info}) in {:.1}s",
        corpora.len(),
        corpora.iter().map(|c| c.cases.len()).sum::<usize>(),
        t0.elapsed().as_secs_f64()
    );
    std::fs::create_dir_all(runs_dir(cache))?;
    let mut n_cases = 0;
    'outer: for sys in systems {
        let path = run_path(cache, dataset, gran, sys);
        let done = done_set(&path);
        let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        let mut new = 0;
        for corpus in corpora.iter() {
            if n_cases >= limit {
                break;
            }
            let todo: Vec<&Case> = corpus.cases.iter().filter(|c| !done.contains(&c.qid)).collect();
            if todo.is_empty() {
                continue;
            }
            let tb = Instant::now();
            let ix = build(sys, corpus, &sh).await?;
            let build_ms = tb.elapsed().as_secs_f64() * 1000.0;
            for (ci, case) in todo.into_iter().enumerate() {
                let ts = Instant::now();
                let (rank, mut extra) = search(&ix, &case.query, &sh, corpus.units.len()).await?;
                if ci > 0 {
                    // write-path outcomes are per corpus: record them once
                    if let Some(o) = extra.as_object_mut() {
                        o.remove("write");
                    }
                }
                let search_ms = ts.elapsed().as_secs_f64() * 1000.0;
                let m = metrics(&rank, corpus, case, dataset, gran);
                let top: Vec<&str> = rank.iter().take(50).map(|&i| corpus.units[i].id.as_str()).collect();
                let rec = json!({"qid": case.qid, "split": split_of(dataset, &case.qid), "cat": case.cat, "n_units": corpus.units.len(), "n_gold": case.gold.len(),
                    "metrics": m, "top": top, "extra": extra, "build_ms": build_ms, "search_ms": search_ms});
                writeln!(out, "{rec}")?;
                new += 1;
                n_cases += 1;
            }
            if t0.elapsed().as_secs_f64() > budget {
                eprintln!("{sys}: +{new} (budget reached)");
                break 'outer;
            }
        }
        eprintln!("{sys}: +{new}, {} total", done.len() + new);
    }
    eprintln!("elapsed {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn cmd_report(cache: &Path, out: &Path, split: &str) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(runs_dir(cache))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    let mut all: BTreeMap<String, Value> = BTreeMap::new();
    for f in &files {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let (dataset, rest) = name.split_once('-').unwrap_or(("?", ""));
        let (gran, sys) = rest.split_once('-').unwrap_or(("?", ""));
        let recs: Vec<Value> = std::io::BufReader::new(std::fs::File::open(f)?)
            .lines()
            .map_while(|l| l.ok())
            .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
            .filter(|r| split == "all" || split_of(dataset, r["qid"].as_str().unwrap_or("")) == split)
            .collect();
        // group: overall + per category; metric: level/name
        let mut agg: BTreeMap<String, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
        let mut returned = vec![];
        let mut search_ms = vec![];
        let mut build_ms = vec![];
        let mut wstats: BTreeMap<String, f64> = BTreeMap::new();
        let mut graph_hits = 0usize;
        for r in &recs {
            let cat = r["cat"].as_str().unwrap_or("").to_string();
            let groups: Vec<String> = if dataset == "locomo" {
                if cat == "5" {
                    vec!["overall(1-5)".into(), format!("cat{cat}")]
                } else {
                    vec!["overall(1-4)".into(), "overall(1-5)".into(), format!("cat{cat}")]
                }
            } else {
                vec!["overall".into(), cat.clone()]
            };
            for (lvl, ms) in r["metrics"].as_object().into_iter().flatten() {
                for (k, v) in ms.as_object().into_iter().flatten() {
                    for g in &groups {
                        agg.entry(g.clone()).or_default().entry(format!("{lvl}/{k}")).or_default().push(v.as_f64().unwrap_or(0.0));
                    }
                }
            }
            if let Some(n) = r["extra"]["returned"].as_f64() {
                returned.push(n);
            }
            if let Some(v) = r["extra"]["via"].as_array() {
                graph_hits += v.iter().filter(|x| x.as_str() != Some("anchor")).count();
            }
            if let Some(w) = r["extra"]["write"].as_object() {
                for k in ["stored", "duplicate", "rejected"] {
                    *wstats.entry(k.into()).or_default() += w[k].as_f64().unwrap_or(0.0);
                }
            }
            search_ms.push(r["search_ms"].as_f64().unwrap_or(0.0));
            build_ms.push(r["build_ms"].as_f64().unwrap_or(0.0));
        }
        let mut res = serde_json::Map::new();
        for (g, ms) in &agg {
            let mut o = serde_json::Map::new();
            o.insert("n".into(), json!(ms.values().next().map(|v| v.len()).unwrap_or(0)));
            for (k, v) in ms {
                o.insert(k.clone(), json!((mean(v) * 10000.0).round() / 10000.0));
            }
            res.insert(g.clone(), Value::Object(o));
        }
        let mut entry = json!({"dataset": dataset, "granularity": gran, "system": sys, "n": recs.len(), "groups": res,
            "search_ms_mean": (mean(&search_ms) * 100.0).round() / 100.0, "build_ms_mean": (mean(&build_ms) * 100.0).round() / 100.0});
        if !returned.is_empty() {
            entry["notes_returned_mean"] = json!((mean(&returned) * 100.0).round() / 100.0);
            entry["graph_sourced_notes"] = json!(graph_hits);
        }
        if !wstats.is_empty() {
            entry["write_outcomes"] = json!(wstats);
        }
        all.insert(name, entry);
    }
    // per-dataset json files
    for ds in ["lme", "locomo"] {
        let v: BTreeMap<&String, &Value> = all.iter().filter(|(_, e)| e["dataset"] == ds).collect();
        if !v.is_empty() {
            let name = if ds == "lme" { "longmemeval_s_retrieval" } else { "locomo_retrieval" };
            let name = if split == "all" { format!("{name}.json") } else { format!("{name}.{split}.json") };
            std::fs::write(out.join(name), serde_json::to_string_pretty(&v)?)?;
        }
    }
    // markdown tables
    let mut md = String::new();
    let g =
        |e: &Value, grp: &str, key: &str| -> String { e["groups"][grp][key].as_f64().map(|x| format!("{x:.3}")).unwrap_or("–".into()) };
    let order = [
        "bm25-okapi-ref",
        "bm25",
        "hash-dense",
        "e5-small",
        "hybrid-rrf",
        "unimem-prior-hash",
        "unimem-prior-e5",
        "unimem-topk-hash",
        "unimem-topk-e5",
        "unimem-hash",
        "unimem-e5",
        "unimem-write-hash",
        "unimem-write-e5",
    ];
    let sorted = |ds: &str, gran: &str| -> Vec<&Value> {
        let mut v: Vec<&Value> = all.values().filter(|e| e["dataset"] == ds && e["granularity"] == gran).collect();
        v.sort_by_key(|e| order.iter().position(|o| e["system"] == *o).unwrap_or(99));
        v
    };
    for gran in ["session", "turn"] {
        let rows = sorted("lme", gran);
        if rows.is_empty() {
            continue;
        }
        md.push_str(&format!("\n### LongMemEval_S, {gran}-granularity index (overall, n = {})\n\n", rows[0]["n"]));
        let lvls: &[&str] = if gran == "turn" { &["turn", "session", "session_true"] } else { &["session"] };
        md.push_str("| system |");
        for l in lvls {
            md.push_str(&format!(" {l} R_all@5 | {l} NDCG@5 | {l} R_all@10 | {l} NDCG@10 | {l} R_any@5 |"));
        }
        md.push_str(" notes/query | search ms |\n|---|");
        for _ in lvls {
            md.push_str("---|---|---|---|---|");
        }
        md.push_str("---|---|\n");
        for e in &rows {
            md.push_str(&format!("| {} |", e["system"].as_str().unwrap_or("")));
            for l in lvls {
                for k in ["recall_all@5", "ndcg_any@5", "recall_all@10", "ndcg_any@10", "recall_any@5"] {
                    md.push_str(&format!(" {} |", g(e, "overall", &format!("{l}/{k}"))));
                }
            }
            md.push_str(&format!(
                " {} | {} |\n",
                e["notes_returned_mean"].as_f64().map(|x| format!("{x:.2}")).unwrap_or("all".into()),
                e["search_ms_mean"]
            ));
        }
        let lvl = if gran == "turn" { "turn" } else { "session" };
        md.push_str(&format!("\nPer question type, {lvl}-level R_all@10 (n per type in header):\n\n"));
        let types = [
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
        ];
        md.push_str("| system |");
        for t in types {
            md.push_str(&format!(" {t} ({}) |", rows[0]["groups"][t]["n"]));
        }
        md.push_str("\n|---|");
        for _ in types {
            md.push_str("---|");
        }
        md.push('\n');
        for e in &rows {
            md.push_str(&format!("| {} |", e["system"].as_str().unwrap_or("")));
            for t in types {
                md.push_str(&format!(" {} |", g(e, t, &format!("{lvl}/recall_all@10"))));
            }
            md.push('\n');
        }
    }
    let rows = sorted("locomo", "turn");
    if !rows.is_empty() {
        md.push_str(&format!("\n### LoCoMo (locomo10), dialog-turn units — recall accuracy (fraction of evidence retrieved), n = {} (cat 1-4) + {} (cat 5)\n\n", rows[0]["groups"]["overall(1-4)"]["n"], rows[0]["groups"]["cat5"]["n"]));
        md.push_str("| system | R@5 | R@10 | R@25 | R@50 | NDCG@10 | multi-hop (1) R@10 | temporal (2) R@10 | open-dom. (3) R@10 | single-hop (4) R@10 | adversarial (5) R@10 | all 1-5 R@5 / R@10 / R@25 / R@50 | notes/query |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for e in &rows {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} / {} / {} / {} | {} |\n",
                e["system"].as_str().unwrap_or(""),
                g(e, "overall(1-4)", "turn/recall_frac@5"),
                g(e, "overall(1-4)", "turn/recall_frac@10"),
                g(e, "overall(1-4)", "turn/recall_frac@25"),
                g(e, "overall(1-4)", "turn/recall_frac@50"),
                g(e, "overall(1-4)", "turn/ndcg_any@10"),
                g(e, "cat1", "turn/recall_frac@10"),
                g(e, "cat2", "turn/recall_frac@10"),
                g(e, "cat3", "turn/recall_frac@10"),
                g(e, "cat4", "turn/recall_frac@10"),
                g(e, "cat5", "turn/recall_frac@10"),
                g(e, "overall(1-5)", "turn/recall_frac@5"),
                g(e, "overall(1-5)", "turn/recall_frac@10"),
                g(e, "overall(1-5)", "turn/recall_frac@25"),
                g(e, "overall(1-5)", "turn/recall_frac@50"),
                e["notes_returned_mean"].as_f64().map(|x| format!("{x:.2}")).unwrap_or("all".into()),
            ));
        }
    }
    let md = format!("<!-- split: {split} (LongMemEval: dev = sha256(question_id)[0] even; LoCoMo is all test) -->\n{md}");
    std::fs::write(out.join(if split == "all" { "tables.md".to_string() } else { format!("tables.{split}.md") }), &md)?;
    println!("{md}");
    Ok(())
}

#[cfg(feature = "onnx")]
async fn cmd_verify_onnx(cache: &Path, model: &Path) -> Result<()> {
    use unimem::embed::onnx::OnnxEmbedder;
    let e = OnnxEmbedder::load(model, 4)?;
    let p = EmbCache::load(cache, "locomo_turn")?;
    let q = EmbCache::load(cache, "locomo_query")?;
    let sample: Vec<&String> = p.texts.iter().step_by(p.texts.len() / 200).collect();
    let t = Instant::now();
    let mut v = vec![];
    for c in sample.chunks(16) {
        v.extend(e.embed(&c.iter().map(|s| s.to_string()).collect::<Vec<_>>()).await?);
    }
    let dt = t.elapsed().as_secs_f64();
    let cos: Vec<f64> = sample.iter().zip(&v).map(|(s, x)| cosine(p.get(s).unwrap(), x) as f64).collect();
    for (s, c) in sample.iter().zip(&cos) {
        if *c < 0.9999 {
            eprintln!("low parity {c:.4}: {:?}", s.chars().take(80).collect::<String>());
        }
    }
    let qs: Vec<&String> = q.texts.iter().step_by(q.texts.len() / 50).collect();
    let mut qc = vec![];
    for s in qs {
        qc.push(cosine(q.get(s).unwrap(), &e.embed_query(s).await?) as f64);
    }
    let min = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
    println!(
        "{}",
        json!({"embedder": e.id(), "passages": sample.len(), "passage_cos_vs_python_min": min(&cos), "passage_cos_mean": mean(&cos),
            "queries": qc.len(), "query_cos_min": min(&qc), "passages_per_s": sample.len() as f64 / dt})
    );
    Ok(())
}

/// Candidate evidence of the engine's anchor stage, with gold labels, for the development split only (input of
/// `bench/std/fit_prior.py`). Test-split questions are never written.
async fn cmd_evidence(cache: &Path, dataset: &str, gran: &str, emb: &str, out: &Path, budget: f64) -> Result<()> {
    let (corpora, _) = load(cache, dataset, gran, false)?;
    let sys = format!("unimem-prior-{emb}");
    let sh = shared_for(cache, dataset, gran, std::slice::from_ref(&sys))?;
    // resumable: questions already in `out` are skipped; stops after `budget` seconds
    let done = done_set(out);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out)?;
    let mut n = 0;
    let t0 = Instant::now();
    for corpus in &corpora {
        if t0.elapsed().as_secs_f64() > budget {
            eprintln!("budget reached");
            break;
        }
        let cases: Vec<&Case> = corpus.cases.iter().filter(|c| split_of(dataset, &c.qid) == "dev" && !done.contains(&c.qid)).collect();
        if cases.is_empty() {
            continue;
        }
        let Index::Engine { engine, ids, .. } = build(&sys, corpus, &sh).await? else { bail!("not an engine system") };
        for case in cases {
            let cands = engine.candidates(&case.query, &["global".to_string()], 30).await?;
            let rows: Vec<Value> = cands
                .iter()
                .map(|c| {
                    let y = ids.get(&c.id).is_some_and(|v| v.iter().any(|&i| case.gold.contains(&corpus.units[i].id)));
                    json!({"y": y as u8, "ev": c.ev})
                })
                .collect();
            writeln!(f, "{}", json!({"qid": case.qid, "cat": case.cat, "n_gold": case.gold.len(), "cands": rows}))?;
            n += 1;
        }
    }
    eprintln!("wrote evidence for {n} more dev questions ({} total) to {}", n + done.len(), out.display());
    Ok(())
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let cache = PathBuf::from(arg(&args, "--cache").or_else(|| std::env::var("UNIMEM_BENCH_CACHE").ok()).unwrap_or("bench-cache".into()));
    match cmd {
        "run" => {
            let dataset = arg(&args, "--dataset").unwrap_or("lme".into());
            let gran = arg(&args, "--gran").unwrap_or("turn".into());
            let systems: Vec<String> =
                arg(&args, "--systems").unwrap_or("bm25-okapi-ref,bm25,e5-small,hybrid-rrf".into()).split(',').map(String::from).collect();
            let budget: f64 = arg(&args, "--budget").and_then(|b| b.parse().ok()).unwrap_or(f64::MAX);
            let limit: usize = arg(&args, "--limit").and_then(|b| b.parse().ok()).unwrap_or(usize::MAX);
            cmd_run(&cache, &dataset, &gran, &systems, budget, limit).await
        }
        "split" => {
            // list the split of every LongMemEval question id (for the Python analysis scripts)
            let (corpora, _) = load(&cache, "lme", "session", true)?;
            for c in &corpora {
                for case in &c.cases {
                    println!("{}\t{}", case.qid, split_of("lme", &case.qid));
                }
            }
            Ok(())
        }
        "dump" => {
            let dataset = arg(&args, "--dataset").unwrap_or("lme".into());
            let gran = arg(&args, "--gran").unwrap_or("turn".into());
            let sys = arg(&args, "--system").unwrap_or("unimem-prior-e5".into());
            let k: usize = arg(&args, "--k").and_then(|b| b.parse().ok()).unwrap_or(50);
            let out = PathBuf::from(arg(&args, "--out").unwrap_or(format!("{dataset}-{gran}-{sys}.retrieval.jsonl")));
            cmd_dump(&cache, &dataset, &gran, &sys, k, &out).await
        }
        "report" => cmd_report(
            &cache,
            &PathBuf::from(arg(&args, "--out").unwrap_or("bench/std/results".into())),
            &arg(&args, "--split").unwrap_or("all".into()),
        ),
        "evidence" => {
            let dataset = arg(&args, "--dataset").unwrap_or("lme".into());
            let gran = arg(&args, "--gran").unwrap_or("turn".into());
            let emb = arg(&args, "--embedder").unwrap_or("hash".into());
            let out = PathBuf::from(arg(&args, "--out").ok_or_else(|| anyhow!("--out FILE"))?);
            let budget: f64 = arg(&args, "--budget").and_then(|b| b.parse().ok()).unwrap_or(f64::MAX);
            cmd_evidence(&cache, &dataset, &gran, &emb, &out, budget).await
        }
        #[cfg(feature = "onnx")]
        "verify-onnx" => cmd_verify_onnx(&cache, &PathBuf::from(arg(&args, "--model").ok_or_else(|| anyhow!("--model DIR"))?)).await,
        _ => {
            eprintln!("usage: unimem-std-bench run|report|dump|evidence|split|verify-onnx --cache DIR [--dataset lme|locomo] [--gran turn|session] [--systems a,b] [--budget s] [--limit n] [--out dir] [--model dir]");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_match_official_definitions() {
        let ids = ["a", "b", "c", "d"];
        let gold: HashSet<&str> = ["b", "d"].into_iter().collect();
        // ranking b, a, d, c: both gold in top-3; DCG = 1 + 1/log2(3); IDCG = 1 + 1/log2(2)
        let (any, all, ndcg) = eval_retrieval(&[1, 0, 3, 2], &gold, &ids, 3);
        assert_eq!((any, all), (1.0, 1.0));
        assert!((ndcg - (1.0 + 1.0 / 3f64.log2()) / 2.0).abs() < 1e-12);
        let (any, all, _) = eval_retrieval(&[1, 0, 3, 2], &gold, &ids, 1);
        assert_eq!((any, all), (1.0, 0.0));
    }

    #[test]
    fn dates_and_evidence_parse() {
        assert_eq!(parse_lme_date("2023/05/20 (Sat) 02:21"), ymd_hm(2023, 5, 20, 2, 21));
        assert_eq!(parse_locomo_date("1:56 pm on 8 May, 2023"), ymd_hm(2023, 5, 8, 13, 56));
        let valid: HashSet<String> = ["D8:6", "D9:17", "D11:26", "D30:5"].iter().map(|s| s.to_string()).collect();
        let (ev, bad) = locomo_evidence(&[json!("D8:6; D9:17"), json!("D:11:26"), json!("D30:05"), json!("D")], &valid);
        assert_eq!(ev.len(), 4);
        assert_eq!(bad, 1);
    }

    #[test]
    fn bm25_prefers_matching_stemmed_doc() {
        let b = Bm25::new(&["I adopted two cats last year", "the weather is nice", "my dog runs fast"]);
        let r = argsort_desc(&b.scores("How many cats did I adopt?"));
        assert_eq!(r[0], 0);
        let o = Okapi::new(&["a b c", "a a d", "e f g"]);
        assert!(o.scores("a")[1] > o.scores("a")[0]);
    }
}
