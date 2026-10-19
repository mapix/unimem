//! System One: typed, probabilistic micro-decisions (Jev-Mem). Pluggable backends:
//!
//! * `JevDecider` — TypeSafe Jev `/v1/systemone` (or any server speaking that schema, e.g. a local jevk5)
//! * `LlmDecider` — any OpenAI-compatible chat model, prompted to answer the same typed questions as JSON
//! * `HeuristicDecider` — offline; returns the code-computed prior every question carries
//!
//! Every question carries a `prior` computed by deterministic code (similarity, regexes, keywords). That keeps
//! the system fully functional with no model at all, and lets a `ChainDecider` degrade gracefully.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::config::{resolve_key, DeciderCfg};

#[derive(Debug, Clone)]
pub enum Kind {
    /// Yes/no; answer is a probability of "yes".
    Noul,
    /// Pick one of the named options (name, description).
    Choice(Vec<(String, String)>),
}

#[derive(Debug, Clone)]
pub struct Question {
    pub instructions: String,
    pub kind: Kind,
    pub prior: Answer,
}

impl Question {
    pub fn noul(instructions: impl Into<String>, prior: f64) -> Self {
        Self { instructions: instructions.into(), kind: Kind::Noul, prior: Answer::p(prior) }
    }
    pub fn choice(instructions: impl Into<String>, options: Vec<(String, String)>, prior: &str) -> Self {
        let prior = Answer::choice(prior, options.iter().map(|o| o.0.clone()).collect());
        Self { instructions: instructions.into(), kind: Kind::Choice(options), prior }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Answer {
    pub p: f64,
    pub choice: Option<String>,
    pub probs: BTreeMap<String, f64>,
    pub by: &'static str,
}

impl Answer {
    pub fn p(p: f64) -> Self {
        Self { p: p.clamp(0.0, 1.0), by: "prior", ..Default::default() }
    }
    fn choice(c: &str, all: Vec<String>) -> Self {
        let mut probs = BTreeMap::new();
        let n = all.len().max(1) as f64;
        for o in all {
            probs.insert(o.clone(), if o == c { 0.6 } else { 0.4 / (n - 1.0).max(1.0) });
        }
        Self { p: 0.6, choice: Some(c.to_string()), probs, by: "prior" }
    }
    pub fn picked(&self) -> &str {
        self.choice.as_deref().unwrap_or("")
    }
}

pub type Questions = Vec<(String, Question)>;
pub type Answers = BTreeMap<String, Answer>;

#[async_trait]
pub trait Decider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn ask(&self, state: &Value, questions: &Questions) -> Result<Answers>;
}

// ---------------------------------------------------------------- heuristic

pub struct HeuristicDecider;

#[async_trait]
impl Decider for HeuristicDecider {
    fn name(&self) -> &'static str {
        "heuristic"
    }
    async fn ask(&self, _state: &Value, questions: &Questions) -> Result<Answers> {
        Ok(questions.iter().map(|(k, q)| (k.clone(), Answer { by: "heuristic", ..q.prior.clone() })).collect())
    }
}

// ---------------------------------------------------------------- Jev

pub struct JevDecider {
    client: reqwest::Client,
    base_url: String,
    key: Option<String>,
    model: Option<String>,
    pub calls: AtomicU64,
    pub input_tokens: AtomicU64,
}

impl JevDecider {
    pub fn new(base_url: &str, key: Option<String>, model: Option<String>, timeout_ms: u64) -> Self {
        Self {
            client: reqwest::Client::builder().timeout(Duration::from_millis(timeout_ms)).build().unwrap(),
            base_url: base_url.trim_end_matches('/').to_string(),
            key,
            model,
            calls: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
        }
    }

    fn question_json(q: &Question) -> Value {
        match &q.kind {
            Kind::Noul => json!({"type": "noul", "instructions": q.instructions}),
            Kind::Choice(opts) => {
                let mut c = Map::new();
                for (n, d) in opts {
                    c.insert(n.clone(), if d.is_empty() { Value::Null } else { Value::String(d.clone()) });
                }
                json!({"type": "choice", "instructions": q.instructions, "criteria": c})
            }
        }
    }

    async fn ask_chunk(&self, state: &Value, qs: &[(String, Question)]) -> Result<Answers> {
        let mut questions = Map::new();
        for (k, q) in qs {
            questions.insert(k.clone(), Self::question_json(q));
        }
        let mut body = json!({"state": state, "questions": questions});
        if let Some(m) = &self.model {
            body["model"] = json!(m);
        }
        let mut rb = self.client.post(format!("{}/v1/systemone", self.base_url)).json(&body);
        if let Some(k) = &self.key {
            rb = rb.bearer_auth(k);
        }
        let r = rb.send().await?;
        let status = r.status();
        if !status.is_success() {
            return Err(anyhow!("jev {status}: {}", crate::util::truncate_chars(&r.text().await.unwrap_or_default(), 200)));
        }
        let v: Value = r.json().await?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        if let Some(t) = v["usage"]["input_tokens"].as_u64() {
            self.input_tokens.fetch_add(t, Ordering::Relaxed);
        }
        let mut out = Answers::new();
        for (k, q) in qs {
            let a = &v["answers"][k];
            let ans = match &q.kind {
                Kind::Noul => Answer { p: a["noul"].as_f64().ok_or_else(|| anyhow!("jev: missing {k}"))?, by: "jev", ..Default::default() },
                Kind::Choice(_) => {
                    let choice = a["choice"].as_str().ok_or_else(|| anyhow!("jev: missing {k}"))?.to_string();
                    let probs: BTreeMap<String, f64> = a["probabilities"]
                        .as_object()
                        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect())
                        .unwrap_or_default();
                    Answer { p: a["confidence"].as_f64().unwrap_or(1.0), choice: Some(choice), probs, by: "jev" }
                }
            };
            out.insert(k.clone(), ans);
        }
        Ok(out)
    }
}

#[async_trait]
impl Decider for JevDecider {
    fn name(&self) -> &'static str {
        "jev"
    }
    async fn ask(&self, state: &Value, questions: &Questions) -> Result<Answers> {
        // Batched requests; chunk to stay under context limits of small local models.
        let mut out = Answers::new();
        let futs = questions.chunks(24).map(|c| self.ask_chunk(state, c));
        for r in futures::future::join_all(futs).await {
            out.extend(r?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- LLM (OpenAI-compatible)

pub struct LlmDecider {
    client: reqwest::Client,
    base_url: String,
    key: Option<String>,
    model: String,
}

impl LlmDecider {
    pub fn new(base_url: &str, key: Option<String>, model: &str, timeout_ms: u64) -> Self {
        Self {
            client: reqwest::Client::builder().timeout(Duration::from_millis(timeout_ms)).build().unwrap(),
            base_url: base_url.trim_end_matches('/').trim_end_matches("/v1").to_string(),
            key,
            model: model.to_string(),
        }
    }
}

const LLM_SYS: &str = "You are a fast, calibrated classifier. You receive a STATE (JSON) and a set of named QUESTIONS. \
Answer every question about the STATE only. For a yes/no question return {\"p\": <probability of yes, 0..1>}. \
For a choice question return {\"choice\": \"<one option name exactly>\", \"confidence\": <0..1>}. \
Reply with ONE JSON object mapping each question name to its answer, nothing else.";

#[async_trait]
impl Decider for LlmDecider {
    fn name(&self) -> &'static str {
        "llm"
    }
    async fn ask(&self, state: &Value, questions: &Questions) -> Result<Answers> {
        let mut qd = Map::new();
        for (k, q) in questions {
            qd.insert(
                k.clone(),
                match &q.kind {
                    Kind::Noul => json!({"type": "yes/no", "question": q.instructions}),
                    Kind::Choice(o) => json!({"type": "choice", "question": q.instructions,
                        "options": o.iter().map(|(n, d)| json!({"name": n, "when": d})).collect::<Vec<_>>()}),
                },
            );
        }
        let user = format!("STATE:\n{}\n\nQUESTIONS:\n{}", state, Value::Object(qd));
        let body = json!({
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": LLM_SYS}, {"role": "user", "content": user}]
        });
        let mut rb = self.client.post(format!("{}/v1/chat/completions", self.base_url)).json(&body);
        if let Some(k) = &self.key {
            rb = rb.bearer_auth(k);
        }
        let r = rb.send().await?;
        if !r.status().is_success() {
            return Err(anyhow!("llm decider {}", r.status()));
        }
        let v: Value = r.json().await?;
        let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("{}");
        let text = text.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```");
        let a: Value = serde_json::from_str(text).map_err(|e| anyhow!("llm decider json: {e}"))?;
        let mut out = Answers::new();
        for (k, q) in questions {
            let x = &a[k];
            let ans = match &q.kind {
                Kind::Noul => Answer { p: x["p"].as_f64().ok_or_else(|| anyhow!("llm: missing {k}"))?.clamp(0.0, 1.0), by: "llm", ..Default::default() },
                Kind::Choice(opts) => {
                    let c = x["choice"].as_str().unwrap_or("");
                    let c = opts.iter().find(|o| o.0 == c).map(|o| o.0.clone()).ok_or_else(|| anyhow!("llm: bad choice {k}"))?;
                    Answer { p: x["confidence"].as_f64().unwrap_or(0.7), choice: Some(c), probs: BTreeMap::new(), by: "llm" }
                }
            };
            out.insert(k.clone(), ans);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- chain

pub struct ChainDecider {
    pub backends: Vec<Box<dyn Decider>>,
    /// Backends that failed recently are skipped until this instant (unix secs).
    cooldown: parking_lot::Mutex<BTreeMap<&'static str, i64>>,
}

impl ChainDecider {
    pub fn new(backends: Vec<Box<dyn Decider>>) -> Self {
        Self { backends, cooldown: parking_lot::Mutex::new(BTreeMap::new()) }
    }

    pub fn from_config(cfg: &DeciderCfg) -> Self {
        let mut b: Vec<Box<dyn Decider>> = vec![];
        for name in &cfg.chain {
            match name.as_str() {
                "jev" => {
                    let key = resolve_key(&cfg.jev.api_key, &cfg.jev.api_key_env);
                    let custom = cfg.jev.base_url != "https://api.typesafe.ai";
                    if key.is_some() || custom {
                        b.push(Box::new(JevDecider::new(&cfg.jev.base_url, key, cfg.jev.model.clone(), cfg.jev.timeout_ms)));
                    }
                }
                "llm" if !cfg.llm.base_url.is_empty() && !cfg.llm.model.is_empty() => {
                    b.push(Box::new(LlmDecider::new(
                        &cfg.llm.base_url,
                        resolve_key(&cfg.llm.api_key, &cfg.llm.api_key_env),
                        &cfg.llm.model,
                        cfg.llm.timeout_ms,
                    )));
                }
                "heuristic" => b.push(Box::new(HeuristicDecider)),
                _ => {}
            }
        }
        if !b.iter().any(|d| d.name() == "heuristic") {
            b.push(Box::new(HeuristicDecider));
        }
        Self::new(b)
    }

    pub fn describe(&self) -> String {
        self.backends.iter().map(|b| b.name()).collect::<Vec<_>>().join(" → ")
    }
}

#[async_trait]
impl Decider for ChainDecider {
    fn name(&self) -> &'static str {
        "chain"
    }
    async fn ask(&self, state: &Value, questions: &Questions) -> Result<Answers> {
        if questions.is_empty() {
            return Ok(Answers::new());
        }
        let now = crate::util::now_ts();
        let mut last_err = None;
        for b in &self.backends {
            if self.cooldown.lock().get(b.name()).is_some_and(|t| *t > now) {
                continue;
            }
            match b.ask(state, questions).await {
                Ok(a) if questions.iter().all(|(k, _)| a.contains_key(k)) => return Ok(a),
                Ok(_) => last_err = Some(anyhow!("{}: incomplete answers", b.name())),
                Err(e) => {
                    tracing::warn!(backend = b.name(), "decider unavailable: {e}");
                    self.cooldown.lock().insert(b.name(), now + 30);
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("no decider")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn chain_falls_back_to_heuristic() {
        let chain = ChainDecider::new(vec![
            Box::new(JevDecider::new("http://127.0.0.1:1", None, None, 300)),
            Box::new(HeuristicDecider),
        ]);
        let qs: Questions = vec![("x".into(), Question::noul("is it?", 0.8))];
        let a = chain.ask(&json!("s"), &qs).await.unwrap();
        assert_eq!(a["x"].by, "heuristic");
        assert!((a["x"].p - 0.8).abs() < 1e-9);
    }
}
