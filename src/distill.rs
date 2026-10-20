//! Passive capture distiller. The gateway records every prompt turn into the episode log; once a session has
//! been idle for `distill.idle_secs`, this extracts the durable parts (user corrections, explicit preferences,
//! decisions with reasons, gotchas) and sends them through the standard write path.
//!
//! Backends: any OpenAI-compatible chat model (`distill.base_url` + `distill.model`), or — when distillation is
//! enabled but no model is set — a deterministic heuristic that keeps explicit preferences and "remember …" asks.

//!
//! v0.4:
//! * sessions are processed in **chunks** of `distill.chunk_chars` (oldest first) and only the episodes of chunks
//!   actually processed are marked distilled (v0.3 marked everything but read only the newest 24k chars);
//! * a model-backed pass runs a **mem0-style update phase**: for each candidate the model sees its nearest stored
//!   neighbours and answers ADD / UPDATE(id) / DELETE(id) / NOOP. Without a model (or when that call fails) every
//!   candidate goes through the standard write path (which de-duplicates and supersedes on its own);
//! * triggers besides idleness: a new session in the same project (session end), the pre-compaction flush, both via
//!   [`spawn_session`]; passes are skipped while the upstream rate limit is low ([`note_rate_limit`]).

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::config::resolve_key;
use crate::engine::{Ctx, Engine, WriteOutcome, WriteReq};
use crate::screen;
use crate::store::Episode;
use crate::util::{now_ts, truncate_chars};

const MAX_PER_CHUNK: usize = 12;
const NEIGHBOURS: usize = 5;

const SYS: &str = "You extract durable long-term memories from a coding-agent conversation transcript. \
Keep ONLY things that will still be true and useful weeks from now: corrections the user made to the agent, \
explicit user preferences and conventions, decisions together with their reasons, non-obvious gotchas/root causes, \
and the right commands or procedures. EXCLUDE transient status, plans, next steps, progress reports, anything obvious \
from the code, and any secrets or credentials. Each memory must be one self-contained sentence with concrete names. \
scope_hint is \"global\" when it is about the user personally or applies across all their projects, otherwise \"project\". \
kind is one of fact, decision, preference, procedure, gotcha. \
Reply with ONE JSON object: {\"memories\": [{\"content\": \"…\", \"why\": \"… or null\", \"scope_hint\": \"global|project\", \"kind\": \"…\"}]}. \
Return {\"memories\": []} when nothing qualifies.";

fn line_of(e: &Episode) -> String {
    format!("[{}] {}\n", e.role, truncate_chars(e.text.trim(), 4000))
}

/// Split episodes (oldest first) into consecutive chunks whose transcript fits `max` chars each.
pub fn chunks(eps: &[Episode], max: usize) -> Vec<&[Episode]> {
    let mut out = vec![];
    let (mut start, mut used) = (0usize, 0usize);
    for (i, e) in eps.iter().enumerate() {
        let n = line_of(e).chars().count();
        if i > start && used + n > max {
            out.push(&eps[start..i]);
            start = i;
            used = 0;
        }
        used += n;
    }
    if start < eps.len() {
        out.push(&eps[start..]);
    }
    out
}

/// Build a transcript from episodes, keeping the newest content within `max` chars.
pub fn transcript(eps: &[Episode], max: usize) -> String {
    let mut parts: Vec<String> = vec![];
    let mut used = 0usize;
    for e in eps.iter().rev() {
        let line = format!("[{}] {}\n", e.role, truncate_chars(e.text.trim(), 4000));
        let n = line.chars().count();
        if used + n > max {
            if parts.is_empty() {
                // A single huge message: keep its tail.
                let tail: String = line.chars().skip(n - max).collect();
                parts.push(tail);
            }
            break;
        }
        used += n;
        parts.push(line);
    }
    parts.reverse();
    parts.concat()
}

fn remember_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)(\bremember\b|\bkeep in mind\b|\bfrom now on\b|\bgoing forward\b|\bnote that\b|记住|记一下|记下|以后都|今后)")
            .unwrap()
    })
}

fn lead_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)^\s*(please\s+)?(remember( that)?|keep in mind( that)?|note that|请)?\s*(记住|记一下|记下)?[,:：，\s]*").unwrap()
    })
}

/// Deterministic fallback: user-role messages that state a preference or explicitly ask to be remembered.
pub fn extract_heuristic(eps: &[Episode]) -> Vec<WriteReq> {
    extract_heuristic_with(eps, &HashSet::new())
}

/// [`extract_heuristic`], also keeping user corrections among the `important` episode ids.
pub fn extract_heuristic_with(eps: &[Episode], important: &HashSet<i64>) -> Vec<WriteReq> {
    let mut out: Vec<WriteReq> = vec![];
    for e in eps.iter().filter(|e| e.role == "user") {
        // a correction (high importance) is kept even without preference wording
        let corrected = important.contains(&e.id) && screen::is_correction(&e.text);
        let t = crate::util::strip_harness_noise(&e.text);
        for raw in t.split(['\n', '。', '！', '!']).map(str::trim) {
            let n = raw.chars().count();
            if !(8..=600).contains(&n) || raw.ends_with('?') || raw.ends_with('？') {
                continue;
            }
            let explicit = remember_re().is_match(raw) || corrected && n >= 16;
            if !explicit && !screen::is_preference_like(raw) {
                continue;
            }
            if screen::has_secret(raw) {
                continue;
            }
            let content = lead_re().replace(raw, "").trim().to_string();
            let content = if content.chars().count() >= 8 { content } else { raw.to_string() };
            if out.iter().any(|w| w.content == content) {
                continue;
            }
            let pref = screen::is_preference_like(&content);
            out.push(WriteReq {
                content,
                why: None,
                scope: Some(if pref { "global".into() } else { "auto".into() }),
                kind: if pref { Some("preference".into()) } else { None },
                pin: false,
            });
        }
    }
    out.truncate(MAX_PER_CHUNK);
    out
}

fn parse_model_reply(text: &str) -> Result<Vec<WriteReq>> {
    let t = text.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let v: Value = serde_json::from_str(t).map_err(|e| anyhow!("distill json: {e}"))?;
    let arr = if v.is_array() { v } else { v["memories"].clone() };
    let arr = arr.as_array().cloned().unwrap_or_default();
    let mut out = vec![];
    for m in arr.into_iter().take(MAX_PER_CHUNK) {
        let Some(content) = m["content"].as_str().map(str::trim).filter(|c| !c.is_empty()) else { continue };
        let scope = match m["scope_hint"].as_str() {
            Some("global") => "global",
            Some("project") => "project",
            _ => "auto",
        };
        out.push(WriteReq {
            content: content.to_string(),
            why: m["why"].as_str().map(str::trim).filter(|w| !w.is_empty() && *w != "null").map(String::from),
            scope: Some(scope.into()),
            kind: m["kind"].as_str().filter(|k| screen::KINDS.iter().any(|(n, _)| n == k)).map(String::from),
            pin: false,
        });
    }
    Ok(out)
}

async fn chat_json(engine: &Engine, client: &reqwest::Client, system: &str, user: String) -> Result<String> {
    let d = &engine.cfg.distill;
    let base = d.base_url.trim_end_matches('/').trim_end_matches("/v1");
    let body = json!({
        "model": d.model, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
    });
    let mut rb = client.post(format!("{base}/v1/chat/completions")).json(&body);
    if let Some(k) = resolve_key(&d.api_key, &d.api_key_env) {
        rb = rb.bearer_auth(k);
    }
    let r = rb.send().await?;
    if !r.status().is_success() {
        let s = r.status();
        return Err(anyhow!("distill model {s}: {}", truncate_chars(&r.text().await.unwrap_or_default(), 300)));
    }
    let v: Value = r.json().await?;
    Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("{}").to_string())
}

async fn extract_model(engine: &Engine, client: &reqwest::Client, text: &str) -> Result<Vec<WriteReq>> {
    parse_model_reply(&chat_json(engine, client, SYS, format!("TRANSCRIPT:\n{text}")).await?)
}

const OPS_SYS: &str = "You maintain a long-term memory store. For each CANDIDATE fact you are shown the most similar EXISTING memories. \
Decide one operation per candidate: ADD (new information), UPDATE (it replaces an existing memory on the same subject whose value or decision changed; give that memory's id), \
DELETE (it says an existing memory is no longer true and has no replacement value; give that id), NOOP (already known, or not worth keeping). \
Reply with ONE JSON object: {\"ops\": [{\"candidate\": <index>, \"op\": \"ADD|UPDATE|DELETE|NOOP\", \"id\": \"<existing id or null>\"}]}.";

/// One mem0-style update decision.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Add,
    Update(String),
    Delete(String),
    Noop,
}

/// Parse the update-phase reply; unknown ids (not among the candidate's neighbours) fall back to ADD.
pub fn parse_ops(text: &str, n: usize, neighbours: &[Vec<String>]) -> Vec<Op> {
    let t = text.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let mut out = vec![Op::Add; n];
    let Ok(v) = serde_json::from_str::<Value>(t) else { return out };
    for o in v["ops"].as_array().cloned().unwrap_or_default() {
        let Some(i) = o["candidate"].as_u64().map(|x| x as usize).filter(|i| *i < n) else { continue };
        let id = o["id"].as_str().unwrap_or("").to_string();
        let known = neighbours.get(i).is_some_and(|ns| ns.contains(&id));
        out[i] = match o["op"].as_str().unwrap_or("").to_uppercase().as_str() {
            "NOOP" => Op::Noop,
            "UPDATE" if known => Op::Update(id),
            "DELETE" if known => Op::Delete(id),
            _ => Op::Add,
        };
    }
    out
}

/// Nearest stored memories of a candidate in its likely scopes (lexical + vector).
async fn neighbours(engine: &Engine, ctx: &Ctx, c: &WriteReq) -> Vec<crate::store::Memory> {
    let mut scopes = vec!["global".to_string()];
    if let Some(s) = ctx.project_scope() {
        scopes.push(s);
    }
    let text = format!("{} {}", c.content, c.why.clone().unwrap_or_default());
    let mut ids: Vec<String> = engine.store.lex_search(&text, &scopes, NEIGHBOURS).into_iter().map(|h| h.id).collect();
    if let Ok(e) = engine.embedder.embed_one(&text).await {
        for (id, _) in engine.store.vec_search(&e, &scopes, NEIGHBOURS, false) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids.truncate(NEIGHBOURS);
    engine.store.get_memories(&ids)
}

/// Apply candidates: with a model, through the ADD/UPDATE/DELETE/NOOP update phase; otherwise straight through the
/// write path. Returns the number of memories stored.
async fn apply(engine: &Engine, client: Option<&reqwest::Client>, ctx: &Ctx, cands: Vec<WriteReq>) -> usize {
    let mut ops = vec![Op::Add; cands.len()];
    if let (Some(client), false) = (client, cands.is_empty()) {
        let mut nb_ids = vec![];
        let mut listing = String::new();
        for (i, c) in cands.iter().enumerate() {
            let ns = neighbours(engine, ctx, c).await;
            listing.push_str(&format!("CANDIDATE {i}: {}\n", c.content));
            for m in &ns {
                listing.push_str(&format!("  EXISTING {}: {}\n", m.id, truncate_chars(&m.content, 400)));
            }
            nb_ids.push(ns.into_iter().map(|m| m.id).collect::<Vec<_>>());
        }
        match chat_json(engine, client, OPS_SYS, listing).await {
            Ok(t) => ops = parse_ops(&t, cands.len(), &nb_ids),
            Err(e) => tracing::warn!("distill update phase failed ({e}); falling back to the write path"),
        }
    }
    let mut n = 0;
    for (c, op) in cands.into_iter().zip(ops) {
        match op {
            Op::Noop => {}
            Op::Delete(id) => {
                if engine.forget(&id).unwrap_or(false) {
                    engine.store.trace("distill_delete", &json!({"id": id, "reason": c.content}));
                }
            }
            Op::Add | Op::Update(_) => {
                let target = if let Op::Update(id) = &op { Some(id.clone()) } else { None };
                match engine.write(ctx, c).await {
                    Ok(WriteOutcome::Stored { id, superseded, .. }) => {
                        n += 1;
                        if let Some(old) = target.filter(|o| !superseded.contains(o)) {
                            let _ = engine.store.supersede(&old, &id);
                            let _ = engine.store.add_edge(&crate::store::Edge {
                                src: id.clone(),
                                dst: old,
                                rel: "supersedes".into(),
                                weight: 0.9,
                                judged_by: "distill".into(),
                            });
                        }
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!("distill write failed: {e}"),
                }
            }
        }
    }
    n
}

fn configured(engine: &Engine) -> (bool, bool) {
    let d = &engine.cfg.distill;
    let use_model = !d.base_url.is_empty() && !d.model.is_empty();
    (use_model || d.enabled && d.model.is_empty(), use_model)
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).timeout(Duration::from_secs(120)).build()?)
}

/// Distill one session's pending episodes, chunk by chunk (oldest first). Each chunk's episodes are marked distilled
/// only after the chunk was processed; a failed model call leaves that chunk and every later one pending.
pub async fn distill_session(engine: &Engine, session: &str) -> Result<usize> {
    let (ok, use_model) = configured(engine);
    if !ok {
        return Ok(0);
    }
    let eps = engine.store.undistilled_episodes(session);
    if eps.is_empty() {
        return Ok(0);
    }
    let row = engine.store.session_row(session);
    let ctx =
        engine.ctx_for(row.as_ref().map(|r| r.harness.as_str()).unwrap_or("distill"), session, row.as_ref().and_then(|r| r.cwd.as_deref()));
    let cl = if use_model { Some(client()?) } else { None };
    let important = engine.store.important_episodes(session, 3);
    let (mut stored, mut done) = (0usize, 0usize);
    for chunk in chunks(&eps, engine.cfg.distill.chunk_chars.max(2000)) {
        let cands = match &cl {
            Some(c) => match extract_model(engine, c, &transcript(chunk, usize::MAX)).await {
                Ok(x) => x,
                Err(e) => {
                    tracing::warn!(session, "distill failed: {e}");
                    break;
                }
            },
            None => extract_heuristic_with(chunk, &important),
        };
        stored += apply(engine, cl.as_ref(), &ctx, cands).await;
        engine.store.mark_distilled_ids(&chunk.iter().map(|e| e.id).collect::<Vec<_>>())?;
        done += chunk.len();
    }
    engine.store.trace("distill", &json!({"session": session, "episodes": eps.len(), "processed": done, "stored": stored, "by": if use_model { "model" } else { "heuristic" }}));
    Ok(stored)
}

// ---------------------------------------------------------------- rate-limit gate

const RATE_KEY: &str = "ratelimit_remaining";

/// Record the remaining share of the upstream rate limit from a response's headers
/// (`anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{remaining,limit}`,
/// `x-ratelimit-{remaining,limit}-{requests,tokens}`). The smallest share wins.
pub fn note_rate_limit(engine: &Engine, h: &reqwest::header::HeaderMap) {
    if let Some(x) = rate_share(h) {
        let _ = engine.store.meta_set(RATE_KEY, &format!("{x} {}", now_ts()));
    }
}

/// The smallest remaining share of any rate limit announced in `h`.
pub fn rate_share(h: &reqwest::header::HeaderMap) -> Option<f64> {
    let num = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).and_then(|s| s.trim().parse::<f64>().ok());
    let mut share: Option<f64> = None;
    for kind in ["requests", "tokens", "input-tokens", "output-tokens"] {
        for (rem, lim) in [
            (format!("anthropic-ratelimit-{kind}-remaining"), format!("anthropic-ratelimit-{kind}-limit")),
            (format!("x-ratelimit-remaining-{kind}"), format!("x-ratelimit-limit-{kind}")),
        ] {
            if let (Some(r), Some(l)) = (num(&rem), num(&lim)) {
                if l > 0.0 {
                    let x = (r / l).clamp(0.0, 1.0);
                    share = Some(share.map_or(x, |s: f64| s.min(x)));
                }
            }
        }
    }
    share
}

/// The last observed remaining share of the rate limit (within the last 10 minutes).
pub fn rate_limit_remaining(engine: &Engine) -> Option<f64> {
    let v = engine.store.meta_get(RATE_KEY)?;
    let mut it = v.split(' ');
    let x: f64 = it.next()?.parse().ok()?;
    let ts: i64 = it.next()?.parse().ok()?;
    (now_ts() - ts < 600).then_some(x)
}

fn rate_ok(engine: &Engine) -> bool {
    rate_limit_remaining(engine).is_none_or(|x| x >= engine.cfg.distill.min_rate_limit_remaining)
}

static IN_FLIGHT: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Distill `session` in the background now (session end, pre-compaction flush). No-op when distillation is not
/// configured, the session is already being distilled, or the rate limit is low.
pub fn spawn_session(engine: Arc<Engine>, session: String, reason: &'static str) {
    if !configured(&engine).0 {
        return;
    }
    if !rate_ok(&engine) {
        engine
            .store
            .trace("distill_skipped", &json!({"session": session, "reason": "rate_limit", "remaining": rate_limit_remaining(&engine)}));
        return;
    }
    if !IN_FLIGHT.lock().get_or_insert_with(HashSet::new).insert(session.clone()) {
        return;
    }
    let Ok(rt) = tokio::runtime::Handle::try_current() else {
        IN_FLIGHT.lock().get_or_insert_with(HashSet::new).remove(&session);
        return;
    };
    rt.spawn(async move {
        engine.store.trace("distill_trigger", &json!({"session": session, "reason": reason}));
        if let Err(e) = distill_session(&engine, &session).await {
            tracing::warn!(session = %session, "background distill failed: {e}");
        }
        IN_FLIGHT.lock().get_or_insert_with(HashSet::new).remove(&session);
    });
}

/// One distillation pass over all idle sessions. Returns the number of memories stored.
pub async fn run_once(engine: &Engine) -> Result<usize> {
    run_with_idle(engine, engine.cfg.distill.idle_secs as i64).await
}

/// Like [`run_once`] with an explicit idle threshold (`0` = every session with pending episodes).
pub async fn run_with_idle(engine: &Engine, idle_secs: i64) -> Result<usize> {
    if !configured(engine).0 {
        return Ok(0);
    }
    if !rate_ok(engine) {
        tracing::info!("distill pass skipped: upstream rate limit below {}", engine.cfg.distill.min_rate_limit_remaining);
        return Ok(0);
    }
    let mut stored = 0usize;
    for s in engine.store.idle_sessions(idle_secs) {
        stored += distill_session(engine, &s.key).await?;
    }
    Ok(stored)
}

/// Background loop: every 60s when `distill.enabled`.
pub fn spawn(engine: Arc<Engine>) {
    if !engine.cfg.distill.enabled {
        return;
    }
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        tick.tick().await;
        loop {
            tick.tick().await;
            match run_once(&engine).await {
                Ok(0) => {}
                Ok(n) => tracing::info!("distilled {n} memories from idle sessions"),
                Err(e) => tracing::warn!("distill pass failed: {e}"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(id: i64, role: &str, text: &str) -> Episode {
        Episode { id, session: "s".into(), harness: "h".into(), pid: None, role: role.into(), text: text.into(), ts: 0 }
    }

    #[test]
    fn heuristic_keeps_preferences_and_explicit_asks() {
        let eps = vec![
            ep(1, "user", "Please fix the failing test in parser.rs"),
            ep(2, "assistant", "I prefer to use tabs here."),
            ep(3, "user", "I prefer tabs over spaces in all my projects"),
            ep(4, "user", "Remember that deploys go through ops/deploy.sh, never kubectl directly"),
            ep(5, "user", "记住：这个项目的测试要用 cargo nextest 跑"),
            ep(6, "user", "do you remember why we did that?"),
            ep(7, "user", "remember my api_key = sk-abcdefghijklmnopqrstuvwxyz123456"),
        ];
        let w = extract_heuristic(&eps);
        let cs: Vec<&str> = w.iter().map(|x| x.content.as_str()).collect();
        assert_eq!(w.len(), 3, "{cs:?}");
        assert_eq!(cs[0], "I prefer tabs over spaces in all my projects");
        assert_eq!(w[0].scope.as_deref(), Some("global"));
        assert_eq!(w[0].kind.as_deref(), Some("preference"));
        assert_eq!(cs[1], "deploys go through ops/deploy.sh, never kubectl directly");
        assert_eq!(w[1].scope.as_deref(), Some("auto"));
        assert_eq!(cs[2], "这个项目的测试要用 cargo nextest 跑");
    }

    #[test]
    fn transcript_keeps_newest() {
        let eps: Vec<Episode> = (0..100).map(|i| ep(i, "user", &format!("message number {i} {}", "x".repeat(50)))).collect();
        let t = transcript(&eps, 1000);
        assert!(t.chars().count() <= 1000);
        assert!(t.contains("message number 99"));
        assert!(!t.contains("message number 0 "));
    }

    #[test]
    fn parses_model_reply() {
        let r = parse_model_reply("```json\n{\"memories\":[{\"content\":\"Use pnpm, not npm\",\"why\":\"lockfile\",\"scope_hint\":\"project\",\"kind\":\"decision\"},{\"content\":\"\"}]}\n```").unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].scope.as_deref(), Some("project"));
        assert_eq!(r[0].kind.as_deref(), Some("decision"));
        assert_eq!(r[0].why.as_deref(), Some("lockfile"));
    }

    #[tokio::test]
    async fn run_once_unconfigured_is_noop() {
        let e = Engine::with_parts(
            crate::config::Config::default(),
            crate::store::Store::open_memory().unwrap(),
            Box::new(crate::embed::HashEmbedder { dim: 64 }),
            crate::decide::ChainDecider::new(vec![Box::new(crate::decide::HeuristicDecider)]),
        );
        e.store.touch_session("s", "h", None, None).unwrap();
        e.store.add_episode("s", "h", None, "user", "I prefer tabs over spaces always").unwrap();
        assert_eq!(run_with_idle(&e, -10).await.unwrap(), 0);
        assert_eq!(e.store.undistilled_episodes("s").len(), 1);
    }

    #[tokio::test]
    async fn heuristic_pass_writes_and_marks() {
        let mut cfg = crate::config::Config::default();
        cfg.distill.enabled = true;
        let e = Engine::with_parts(
            cfg,
            crate::store::Store::open_memory().unwrap(),
            Box::new(crate::embed::HashEmbedder { dim: 64 }),
            crate::decide::ChainDecider::new(vec![Box::new(crate::decide::HeuristicDecider)]),
        );
        e.store.touch_session("s", "h", None, None).unwrap();
        e.store.add_episode("s", "h", None, "user", "I prefer tabs over spaces in all my projects").unwrap();
        assert_eq!(run_with_idle(&e, -10).await.unwrap(), 1);
        assert!(e.store.undistilled_episodes("s").is_empty());
    }
}
