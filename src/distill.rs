//! Passive capture distiller. The gateway records every prompt turn into the episode log; once a session has
//! been idle for `distill.idle_secs`, this extracts the durable parts (user corrections, explicit preferences,
//! decisions with reasons, gotchas) and sends them through the standard write path.
//!
//! Backends: any OpenAI-compatible chat model (`distill.base_url` + `distill.model`), or — when distillation is
//! enabled but no model is set — a deterministic heuristic that keeps explicit preferences and "remember …" asks.

use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::config::resolve_key;
use crate::engine::{Engine, WriteOutcome, WriteReq};
use crate::screen;
use crate::store::Episode;
use crate::util::truncate_chars;

const MAX_TRANSCRIPT: usize = 24_000;
const MAX_PER_SESSION: usize = 12;

const SYS: &str = "You extract durable long-term memories from a coding-agent conversation transcript. \
Keep ONLY things that will still be true and useful weeks from now: corrections the user made to the agent, \
explicit user preferences and conventions, decisions together with their reasons, non-obvious gotchas/root causes, \
and the right commands or procedures. EXCLUDE transient status, plans, next steps, progress reports, anything obvious \
from the code, and any secrets or credentials. Each memory must be one self-contained sentence with concrete names. \
scope_hint is \"global\" when it is about the user personally or applies across all their projects, otherwise \"project\". \
kind is one of fact, decision, preference, procedure, gotcha. \
Reply with ONE JSON object: {\"memories\": [{\"content\": \"…\", \"why\": \"… or null\", \"scope_hint\": \"global|project\", \"kind\": \"…\"}]}. \
Return {\"memories\": []} when nothing qualifies.";

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
        Regex::new(r"(?i)(\bremember\b|\bkeep in mind\b|\bfrom now on\b|\bgoing forward\b|\bnote that\b|记住|记一下|记下|以后都|今后)").unwrap()
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
    let mut out: Vec<WriteReq> = vec![];
    for e in eps.iter().filter(|e| e.role == "user") {
        let t = crate::util::strip_harness_noise(&e.text);
        for raw in t.split(['\n', '。', '！', '!']).map(str::trim) {
            let n = raw.chars().count();
            if !(8..=600).contains(&n) || raw.ends_with('?') || raw.ends_with('？') {
                continue;
            }
            let explicit = remember_re().is_match(raw);
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
    out.truncate(MAX_PER_SESSION);
    out
}

fn parse_model_reply(text: &str) -> Result<Vec<WriteReq>> {
    let t = text.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let v: Value = serde_json::from_str(t).map_err(|e| anyhow!("distill json: {e}"))?;
    let arr = if v.is_array() { v } else { v["memories"].clone() };
    let arr = arr.as_array().cloned().unwrap_or_default();
    let mut out = vec![];
    for m in arr.into_iter().take(MAX_PER_SESSION) {
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

async fn extract_model(engine: &Engine, client: &reqwest::Client, text: &str) -> Result<Vec<WriteReq>> {
    let d = &engine.cfg.distill;
    let base = d.base_url.trim_end_matches('/').trim_end_matches("/v1");
    let body = json!({
        "model": d.model, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": format!("TRANSCRIPT:\n{text}")}]
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
    parse_model_reply(v["choices"][0]["message"]["content"].as_str().unwrap_or("{}"))
}

/// One distillation pass over all idle sessions. Returns the number of memories stored.
pub async fn run_once(engine: &Engine) -> Result<usize> {
    run_with_idle(engine, engine.cfg.distill.idle_secs as i64).await
}

/// Like [`run_once`] with an explicit idle threshold (`0` = every session with pending episodes).
pub async fn run_with_idle(engine: &Engine, idle_secs: i64) -> Result<usize> {
    let d = &engine.cfg.distill;
    let use_model = !d.base_url.is_empty() && !d.model.is_empty();
    if !use_model && !(d.enabled && d.model.is_empty()) {
        return Ok(0);
    }
    let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).timeout(Duration::from_secs(120)).build()?;
    let mut stored = 0usize;
    for s in engine.store.idle_sessions(idle_secs) {
        let eps = engine.store.undistilled_episodes(&s.key);
        let Some(upto) = eps.last().map(|e| e.id) else { continue };
        let cands = if use_model {
            match extract_model(engine, &client, &transcript(&eps, MAX_TRANSCRIPT)).await {
                Ok(c) => c,
                Err(e) => {
                    // Leave the episodes pending; retry next pass.
                    tracing::warn!(session = %s.key, "distill failed: {e}");
                    continue;
                }
            }
        } else {
            extract_heuristic(&eps)
        };
        let ctx = engine.ctx_for(&s.harness, &s.key, s.cwd.as_deref());
        let mut n = 0;
        for c in cands {
            match engine.write(&ctx, c).await {
                Ok(WriteOutcome::Stored { .. }) => n += 1,
                Ok(_) => {}
                Err(e) => tracing::warn!("distill write failed: {e}"),
            }
        }
        engine.store.mark_distilled(&s.key, upto)?;
        engine.store.trace("distill", &json!({"session": s.key, "episodes": eps.len(), "stored": n, "by": if use_model { "model" } else { "heuristic" }}));
        stored += n;
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
