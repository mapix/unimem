//! Token budgets for everything unimem injects.
//!
//! The estimator is deliberately cheap and tokenizer-free: about 4 ASCII characters per token, about 2 other
//! non-CJK characters per token, and one token per CJK character (Claude and o200k tokenizers typically spend
//! 1–1.5 tokens per Han character, so this slightly *under*-counts CJK and over-counts nothing else much). It is
//! only used to keep injections inside caps that sit far below the context window, so ±30% is good enough.
//!
//! Context windows come from a static model table, overridable with `inject.context_windows` (longest matching
//! model-id prefix wins). Defaults are conservative: Claude 200k (1M when the model id carries `[1m]` or the
//! request's `anthropic-beta` header enables a `context-1m` beta), GPT-5 / Codex class 272k, o-series 200k,
//! GPT-4o class 128k, anything unknown 128k.

use std::collections::BTreeMap;

use crate::config::Inject;

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x3000..=0x303F | 0x20000..=0x2A6DF)
}

/// Rough token count of `s` (see the module docs).
pub fn estimate_tokens(s: &str) -> usize {
    let (mut cjk, mut ascii, mut other) = (0usize, 0usize, 0usize);
    for c in s.chars() {
        if c.is_ascii() {
            ascii += 1;
        } else if is_cjk(c) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + ascii.div_ceil(4) + other.div_ceil(2)
}

/// Cut `s` so that its estimate is at most `max` tokens (on a char boundary), appending `…` when cut.
pub fn truncate_tokens(s: &str, max: usize) -> String {
    if estimate_tokens(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let (mut cjk, mut ascii, mut other) = (0usize, 0usize, 0usize);
    for c in s.chars() {
        if c.is_ascii() {
            ascii += 1;
        } else if is_cjk(c) {
            cjk += 1;
        } else {
            other += 1;
        }
        if cjk + ascii.div_ceil(4) + other.div_ceil(2) + 1 > max {
            break;
        }
        out.push(c);
    }
    out.push('…');
    out
}

/// Context window (tokens) of `model`. `beta` is the request's `anthropic-beta` header ("" if none).
pub fn context_window(model: &str, beta: &str, overrides: &BTreeMap<String, u64>) -> u64 {
    let m = model.trim().to_lowercase();
    if let Some((_, w)) = overrides.iter().filter(|(k, _)| m.starts_with(&k.to_lowercase())).max_by_key(|(k, _)| k.len()) {
        return *w;
    }
    let one_m = m.contains("[1m]") || m.ends_with("-1m") || beta.to_lowercase().contains("context-1m");
    if m.starts_with("claude") || m.contains("sonnet") || m.contains("opus") || m.contains("haiku") {
        return if one_m { 1_000_000 } else { 200_000 };
    }
    if m.starts_with("gpt-5") || m.contains("codex") {
        return 272_000;
    }
    if m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") {
        return 200_000;
    }
    if let Some(w) = overrides.get("*") {
        return *w;
    }
    128_000
}

/// Token caps for one request, derived from the config and the model's window.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub window: u64,
    /// Passive recall + hints appended to one message.
    pub turn: usize,
    /// Session preamble.
    pub preamble: usize,
    /// One unimem tool result.
    pub tool: usize,
    /// Tool-result-triggered recall.
    pub tool_recall: usize,
    /// Everything unimem keeps in the live context of one conversation (replayed overlays + new text).
    pub session: usize,
    pub enabled: bool,
}

impl Budget {
    pub fn new(cfg: &Inject, window: u64) -> Self {
        if !cfg.budgets {
            return Self {
                window,
                turn: usize::MAX,
                preamble: usize::MAX,
                tool: usize::MAX,
                tool_recall: usize::MAX,
                session: usize::MAX,
                enabled: false,
            };
        }
        let share = |f: f64| (window as f64 * f) as usize;
        Self {
            window,
            turn: cfg.max_inject_tokens.min(share(0.006)).max(64),
            preamble: cfg.max_preamble_tokens.min(share(0.01)).max(128),
            tool: cfg.max_tool_tokens,
            tool_recall: cfg.tool_recall_max_tokens,
            session: share(cfg.max_session_share),
            enabled: true,
        }
    }
    /// The tool-output cap alone (no window needed).
    pub fn tool_cap(cfg: &Inject) -> usize {
        if cfg.budgets {
            cfg.max_tool_tokens
        } else {
            usize::MAX
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_ascii_and_cjk() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("部署脚本"), 4);
        // mixed: 4 CJK + 8 ASCII
        assert_eq!(estimate_tokens("用 ops/deploy 部署"), 3 + 3);
        let long = "word ".repeat(400);
        assert_eq!(estimate_tokens(&long), 500);
    }

    #[test]
    fn truncation_respects_cap() {
        let s = "数据库".repeat(100);
        let t = truncate_tokens(&s, 50);
        assert!(estimate_tokens(&t) <= 51, "{}", estimate_tokens(&t));
        assert!(t.ends_with('…'));
        assert_eq!(truncate_tokens("short", 10), "short");
    }

    #[test]
    fn windows_table_and_overrides() {
        let none = BTreeMap::new();
        assert_eq!(context_window("claude-sonnet-4-5", "", &none), 200_000);
        assert_eq!(context_window("claude-sonnet-4-5[1m]", "", &none), 1_000_000);
        assert_eq!(context_window("claude-sonnet-4-5", "oauth-2025-04-20,context-1m-2025-08-07", &none), 1_000_000);
        assert_eq!(context_window("gpt-5-codex", "", &none), 272_000);
        assert_eq!(context_window("some-local-model", "", &none), 128_000);
        let mut o = BTreeMap::new();
        o.insert("claude".to_string(), 150_000);
        o.insert("claude-opus".to_string(), 500_000);
        o.insert("*".to_string(), 32_000);
        assert_eq!(context_window("claude-sonnet-4", "", &o), 150_000);
        assert_eq!(context_window("claude-opus-4", "", &o), 500_000);
        assert_eq!(context_window("llama3", "", &o), 32_000);
    }

    #[test]
    fn budget_caps_scale_with_window() {
        let cfg = Inject::default();
        let b = Budget::new(&cfg, 200_000);
        assert_eq!(b.turn, 1200);
        assert_eq!(b.preamble, 1500);
        assert_eq!(b.session, 10_000);
        let small = Budget::new(&cfg, 32_000);
        assert_eq!(small.turn, 192);
        assert_eq!(small.preamble, 320);
        assert_eq!(small.session, 1600);
        let off = Budget::new(&Inject { budgets: false, ..Inject::default() }, 200_000);
        assert_eq!(off.turn, usize::MAX);
    }
}
