//! Small shared helpers: time, hashing, tokenization.

use sha2::{Digest, Sha256};

pub fn now_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn fmt_date(ts: i64) -> String {
    time::OffsetDateTime::from_unix_timestamp(ts)
        .ok()
        .and_then(|t| t.format(&time::macros::format_description!("[year]-[month]-[day]")).ok())
        .unwrap_or_else(|| "?".into())
}

/// Short, stable content hash (128 bits, hex).
pub fn hash_hex(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    let d = h.finalize();
    d[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// Monotonic ULID-based id: ids sort in creation order even within one millisecond, so "newest" is well defined.
pub fn new_id() -> String {
    static LAST: std::sync::Mutex<Option<ulid::Ulid>> = std::sync::Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let mut u = ulid::Ulid::new();
    if let Some(l) = *last {
        if u <= l {
            u = l.increment().unwrap_or(u);
        }
    }
    *last = Some(u);
    format!("m_{}", u.to_string().to_lowercase())
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0x20000..=0x2A6DF)
}

/// Language-agnostic tokenizer used for both the FTS index and the hashing embedder:
/// lowercase alphanumeric words (split on non-word chars and camelCase / snake boundaries are kept whole),
/// plus overlapping CJK bigrams (and lone CJK chars), so Chinese/Japanese text is searchable without a dictionary.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut cjk_run: Vec<char> = Vec::new();
    let flush_word = |w: &mut String, out: &mut Vec<String>| {
        if !w.is_empty() {
            out.push(w.to_lowercase());
            w.clear();
        }
    };
    let flush_cjk = |run: &mut Vec<char>, out: &mut Vec<String>| {
        match run.len() {
            0 => {}
            1 => out.push(run[0].to_string()),
            _ => {
                for w in run.windows(2) {
                    out.push(w.iter().collect());
                }
            }
        }
        run.clear();
    };
    for c in s.chars() {
        if is_cjk(c) {
            flush_word(&mut word, &mut out);
            cjk_run.push(c);
        } else if c.is_alphanumeric() || c == '_' {
            flush_cjk(&mut cjk_run, &mut out);
            word.push(c);
        } else {
            flush_word(&mut word, &mut out);
            flush_cjk(&mut cjk_run, &mut out);
        }
    }
    flush_word(&mut word, &mut out);
    flush_cjk(&mut cjk_run, &mut out);
    out
}

pub fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push('…');
        t
    }
}

pub fn expand_tilde(p: &str) -> std::path::PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest);
        }
    }
    std::path::PathBuf::from(p)
}

/// Remove `<system-reminder>…</system-reminder>` and our own `<unimem…>` blocks from user text.
pub fn strip_harness_noise(s: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"(?s)<system-reminder>.*?</system-reminder>|<unimem[^>]*>.*?</unimem[^>]*>|<environment_context>.*?</environment_context>")
            .unwrap()
    });
    re.replace_all(s, "").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokenizes_mixed() {
        let t = tokenize("Deploy via ops/deploy.sh 部署脚本");
        assert!(t.contains(&"deploy".to_string()));
        assert!(t.contains(&"sh".to_string()));
        assert!(t.contains(&"部署".to_string()));
        assert!(t.contains(&"脚本".to_string()));
    }
    #[test]
    fn strips_noise() {
        assert_eq!(strip_harness_noise("<system-reminder>x</system-reminder> hi"), "hi");
    }
}
