//! Light normalisation shared by the lexical index, the hashing embedder and the subject/duplicate checks:
//! a tiny English suffix stripper (`deploying` / `deployed` / `deploys` → `deploy`) and a small bilingual hint
//! table of generic developer vocabulary (部署 ↔ deploy, 端口 ↔ port …). Both are applied symmetrically, at
//! index time and at query time, so they only ever *add* matching evidence; the original tokens stay.
//!
//! Concept tokens are prefixed with `~` so they can never collide with a word from the text.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::util::tokenize;

/// Generic developer vocabulary: (concept, English surface forms, Chinese terms). Keep it generic and small.
const HINTS: &[(&str, &[&str], &[&str])] = &[
    ("deploy", &["deploy", "deployment", "deploys"], &["部署", "上线"]),
    ("test", &["test", "tests", "testing"], &["测试", "单测"]),
    ("port", &["port", "ports"], &["端口"]),
    ("database", &["database", "databases", "db"], &["数据库"]),
    ("dependency", &["dependency", "dependencies", "deps"], &["依赖"]),
    ("release", &["release", "releases"], &["发布", "发版"]),
    ("config", &["config", "configuration", "configure", "settings"], &["配置"]),
    ("branch", &["branch", "branches"], &["分支"]),
    ("commit", &["commit", "commits"], &["提交"]),
    ("build", &["build", "builds"], &["构建"]),
    ("compile", &["compile", "compiler", "compilation"], &["编译"]),
    ("version", &["version", "versions"], &["版本"]),
    ("upgrade", &["upgrade", "upgraded", "bump", "bumped"], &["升级"]),
    ("migrate", &["migrate", "migration", "migrations"], &["迁移"]),
    ("log", &["log", "logs", "logging"], &["日志"]),
    ("error", &["error", "errors"], &["错误", "报错"]),
    ("fix", &["fix", "fixes", "bugfix"], &["修复"]),
    ("install", &["install", "installation"], &["安装"]),
    ("server", &["server", "servers"], &["服务器"]),
    ("cache", &["cache", "cached", "caching"], &["缓存"]),
    ("queue", &["queue", "queues"], &["队列"]),
    ("schedule", &["schedule", "scheduled", "cron", "cronjob"], &["定时", "调度"]),
    ("timeout", &["timeout", "timeouts"], &["超时"]),
    ("retry", &["retry", "retries"], &["重试"]),
    ("limit", &["limit", "limits", "cap", "capped", "max", "maximum"], &["上限", "限制", "最大"]),
    ("password", &["password", "passwords", "credential", "credentials"], &["密码", "凭据"]),
    ("permission", &["permission", "permissions"], &["权限"]),
    ("approve", &["approve", "approval", "approver", "signoff", "sign-off"], &["批准", "审批"]),
    ("review", &["review", "reviewer", "reviews"], &["审查", "评审", "审阅"]),
    ("merge", &["merge", "merges", "merged"], &["合并"]),
    ("repo", &["repo", "repository", "repositories", "repos"], &["仓库"]),
    ("directory", &["directory", "folder", "dir"], &["目录", "文件夹"]),
    ("script", &["script", "scripts"], &["脚本"]),
    ("command", &["command", "commands", "cli"], &["命令"]),
    ("environment", &["environment", "environments", "env"], &["环境"]),
    ("variable", &["variable", "variables", "var"], &["变量"]),
    ("production", &["production", "prod"], &["生产"]),
    ("docs", &["doc", "docs", "documentation"], &["文档"]),
    ("comment", &["comment", "comments"], &["注释"]),
    ("format", &["format", "formatting", "formatter"], &["格式"]),
    ("language", &["language", "languages"], &["语言"]),
    ("editor", &["editor", "ide"], &["编辑器"]),
    ("terminal", &["terminal", "shell"], &["终端"]),
    ("font", &["font", "fonts"], &["字体"]),
    ("image", &["image", "images"], &["镜像"]),
    ("container", &["container", "containers"], &["容器"]),
    ("cluster", &["cluster", "clusters"], &["集群"]),
    ("memory", &["memory", "ram"], &["内存", "显存"]),
    ("storage", &["storage", "disk", "quota"], &["存储", "磁盘", "配额"]),
    ("backup", &["backup", "backups", "snapshot"], &["备份", "快照"]),
    ("alert", &["alert", "alerts", "alerting"], &["告警", "报警"]),
    ("threshold", &["threshold", "thresholds"], &["阈值"]),
    ("performance", &["performance", "fps", "latency"], &["性能", "帧率", "延迟"]),
    ("interval", &["interval", "period", "frequency"], &["间隔", "周期", "频率"]),
    ("deadline", &["deadline", "due"], &["截止", "期限"]),
    ("timezone", &["timezone", "tz"], &["时区"]),
    ("reply", &["reply", "replies", "respond", "answer", "answers"], &["回复", "回答"]),
    ("english", &["english"], &["英文", "英语"]),
    ("chinese", &["chinese"], &["中文", "汉语"]),
    ("delete", &["delete", "deleted", "remove", "removed"], &["删除"]),
    ("plot", &["plot", "plots", "chart", "charts", "figure", "figures"], &["作图", "画图", "图表"]),
    ("color", &["color", "colors", "colour", "colours"], &["颜色", "配色"]),
    ("reason", &["reason", "reasons", "because", "why", "rationale"], &["原因", "为什么", "因为", "为了", "为啥"]),
    ("translation", &["translation", "translations", "translate", "i18n", "locale", "locales"], &["翻译"]),
];

struct Tables {
    /// stemmed English surface form → concept token
    en: HashMap<String, String>,
    /// Chinese term → concept token
    zh: Vec<(String, String)>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut en = HashMap::new();
        let mut zh = vec![];
        for (c, ens, zhs) in HINTS {
            let concept = format!("~{c}");
            for w in *ens {
                for t in tokenize(w) {
                    en.insert(stem(&t), concept.clone());
                }
            }
            for z in *zhs {
                zh.push((z.to_string(), concept.clone()));
            }
        }
        Tables { en, zh }
    })
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0x20000..=0x2A6DF)
}

pub fn is_cjk_token(t: &str) -> bool {
    t.chars().next().is_some_and(is_cjk)
}

/// Minimal English suffix stripping. Deterministic and conservative: only plain ASCII lowercase words of
/// 4+ letters, one inflectional suffix, a trailing silent `e` folded (`release` / `released` → `releas`).
pub fn stem(w: &str) -> String {
    if w.len() < 4 || !w.bytes().all(|b| b.is_ascii_lowercase()) {
        return w.to_string();
    }
    let mut s = w.to_string();
    if let Some(x) = s.strip_suffix("ies").filter(|x| x.len() >= 3) {
        s = format!("{x}y");
    } else if s.ends_with("sses") {
        s.truncate(s.len() - 2);
    } else if s.ends_with('s') && !s.ends_with("ss") && !s.ends_with("us") && !s.ends_with("is") && s.len() >= 4 {
        s.truncate(s.len() - 1);
    }
    let undouble = |x: &str| -> String {
        let b = x.as_bytes();
        let n = b.len();
        if n >= 4 && b[n - 1] == b[n - 2] && !matches!(b[n - 1], b'l' | b's' | b'z' | b'e' | b'o') {
            x[..n - 1].to_string()
        } else {
            x.to_string()
        }
    };
    if let Some(x) = s.strip_suffix("ing").filter(|x| x.len() >= 3 && x.bytes().any(|b| b"aeiouy".contains(&b))) {
        s = undouble(x);
    } else if let Some(x) = s.strip_suffix("ed").filter(|x| x.len() >= 3 && !x.ends_with('e') && x.bytes().any(|b| b"aeiouy".contains(&b)))
    {
        s = undouble(x);
    }
    if s.len() >= 4 && s.ends_with('e') {
        s.truncate(s.len() - 1);
    }
    s
}

/// Parts of a compound identifier: `SyncPeriod` → sync, period; `minimum_chrome_version` → minimum, chrome,
/// version; `kelp-operator` stays whole (the tokenizer already splits on `-`). Lowercased; empty if not compound.
pub fn identifier_parts(raw: &str) -> Vec<String> {
    let mut parts: Vec<String> = vec![];
    for chunk in raw.split('_') {
        let cs: Vec<char> = chunk.chars().collect();
        let mut cur = String::new();
        for (i, c) in cs.iter().enumerate() {
            let boundary = i > 0
                && c.is_uppercase()
                && (cs[i - 1].is_lowercase() || cs.get(i + 1).is_some_and(|n| n.is_lowercase()) && cs[i - 1].is_uppercase());
            if boundary && !cur.is_empty() {
                parts.push(cur.to_lowercase());
                cur.clear();
            }
            cur.push(*c);
        }
        if !cur.is_empty() {
            parts.push(cur.to_lowercase());
        }
    }
    parts.retain(|p| p.chars().count() >= 2 && !p.chars().all(|c| c.is_ascii_digit()));
    if parts.len() < 2 {
        return vec![];
    }
    parts
}

/// Raw (case-preserving) word tokens of `s`: alphanumeric / underscore runs that contain a letter.
fn raw_words(s: &str) -> Vec<&str> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_') || is_cjk(c)).filter(|w| w.chars().any(|c| c.is_ascii_alphabetic())).collect()
}

/// Concept tokens (`~deploy`, …) found in `text`, from English words (after stemming) and Chinese terms.
pub fn concepts(text: &str) -> Vec<String> {
    let t = tables();
    let mut out: Vec<String> = vec![];
    for tok in tokenize(text) {
        if let Some(c) = t.en.get(&stem(&tok)) {
            if !out.contains(c) {
                out.push(c.clone());
            }
        }
    }
    if text.chars().any(is_cjk) {
        for (z, c) in &t.zh {
            if text.contains(z.as_str()) && !out.contains(c) {
                out.push(c.clone());
            }
        }
    }
    out
}

/// Index terms of a note: original tokens, stems of English words that differ from the word, CJK unigrams
/// (as a fallback for short Chinese queries whose bigrams straddle word boundaries) and concept tokens.
pub fn index_terms(text: &str) -> Vec<String> {
    let toks = tokenize(text);
    let mut out = Vec::with_capacity(toks.len() * 2);
    for t in &toks {
        out.push(t.clone());
        let s = stem(t);
        if s != *t {
            out.push(s);
        }
    }
    for w in raw_words(text) {
        for p in identifier_parts(w) {
            out.push(stem(&p));
        }
    }
    for c in text.chars().filter(|c| is_cjk(*c)) {
        out.push(format!("{c}\u{1}"));
    }
    out.extend(concepts(text));
    out
}

/// One weighted query term. Terms sharing a `group` are variants of one query word (the word and its stem);
/// `group == None` marks auxiliary evidence (CJK unigrams) that scores but does not count towards coverage.
#[derive(Debug, Clone)]
pub struct QTerm {
    pub term: String,
    pub weight: f64,
    pub group: Option<usize>,
}

/// Weighted query terms: exact tokens weigh 1, stems 0.8, concepts 0.6, CJK unigrams 0.25; stopwords and
/// question words are dropped (unless the query is nothing but those).
pub fn query_terms(q: &str) -> Vec<QTerm> {
    let mut out: Vec<QTerm> = vec![];
    let mut groups = 0usize;
    let push = |out: &mut Vec<QTerm>, term: String, weight: f64, group: Option<usize>| {
        if !out.iter().any(|x| x.term == term) {
            out.push(QTerm { term, weight, group });
        }
    };
    let toks = tokenize(q);
    let skip = |t: &str| crate::screen::is_stopword(t) || crate::screen::is_question_word(t);
    let use_all = toks.iter().all(|t| skip(t));
    for t in &toks {
        if skip(t) && !use_all || out.iter().any(|x| &x.term == t) {
            continue;
        }
        let g = groups;
        groups += 1;
        let one_ascii = t.is_ascii() && t.chars().count() == 1;
        push(&mut out, t.clone(), if one_ascii { 0.3 } else { 1.0 }, Some(g));
        let s = stem(t);
        if s != *t {
            push(&mut out, s, 0.8, Some(g));
        }
    }
    for w in raw_words(q) {
        let parts = identifier_parts(w);
        let g = out.iter().find(|x| x.term == w.to_lowercase()).and_then(|x| x.group);
        for p in parts {
            let p = stem(&p);
            push(&mut out, p, 0.7, g);
        }
    }
    for c in q.chars().filter(|c| is_cjk(*c)) {
        if !crate::screen::is_cjk_function_char(c) {
            push(&mut out, format!("{c}\u{1}"), 0.25, None);
        }
    }
    for c in concepts(q) {
        let g = groups;
        groups += 1;
        push(&mut out, c, 0.6, Some(g));
    }
    out
}

/// Normalised content tokens for subject / duplicate comparison: stems instead of words, plus concepts.
pub fn normalized_set(tokens: &HashSet<String>) -> HashSet<String> {
    tokens.iter().map(|t| stem(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stems() {
        for (a, b) in [
            ("deploying", "deployed"),
            ("deploys", "deploy"),
            ("releases", "release"),
            ("running", "run"),
            ("rebasing", "rebase"),
            ("dependencies", "dependency"),
        ] {
            assert_eq!(stem(a), stem(b), "{a} {b}");
        }
        assert_eq!(stem("need"), "need");
        assert_eq!(stem("status"), "status");
        assert_eq!(stem("use"), "use");
    }
    #[test]
    fn identifiers_split() {
        assert_eq!(identifier_parts("SyncPeriod"), vec!["sync", "period"]);
        assert_eq!(identifier_parts("minimum_chrome_version"), vec!["minimum", "chrome", "version"]);
        assert_eq!(identifier_parts("HTTPServer"), vec!["http", "server"]);
        assert!(identifier_parts("deploy").is_empty());
        assert!(index_terms("the manager SyncPeriod is 10m").contains(&"period".to_string()));
    }
    #[test]
    fn bilingual_concepts() {
        assert!(concepts("怎么部署到生产环境").contains(&"~deploy".to_string()));
        assert!(concepts("how do we deploy to production").contains(&"~deploy".to_string()));
        assert!(concepts("数据库端口是多少").contains(&"~port".to_string()));
        assert!(index_terms("Deployment notes").contains(&"~deploy".to_string()));
        let q = query_terms("What is the deploy command?");
        assert!(q.iter().all(|t| t.term != "what" && t.term != "the"));
        assert!(q.iter().any(|t| t.term == "~command"));
    }
}
