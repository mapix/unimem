//! Configuration: `~/.unimem/config.toml`, every field optional.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::util::expand_tilde;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub listen: String,
    pub data_dir: String,
    pub upstream: Upstream,
    pub inject: Inject,
    pub decider: DeciderCfg,
    pub embedder: EmbedderCfg,
    pub distill: DistillCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Upstream {
    /// Upstream for the Anthropic dialect (`/v1/messages`).
    pub anthropic: String,
    /// Upstream for OpenAI dialects with an API key.
    pub openai: String,
    /// Upstream for Codex signed in with ChatGPT (requests carrying `chatgpt-account-id`).
    pub chatgpt: String,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Inject {
    /// Master switch; when false unimem is a pure pass-through recorder.
    pub enabled: bool,
    /// Expose `unimem_*` tools to the model.
    pub tools: bool,
    /// Passive recall appended to each new user prompt.
    pub recall: bool,
    /// Session preamble (project card + pinned memories) on the first user message.
    pub preamble: bool,
    pub recall_budget_ms: u64,
    pub max_recall_notes: usize,
    pub preamble_max_notes: usize,
    /// Max internal continuation rounds per visible response.
    pub max_continuations: usize,
    /// Record every prompt turn into the episode log.
    pub record_episodes: bool,
    /// Token budgets (see `budget.rs`). `false` disables every cap below (for comparisons only).
    pub budgets: bool,
    /// Skip notes whose id is still visible in the live request (cross-turn de-dup).
    pub dedup: bool,
    /// Per-message cap for passive recall + hints: min(this, 0.6% of the context window).
    pub max_inject_tokens: usize,
    /// Preamble cap: min(this, 1% of the context window).
    pub max_preamble_tokens: usize,
    /// Cap on one unimem tool result.
    pub max_tool_tokens: usize,
    /// Everything unimem keeps in one live context: at most this share of the window (passive recall stops beyond).
    pub max_session_share: f64,
    /// Model id prefix → context window (tokens); `"*"` sets the fallback for unknown models.
    pub context_windows: std::collections::BTreeMap<String, u64>,
    /// Recall on tool-result turns that show an error (≤2 notes, `tool_recall_max_tokens`).
    pub tool_recall: bool,
    /// At most one tool-triggered recall per this many tool turns of a conversation.
    pub tool_recall_every: u64,
    pub tool_recall_max_tokens: usize,
    /// Pre-compaction flush: when the estimated context exceeds this share of the window, distill the session in
    /// the background and nudge the agent once to save durable facts.
    pub flush_at: f64,
    /// Append a short "save it with unimem_remember" hint when the user corrects the agent.
    pub correction_hints: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeciderCfg {
    /// Tried in order; unavailable backends fall through. Values: "jev", "llm", "heuristic".
    pub chain: Vec<String>,
    pub jev: JevCfg,
    pub llm: LlmCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JevCfg {
    pub base_url: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmCfg {
    /// Any OpenAI-compatible `/v1/chat/completions` endpoint (Ollama, vLLM, OpenAI, a proxy…).
    pub base_url: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub model: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedderCfg {
    /// "hash" (offline, zero-setup), "onnx" (local transformer, e.g. multilingual-e5-small; needs a build with
    /// `--features onnx`, `unimem models download e5-small` and ONNX Runtime) or "openai" (any `/v1/embeddings`).
    pub kind: String,
    pub base_url: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    /// openai: the model name; onnx: the directory holding `model.onnx` + `tokenizer.json`.
    pub model: String,
    pub dim: usize,
    /// onnx: path of the ONNX Runtime shared library (`libonnxruntime.so` / `.dylib`); default `$ORT_DYLIB_PATH`.
    pub ort_dylib: String,
    /// onnx: intra-op threads (0 = 4).
    pub threads: usize,
    /// Near-duplicate cosine for this embedder (default per embedder: hash 0.97, e5 0.95).
    pub near_dup_cos: Option<f32>,
    /// Typical cosine of unrelated texts, subtracted before the code-side priors (default: hash 0, e5 0.75).
    pub cos_baseline: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillCfg {
    pub enabled: bool,
    pub base_url: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub model: String,
    /// A session is distilled after being idle this long.
    pub idle_secs: u64,
    /// Episodes older than this are deleted (distilled or not).
    pub episode_ttl_days: u64,
    /// Skip background distillation when the last upstream response showed less than this share of the rate limit
    /// remaining (`anthropic-ratelimit-*-remaining` / `x-ratelimit-remaining-*`).
    pub min_rate_limit_remaining: f64,
    /// Characters of transcript per distillation call (sessions are processed in chunks of this size).
    pub chunk_chars: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:4777".into(),
            data_dir: "~/.unimem".into(),
            upstream: Upstream::default(),
            inject: Inject::default(),
            decider: DeciderCfg::default(),
            embedder: EmbedderCfg::default(),
            distill: DistillCfg::default(),
        }
    }
}
impl Default for Upstream {
    fn default() -> Self {
        Self {
            anthropic: "https://api.anthropic.com".into(),
            openai: "https://api.openai.com".into(),
            chatgpt: "https://chatgpt.com/backend-api/codex".into(),
            timeout_secs: 600,
        }
    }
}
impl Default for Inject {
    fn default() -> Self {
        Self {
            enabled: true,
            tools: true,
            recall: true,
            preamble: true,
            recall_budget_ms: 1500,
            max_recall_notes: 5,
            preamble_max_notes: 8,
            max_continuations: 6,
            record_episodes: true,
            budgets: true,
            dedup: true,
            max_inject_tokens: 1200,
            max_preamble_tokens: 1500,
            max_tool_tokens: 3000,
            max_session_share: 0.05,
            context_windows: Default::default(),
            tool_recall: true,
            tool_recall_every: 5,
            tool_recall_max_tokens: 400,
            flush_at: 0.7,
            correction_hints: true,
        }
    }
}
impl Default for DeciderCfg {
    fn default() -> Self {
        Self { chain: vec!["jev".into(), "llm".into(), "heuristic".into()], jev: JevCfg::default(), llm: LlmCfg::default() }
    }
}
impl Default for JevCfg {
    fn default() -> Self {
        Self {
            base_url: "https://api.typesafe.ai".into(),
            api_key_env: "TYPESAFE_API_KEY".into(),
            api_key: None,
            model: None,
            timeout_ms: 8000,
        }
    }
}
impl Default for LlmCfg {
    fn default() -> Self {
        Self { base_url: String::new(), api_key_env: "UNIMEM_LLM_API_KEY".into(), api_key: None, model: String::new(), timeout_ms: 15000 }
    }
}
impl Default for EmbedderCfg {
    fn default() -> Self {
        Self {
            kind: "hash".into(),
            base_url: String::new(),
            api_key_env: "UNIMEM_EMBED_API_KEY".into(),
            api_key: None,
            model: String::new(),
            dim: 512,
            ort_dylib: String::new(),
            threads: 0,
            near_dup_cos: None,
            cos_baseline: None,
        }
    }
}
impl Default for DistillCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: String::new(),
            api_key_env: "UNIMEM_DISTILL_API_KEY".into(),
            api_key: None,
            model: String::new(),
            idle_secs: 600,
            episode_ttl_days: 30,
            min_rate_limit_remaining: 0.1,
            chunk_chars: 24_000,
        }
    }
}

pub fn resolve_key(explicit: &Option<String>, env: &str) -> Option<String> {
    explicit.clone().filter(|s| !s.is_empty()).or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
}

impl Config {
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("UNIMEM_CONFIG") {
            return PathBuf::from(p);
        }
        expand_tilde("~/.unimem/config.toml")
    }

    pub fn load(path: Option<&std::path::Path>) -> anyhow::Result<Self> {
        let p = path.map(PathBuf::from).unwrap_or_else(Self::default_path);
        let mut cfg: Config = if p.exists() {
            toml::from_str(&std::fs::read_to_string(&p)?).map_err(|e| anyhow::anyhow!("invalid config {}: {e}", p.display()))?
        } else {
            Config::default()
        };
        if let Ok(d) = std::env::var("UNIMEM_DATA_DIR") {
            cfg.data_dir = d;
        }
        if let Ok(l) = std::env::var("UNIMEM_LISTEN") {
            cfg.listen = l;
        }
        Ok(cfg)
    }

    pub fn data_dir(&self) -> PathBuf {
        expand_tilde(&self.data_dir)
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir().join("unimem.db")
    }

    pub fn sample() -> String {
        toml::to_string_pretty(&Config::default()).unwrap_or_default()
    }
}
