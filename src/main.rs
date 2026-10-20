//! `unimem` — the unimem gateway and its command-line interface.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use unimem::config::Config;
use unimem::engine::{Ctx, Engine, Hit, RecallReq, WriteOutcome, WriteReq};
use unimem::gateway::{self, AppState};
use unimem::project::scope_of;
use unimem::store::{Memory, Store};
use unimem::util::{fmt_date, truncate_chars};

#[derive(Parser)]
#[command(name = "unimem", version, about = "unimem: one shared, project-aware long-term memory for every coding agent")]
struct Cli {
    /// Config file (default: $UNIMEM_CONFIG or ~/.unimem/config.toml).
    #[arg(long, global = true, env = "UNIMEM_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the gateway (Anthropic / OpenAI Responses / Chat Completions / MCP).
    Serve {
        /// Listen address (default from config: 127.0.0.1:4777).
        #[arg(long)]
        listen: Option<String>,
    },
    /// Store a memory.
    Remember {
        content: String,
        #[arg(long)]
        why: Option<String>,
        /// auto (default), global, project, or a known project name.
        #[arg(long)]
        scope: Option<String>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        pin: bool,
        /// Working directory used to detect the project (default: current dir).
        #[arg(long)]
        cwd: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Search memory (current project + global).
    Recall {
        query: String,
        /// Also search another project (name / pid) or "all".
        #[arg(long)]
        project: Option<String>,
        /// Return the k best notes (ranked, with scores). Default: the adaptive cut-off (the few clearly relevant notes, at most 5).
        #[arg(short, long)]
        k: Option<usize>,
        #[arg(long)]
        cwd: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Delete a memory by id.
    Forget { id: String },
    /// List memories.
    List {
        /// global, project (current dir), a project name / pid, or a raw scope.
        #[arg(long)]
        scope: Option<String>,
        /// Include superseded memories.
        #[arg(long)]
        all: bool,
        #[arg(short = 'n', long, default_value_t = 30)]
        n: usize,
        #[arg(long)]
        cwd: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// List known projects.
    Projects {
        #[arg(long)]
        json: bool,
    },
    /// Show (or update) the project card for a directory.
    Project {
        #[arg(long)]
        cwd: Option<String>,
        /// Set the project summary.
        #[arg(long)]
        summary: Option<String>,
        /// Set the display name.
        #[arg(long)]
        name: Option<String>,
    },
    /// Counts and configuration overview.
    Stats {
        #[arg(long)]
        json: bool,
    },
    /// Recent engine decisions (writes, recalls, distills).
    Trace {
        #[arg(short = 'n', long, default_value_t = 20)]
        n: usize,
    },
    /// Recompute all embeddings with the configured embedder.
    Reembed,
    /// Local embedding models for `[embedder] kind = "onnx"`.
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Distill idle sessions' episodes into memories now.
    Distill {
        /// Ignore the idle threshold (distill every session with pending episodes).
        #[arg(long)]
        now: bool,
    },
    /// Print the config path and a commented sample config.
    Config,
    /// Print setup instructions for a harness: claude-code, codex, shell, mcp.
    Setup { harness: String },
    /// Print the web UI URL (served by `unimem serve` at /ui) and open it in a browser.
    Ui {
        /// Only print the URL.
        #[arg(long)]
        no_open: bool,
    },
    /// Write ~45 realistic demo memories (3 projects + global, every kind, supersede chains, causal links).
    DemoSeed {
        /// Where to create the demo project repos (default: <data_dir>/demo-repos).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Download a model into <data_dir>/models/ (sha256-checked): e5-small = intfloat/multilingual-e5-small.
    Download {
        name: String,
        /// Target directory (default: <data_dir>/models/<model>).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// List downloadable models and what is installed.
    List,
}

async fn models_download(cfg: &Config, name: &str, dir: Option<PathBuf>) -> Result<()> {
    use sha2::Digest;
    let spec = unimem::embed::model_spec(name).ok_or_else(|| anyhow::anyhow!("unknown model '{name}' (known: e5-small)"))?;
    let dir = dir.unwrap_or_else(|| cfg.data_dir().join("models").join(spec.dir));
    std::fs::create_dir_all(&dir)?;
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(1800)).build()?;
    for (file, sha) in spec.files {
        let path = dir.join(file);
        let ok = |p: &std::path::Path| -> bool {
            sha.is_empty() || std::fs::read(p).map(|b| format!("{:x}", sha2::Sha256::digest(&b)) == *sha).unwrap_or(false)
        };
        if path.exists() && ok(&path) {
            println!("ok        {}", path.display());
            continue;
        }
        let url = format!("{}/{file}", spec.base_url);
        println!("download  {url}");
        let r = client.get(&url).send().await?.error_for_status()?;
        let bytes = r.bytes().await?;
        let tmp = path.with_extension("part");
        std::fs::write(&tmp, &bytes)?;
        if !ok(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            anyhow::bail!("sha256 mismatch for {file}");
        }
        std::fs::rename(&tmp, &path)?;
    }
    println!(
        "\n{} is in {}. To use it, build with `cargo build --release --features onnx`, then in {}:\n\n[embedder]\nkind  = \"onnx\"\nmodel = \"{}\"\n# ort_dylib = \"/path/to/libonnxruntime.so\"   # or set ORT_DYLIB_PATH (e.g. from `pip install onnxruntime`)\n\nand run `unimem reembed` once.",
        spec.name,
        dir.display(),
        Config::default_path().display(),
        dir.display()
    );
    Ok(())
}

const SAMPLE_CONFIG: &str = r#"# ~/.unimem/config.toml — every field is optional.
listen   = "127.0.0.1:4777"
data_dir = "~/.unimem"              # unimem.db lives here

[upstream]
anthropic    = "https://api.anthropic.com"               # /v1/messages
openai       = "https://api.openai.com"                  # Responses / Chat with an API key
chatgpt      = "https://chatgpt.com/backend-api/codex"   # Codex signed in with ChatGPT
timeout_secs = 600

[inject]
enabled           = true    # false = pure pass-through recorder
tools             = true    # expose unimem_recall / remember / forget / project
recall            = true    # passive recall appended to each new user prompt
preamble          = true    # project card + standing notes on the first user message
recall_budget_ms  = 1500    # passive recall is skipped if slower than this
max_recall_notes  = 5
preamble_max_notes = 8
max_continuations = 6       # internal tool rounds per visible response
record_episodes   = true    # keep a raw episode log (input for the distiller)

# System One: the fast typed-decision model. Backends are tried in order;
# unavailable ones fall through. "heuristic" is offline and always appended.
[decider]
chain = ["jev", "llm", "heuristic"]

[decider.jev]                      # TypeSafe Jev /v1/systemone (or a compatible local server)
base_url    = "https://api.typesafe.ai"
api_key_env = "TYPESAFE_API_KEY"   # used only if set
timeout_ms  = 8000

[decider.llm]                      # any OpenAI-compatible chat model, e.g. Ollama
base_url    = ""                   # "http://127.0.0.1:11434/v1"
model       = ""                   # "qwen2.5:7b-instruct"
api_key_env = "UNIMEM_LLM_API_KEY"
timeout_ms  = 15000

[embedder]
kind  = "hash"                     # "hash" (offline), "onnx" (local e5; --features onnx) or "openai" (any /v1/embeddings)
# kind      = "onnx"                       # `unimem models download e5-small` first
# model     = "~/.unimem/models/multilingual-e5-small"   # default for onnx
# ort_dylib = "/path/to/libonnxruntime.so" # or ORT_DYLIB_PATH
# kind     = "openai"
# base_url = "http://127.0.0.1:11434/v1"   # Ollama
# model    = "bge-m3"
# dim      = 1024
api_key_env = "UNIMEM_EMBED_API_KEY"
dim   = 512
# after changing the embedder run: unimem reembed

[distill]                          # passive capture: extract memories from idle sessions
enabled     = false
base_url    = ""                   # OpenAI-compatible; empty model + enabled = heuristic extractor
model       = ""
api_key_env = "UNIMEM_DISTILL_API_KEY"
idle_secs   = 600
"#;

fn setup_text(harness: &str, listen: &str) -> Option<String> {
    let base = format!("http://{listen}");
    Some(match harness {
        "claude-code" | "claude" | "cc" => format!(
            "# Claude Code — route model traffic through unimem (API key and subscription OAuth both pass through):\n\
             export ANTHROPIC_BASE_URL={base}\n\
             # add it to ~/.zshrc / ~/.bashrc, or to ~/.claude/settings.json:\n\
             #   {{ \"env\": {{ \"ANTHROPIC_BASE_URL\": \"{base}\" }} }}\n\n\
             # MCP-only alternative (no gateway; the agent calls the tools itself):\n\
             claude mcp add --transport http unimem {base}/mcp\n"
        ),
        "codex" => format!(
            "# Codex CLI — add to ~/.codex/config.toml:\n\
             model_provider = \"unimem\"\n\n\
             [model_providers.unimem]\n\
             name = \"unimem\"\n\
             base_url = \"{base}/v1\"\n\
             wire_api = \"responses\"\n\
             requires_openai_auth = true   # keeps your ChatGPT login / OPENAI_API_KEY\n\n\
             # MCP-only alternative (no gateway):\n\
             [mcp_servers.unimem]\n\
             url = \"{base}/mcp\"\n"
        ),
        "shell" | "env" => format!(
            "# Any OpenAI / Anthropic compatible client:\n\
             export ANTHROPIC_BASE_URL={base}\n\
             export OPENAI_BASE_URL={base}/v1\n"
        ),
        "mcp" => format!("# Streamable HTTP MCP endpoint:\n{base}/mcp\n"),
        _ => return None,
    })
}

fn open(cfg: &Config) -> Result<Engine> {
    let path = cfg.db_path();
    let store = Store::open(&path).with_context(|| format!("opening {}", path.display()))?;
    Ok(Engine::new(cfg.clone(), store))
}

fn cwd_or_here(cwd: Option<String>) -> Option<String> {
    cwd.or_else(|| std::env::current_dir().ok().map(|p| p.display().to_string()))
}

fn cli_ctx(e: &Engine, cwd: Option<String>) -> Ctx {
    let cwd = cwd_or_here(cwd);
    e.ctx_for("cli", "cli", cwd.as_deref())
}

fn scope_label(e: &Engine, scope: &str) -> String {
    if scope == "global" {
        return "global".into();
    }
    let pid = scope.trim_start_matches("project:");
    e.store.get_project(pid).map(|p| p.name).unwrap_or_else(|| pid.to_string())
}

fn print_hit(e: &Engine, h: &Hit) {
    println!(
        "{}  {}  [{} · {} · p={:.2}{}]",
        h.id,
        h.date,
        scope_label(e, &h.scope),
        h.kind,
        h.p,
        if h.superseded { " · superseded" } else { "" }
    );
    println!("    {}", h.content);
    if let Some(w) = &h.why {
        println!("    because: {w}");
    }
}

fn print_mem(e: &Engine, m: &Memory) {
    let mut flags = String::new();
    if m.pinned {
        flags.push_str(" · pinned");
    }
    if m.superseded_by.is_some() {
        flags.push_str(" · superseded");
    }
    println!(
        "{}  {}  [{} · {}{}]  via {}",
        m.id,
        fmt_date(m.created_at),
        scope_label(e, &m.scope),
        m.kind,
        flags,
        if m.source.harness.is_empty() { "?" } else { &m.source.harness }
    );
    println!("    {}", m.content);
    if let Some(w) = &m.why {
        println!("    because: {w}");
    }
}

fn pj<T: serde::Serialize>(v: &T) {
    println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(cli.config.as_deref())?;
    match cli.cmd {
        Cmd::Serve { listen } => serve(cfg, listen).await,
        Cmd::Remember { content, why, scope, kind, pin, cwd, json } => {
            let e = open(&cfg)?;
            let ctx = cli_ctx(&e, cwd);
            let out = e.write(&ctx, WriteReq { content, why, scope, kind, pin }).await?;
            if json {
                pj(&out);
                return Ok(());
            }
            match out {
                WriteOutcome::Stored { id, scope, kind, links, superseded, decided_by } => {
                    println!("stored {id}  [{} · {kind}]  (decided by {decided_by})", scope_label(&e, &scope));
                    for s in superseded {
                        println!("  supersedes {s}");
                    }
                    for (rel, id) in links.iter().filter(|l| l.0 != "supersedes") {
                        println!("  {rel} → {id}");
                    }
                }
                WriteOutcome::Duplicate { of, content } => println!("already known as {of}: {content}"),
                WriteOutcome::Rejected { reason } => {
                    eprintln!("not stored: {reason}");
                    std::process::exit(1);
                }
            }
            Ok(())
        }
        Cmd::Recall { query, project, k, cwd, json } => {
            let e = open(&cfg)?;
            let ctx = cli_ctx(&e, cwd);
            let depth = if k.is_some() { unimem::engine::Depth::TopK } else { unimem::engine::Depth::Adaptive };
            let out = e.recall(&ctx, RecallReq { query, project, k: k.unwrap_or(5), passive: false, depth }).await?;
            if json {
                pj(&out);
                return Ok(());
            }
            let scopes: Vec<String> = out.scopes.iter().map(|s| scope_label(&e, s)).collect();
            println!(
                "searched: {}  ·  mode {}  ·  decided by {}  ·  sufficient: {}",
                scopes.join(", "),
                out.mode,
                if out.decided_by.is_empty() { "-" } else { &out.decided_by },
                out.sufficient
            );
            if out.notes.is_empty() {
                println!("(no matching memories)");
            }
            for h in &out.notes {
                print_hit(&e, h);
            }
            Ok(())
        }
        Cmd::Forget { id } => {
            let e = open(&cfg)?;
            if e.forget(&id)? {
                println!("forgot {id}");
            } else {
                eprintln!("no memory with id {id}");
                std::process::exit(1);
            }
            Ok(())
        }
        Cmd::List { scope, all, n, cwd, json } => {
            let e = open(&cfg)?;
            let scopes: Option<Vec<String>> = match scope.as_deref() {
                None => None,
                Some("global") => Some(vec!["global".into()]),
                Some("project") => {
                    let ctx = cli_ctx(&e, cwd);
                    Some(ctx.project_scope().into_iter().collect())
                }
                Some(s) if s.starts_with("project:") => Some(vec![s.to_string()]),
                Some(name) => {
                    let n = name.to_lowercase();
                    match e.store.list_projects().into_iter().find(|p| {
                        p.pid.to_lowercase() == n || p.name.to_lowercase() == n || p.aliases.iter().any(|a| a.to_lowercase() == n)
                    }) {
                        Some(p) => Some(vec![scope_of(&p.pid)]),
                        None => anyhow::bail!("unknown scope / project '{name}'"),
                    }
                }
            };
            let mems = e.store.list_memories(scopes.as_deref(), all, n);
            if json {
                pj(&mems);
                return Ok(());
            }
            if mems.is_empty() {
                println!("(no memories)");
            }
            for m in &mems {
                print_mem(&e, m);
            }
            Ok(())
        }
        Cmd::Projects { json } => {
            let e = open(&cfg)?;
            let ps = e.store.list_projects();
            if json {
                pj(&ps);
                return Ok(());
            }
            let counts = e.store.count_memories();
            if ps.is_empty() {
                println!("(no projects yet — they are detected from the agents' working directories)");
            }
            for p in ps {
                let n = counts.get(&scope_of(&p.pid)).copied().unwrap_or(0);
                println!("{}  [{}]  {} memories", p.name, p.pid, n);
                println!("    root: {}", p.root);
                if !p.summary.is_empty() {
                    println!("    {}", truncate_chars(&p.summary, 200));
                }
            }
            Ok(())
        }
        Cmd::Project { cwd, summary, name } => {
            let e = open(&cfg)?;
            let ctx = cli_ctx(&e, cwd);
            let card = if summary.is_some() || name.is_some() {
                e.update_project(&ctx, name.as_deref(), summary.as_deref(), None)?
            } else {
                ctx.project.as_ref().map(|r| e.store.get_project(&r.project.pid).unwrap_or(r.project.clone()))
            };
            match card {
                Some(p) => {
                    println!("name:     {}", p.name);
                    println!("pid:      {}", p.pid);
                    println!("root:     {}", p.root);
                    if let Some(r) = &p.remote {
                        println!("remote:   {r}");
                    }
                    println!("stack:    {}", p.stack);
                    println!("aliases:  {}", p.aliases.join(", "));
                    println!(
                        "summary:  {}{}",
                        if p.summary.is_empty() { "(none)" } else { &p.summary },
                        if p.summary_by.is_empty() { String::new() } else { format!("  [by {}]", p.summary_by) }
                    );
                    println!("memories: {}", e.store.count_memories().get(&scope_of(&p.pid)).copied().unwrap_or(0));
                }
                None => println!("no project detected for {}", ctx.cwd.unwrap_or_default()),
            }
            Ok(())
        }
        Cmd::Stats { json } => {
            let e = open(&cfg)?;
            let counts = e.store.count_memories();
            let total: i64 = counts.values().sum();
            let projects = e.store.list_projects().len();
            let episodes = e.store.episode_count();
            if json {
                pj(&json!({"memories": total, "by_scope": counts, "projects": projects, "episodes": episodes,
                    "decider": e.decider.describe(), "embedder": e.embedder.id(), "db": cfg.db_path()}));
                return Ok(());
            }
            println!("db:        {}", cfg.db_path().display());
            println!("memories:  {total} live");
            let mut cs: Vec<_> = counts.into_iter().collect();
            cs.sort_by_key(|a| std::cmp::Reverse(a.1));
            for (s, n) in cs {
                println!("  {:>6}  {}", n, scope_label(&e, &s));
            }
            println!("projects:  {projects}");
            println!("episodes:  {episodes}");
            println!("decider:   {}", e.decider.describe());
            println!("embedder:  {}", e.embedder.id());
            println!(
                "distill:   {}",
                if !cfg.distill.enabled {
                    "off".to_string()
                } else if cfg.distill.model.is_empty() {
                    "heuristic".into()
                } else {
                    cfg.distill.model.clone()
                }
            );
            Ok(())
        }
        Cmd::Trace { n } => {
            let e = open(&cfg)?;
            let mut ts = e.store.recent_traces(n);
            ts.reverse();
            for (t, kind, data) in ts {
                let when = time::OffsetDateTime::from_unix_timestamp(t)
                    .ok()
                    .and_then(|d| d.format(&time::macros::format_description!("[year]-[month]-[day] [hour]:[minute]:[second]")).ok())
                    .unwrap_or_default();
                println!("{when}  {kind:<12} {data}");
            }
            Ok(())
        }
        Cmd::Models { cmd } => match cmd {
            ModelsCmd::Download { name, dir } => models_download(&cfg, &name, dir).await,
            ModelsCmd::List => {
                let d = cfg.data_dir().join("models").join(unimem::embed::E5_SMALL.dir);
                println!(
                    "e5-small  intfloat/multilingual-e5-small (118M, 384-d, multilingual)  {}",
                    if d.join("model.onnx").exists() { format!("installed: {}", d.display()) } else { "not installed".into() }
                );
                println!("onnx feature in this build: {}", cfg!(feature = "onnx"));
                Ok(())
            }
        },
        Cmd::Reembed => {
            let e = open(&cfg)?;
            let n = e.reembed().await?;
            println!("re-embedded {n} memories with {}", e.embedder.id());
            Ok(())
        }
        Cmd::Distill { now } => {
            let e = open(&cfg)?;
            if !cfg.distill.enabled && (cfg.distill.base_url.is_empty() || cfg.distill.model.is_empty()) {
                println!(
                    "distiller not configured: set [distill] enabled = true (heuristic) or base_url + model in {}",
                    cli.config.unwrap_or_else(Config::default_path).display()
                );
                return Ok(());
            }
            let n = if now { unimem::distill::run_with_idle(&e, 0).await? } else { unimem::distill::run_once(&e).await? };
            println!("distilled {n} new memories");
            Ok(())
        }
        Cmd::Config => {
            let p = cli.config.unwrap_or_else(Config::default_path);
            println!("# config path: {} ({})", p.display(), if p.exists() { "exists" } else { "not found; defaults in use" });
            println!("# data dir:    {}\n", cfg.data_dir().display());
            print!("{SAMPLE_CONFIG}");
            Ok(())
        }
        Cmd::Setup { harness } => match setup_text(&harness, &cfg.listen) {
            Some(t) => {
                print!("{t}");
                Ok(())
            }
            None => anyhow::bail!("unknown harness '{harness}' (try: claude-code, codex, shell, mcp)"),
        },
        Cmd::Ui { no_open } => {
            let url = format!("http://{}/ui", cfg.listen);
            let up =
                tokio::time::timeout(Duration::from_secs(2), tokio::net::TcpStream::connect(&cfg.listen)).await.is_ok_and(|r| r.is_ok());
            println!("{url}");
            if !up {
                eprintln!("(the gateway is not running on {}; start it with `unimem serve`)", cfg.listen);
            }
            if !no_open {
                let opener = if cfg!(target_os = "macos") {
                    "open"
                } else if cfg!(windows) {
                    "explorer"
                } else {
                    "xdg-open"
                };
                let _ = std::process::Command::new(opener)
                    .arg(&url)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }
            Ok(())
        }
        Cmd::DemoSeed { dir } => {
            let e = open(&cfg)?;
            let dir = dir.unwrap_or_else(|| cfg.data_dir().join("demo-repos"));
            let n = gateway::ui::demo_seed(&e, &dir).await?;
            println!("seeded {n} demo memories across {} projects + global (repos in {})", e.store.list_projects().len(), dir.display());
            println!("open the UI: unimem serve, then http://{}/ui", cfg.listen);
            Ok(())
        }
    }
}

async fn serve(mut cfg: Config, listen: Option<String>) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("unimem=info")),
        )
        .init();
    if let Some(l) = listen {
        cfg.listen = l;
    }
    let engine = Arc::new(open(&cfg)?);
    let client = gateway::upstream_client(cfg.upstream.timeout_secs)?;
    let state = Arc::new(AppState { engine: engine.clone(), client });

    unimem::distill::spawn(engine.clone());
    {
        let e = engine.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(6 * 3600));
            loop {
                tick.tick().await;
                match e.store.gc_overlays(30 * 86400) {
                    Ok(0) => {}
                    Ok(n) => tracing::info!("gc: removed {n} overlays older than 30 days"),
                    Err(err) => tracing::warn!("overlay gc failed: {err}"),
                }
                let _ = e.store.gc_session_state(30 * 86400);
                let ttl = e.cfg.distill.episode_ttl_days as i64 * 86400;
                if ttl > 0 {
                    match e.store.gc_episodes(ttl) {
                        Ok(0) => {}
                        Ok(n) => tracing::info!("gc: removed {n} episodes older than {} days", e.cfg.distill.episode_ttl_days),
                        Err(err) => tracing::warn!("episode gc failed: {err}"),
                    }
                }
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&cfg.listen).await.with_context(|| format!("binding {}", cfg.listen))?;
    let addr = listener.local_addr()?;
    println!("unimem {} listening on http://{addr}", env!("CARGO_PKG_VERSION"));
    println!("  db:       {}", cfg.db_path().display());
    println!("  decider:  {}", engine.decider.describe());
    println!("  embedder: {}", engine.embedder.id());
    println!(
        "  distill:  {}",
        if !cfg.distill.enabled {
            "off".to_string()
        } else if cfg.distill.model.is_empty() {
            "heuristic".into()
        } else {
            cfg.distill.model.clone()
        }
    );
    println!("  inject:   {}", if cfg.inject.enabled { "on" } else { "off (pass-through)" });
    println!("setup:");
    println!("  Claude Code:  export ANTHROPIC_BASE_URL=http://{addr}");
    println!("  Codex:        unimem setup codex   (model_provider with base_url http://{addr}/v1, wire_api = \"responses\")");
    println!("  MCP only:     http://{addr}/mcp");
    println!("web UI:         http://{addr}/ui");

    gateway::serve_with_shutdown(listener, gateway::router(state), async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}
