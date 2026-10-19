//! Project identity. Hard signals are deterministic (cwd → git root → remote → manifest → README);
//! the semantic layer (what the project *is*, which scope a memory belongs to) is decided by models:
//! the in-loop agent can rewrite the project card via `unimem_project`, System One picks write scopes.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::store::{Project, Store};
use crate::util::{now_ts, truncate_chars};

#[derive(Debug, Clone, Default)]
pub struct GitInfo {
    pub root: PathBuf,
    pub remote: Option<String>,
    pub branch: String,
    pub commit: String,
}

/// Find the repository root. Handles worktrees (`.git` file → `gitdir:` → commondir) so all worktrees map to
/// one project.
pub fn git_info(cwd: &Path) -> Option<GitInfo> {
    let mut d = Some(cwd);
    while let Some(dir) = d {
        let dotgit = dir.join(".git");
        if dotgit.exists() {
            let gitdir = if dotgit.is_file() {
                let s = std::fs::read_to_string(&dotgit).ok()?;
                let p = s.trim().strip_prefix("gitdir:")?.trim().to_string();
                let p = if Path::new(&p).is_absolute() { PathBuf::from(p) } else { dir.join(p) };
                // worktree: <common>/.git/worktrees/<name>; commondir file points to the shared dir
                match std::fs::read_to_string(p.join("commondir")) {
                    Ok(c) => {
                        let c = c.trim();
                        if Path::new(c).is_absolute() {
                            PathBuf::from(c)
                        } else {
                            p.join(c)
                        }
                    }
                    Err(_) => p.clone(),
                }
                .canonicalize()
                .unwrap_or(p)
            } else {
                dotgit.clone()
            };
            let head_dir = if dotgit.is_file() {
                std::fs::read_to_string(&dotgit)
                    .ok()
                    .and_then(|s| s.trim().strip_prefix("gitdir:").map(|x| x.trim().to_string()))
                    .map(|p| if Path::new(&p).is_absolute() { PathBuf::from(p) } else { dir.join(p) })
                    .unwrap_or_else(|| gitdir.clone())
            } else {
                gitdir.clone()
            };
            let head = std::fs::read_to_string(head_dir.join("HEAD")).unwrap_or_default();
            let head = head.trim();
            let (branch, commit) = if let Some(r) = head.strip_prefix("ref: ") {
                let b = r.trim_start_matches("refs/heads/").to_string();
                let c = std::fs::read_to_string(gitdir.join(r)).map(|s| s.trim().to_string()).unwrap_or_else(|_| packed_ref(&gitdir, r));
                (b, c)
            } else {
                ("HEAD".into(), head.to_string())
            };
            // Root of the *main* worktree = parent of the common .git dir.
            let root = if gitdir.file_name().is_some_and(|n| n == ".git") {
                gitdir.parent().map(Path::to_path_buf).unwrap_or_else(|| dir.to_path_buf())
            } else {
                dir.to_path_buf()
            };
            return Some(GitInfo { root, remote: origin_url(&gitdir), branch, commit: commit.chars().take(12).collect() });
        }
        d = dir.parent();
    }
    None
}

fn packed_ref(gitdir: &Path, r: &str) -> String {
    std::fs::read_to_string(gitdir.join("packed-refs"))
        .ok()
        .and_then(|s| s.lines().find(|l| l.ends_with(r)).map(|l| l.split(' ').next().unwrap_or("").to_string()))
        .unwrap_or_default()
}

fn origin_url(gitdir: &Path) -> Option<String> {
    let cfg = std::fs::read_to_string(gitdir.join("config")).ok()?;
    let mut in_remote: Option<String> = None;
    let mut urls: Vec<(String, String)> = vec![];
    for line in cfg.lines() {
        let l = line.trim();
        if l.starts_with('[') {
            in_remote = l.strip_prefix("[remote \"").and_then(|x| x.strip_suffix("\"]")).map(String::from);
        } else if let (Some(r), Some(u)) = (&in_remote, l.strip_prefix("url")) {
            if let Some(u) = u.trim().strip_prefix('=') {
                urls.push((r.clone(), u.trim().to_string()));
            }
        }
    }
    urls.iter().find(|(r, _)| r == "origin").or(urls.first()).map(|(_, u)| normalize_remote(u))
}

/// `git@github.com:Foo/Bar.git`, `https://user@github.com/foo/bar` → `github.com/foo/bar`
pub fn normalize_remote(u: &str) -> String {
    let mut s = u.trim().to_string();
    for p in ["https://", "http://", "ssh://", "git://", "git+ssh://"] {
        if let Some(r) = s.strip_prefix(p) {
            s = r.to_string();
        }
    }
    if let Some(at) = s.find('@') {
        if !s[..at].contains('/') {
            s = s[at + 1..].to_string();
        }
    }
    if let Some(colon) = s.find(':') {
        if !s[..colon].contains('/') {
            let (h, rest) = s.split_at(colon);
            let rest = rest[1..].trim_start_matches(|c: char| c.is_ascii_digit()).trim_start_matches('/');
            s = format!("{h}/{rest}");
        }
    }
    s.trim_end_matches('/').trim_end_matches(".git").to_lowercase()
}

/// Files whose presence marks a directory as a project root.
pub const MANIFESTS: &[&str] = &[
    "Cargo.toml", "package.json", "pyproject.toml", "setup.py", "setup.cfg", "go.mod", "Package.swift", "pom.xml", "build.gradle",
    "build.gradle.kts", "Gemfile", "composer.json", "CMakeLists.txt", "mix.exs", "deno.json", "pubspec.yaml", "Makefile", "flake.nix",
];

/// Nearest directory at or above `cwd` that contains a project manifest. The walk stops at (and never returns)
/// the home directory or the filesystem root, so a stray `~/package.json` does not turn every folder into one project.
pub fn manifest_root(cwd: &Path, home: Option<&Path>) -> Option<PathBuf> {
    let mut d = Some(cwd);
    while let Some(dir) = d {
        if Some(dir) == home || dir.parent().is_none() {
            return None;
        }
        if MANIFESTS.iter().any(|m| dir.join(m).is_file()) {
            return Some(dir.to_path_buf());
        }
        d = dir.parent();
    }
    None
}

#[derive(Debug, Clone, Default)]
pub struct Manifest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub stack: Vec<String>,
    pub readme: Option<String>,
}

pub fn read_manifest(root: &Path) -> Manifest {
    let mut m = Manifest::default();
    let rd = |f: &str| std::fs::read_to_string(root.join(f)).ok();
    if let Some(s) = rd("Cargo.toml") {
        m.stack.push("rust".into());
        if let Ok(v) = s.parse::<toml::Table>() {
            let pkg = v.get("package").or_else(|| v.get("workspace").and_then(|w| w.get("package")));
            m.name = pkg.and_then(|p| p.get("name")).and_then(|x| x.as_str()).map(String::from);
            m.description = pkg.and_then(|p| p.get("description")).and_then(|x| x.as_str()).map(String::from);
        }
    }
    if let Some(s) = rd("package.json") {
        m.stack.push("node".into());
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            m.name = m.name.or(v["name"].as_str().map(String::from));
            m.description = m.description.or(v["description"].as_str().map(String::from));
        }
    }
    if let Some(s) = rd("pyproject.toml") {
        m.stack.push("python".into());
        if let Ok(v) = s.parse::<toml::Table>() {
            let p = v.get("project").or_else(|| v.get("tool").and_then(|t| t.get("poetry")));
            m.name = m.name.or(p.and_then(|p| p.get("name")).and_then(|x| x.as_str()).map(String::from));
            m.description = m.description.or(p.and_then(|p| p.get("description")).and_then(|x| x.as_str()).map(String::from));
        }
    }
    for (f, t) in [("go.mod", "go"), ("Package.swift", "swift"), ("pom.xml", "java"), ("build.gradle", "jvm"), ("Gemfile", "ruby"), ("composer.json", "php"), ("CMakeLists.txt", "c++")] {
        if root.join(f).exists() {
            m.stack.push(t.into());
        }
    }
    for f in ["README.md", "README", "readme.md", "README.zh.md", "AGENTS.md", "CLAUDE.md"] {
        if let Some(s) = rd(f) {
            m.readme = Some(readme_lead(&s));
            if m.readme.as_deref().is_some_and(|r| !r.is_empty()) {
                break;
            }
        }
    }
    m
}

/// First meaningful paragraph of a README (skip headings, badges, html).
fn readme_lead(s: &str) -> String {
    let mut para = String::new();
    for line in s.lines() {
        let l = line.trim();
        if l.is_empty() {
            if para.len() > 40 {
                break;
            }
            continue;
        }
        if l.starts_with('#') || l.starts_with('<') || l.starts_with("[![") || l.starts_with("![") || l.starts_with("```") || l.starts_with('|') || l.starts_with("---") {
            if para.len() > 40 {
                break;
            }
            continue;
        }
        let l = l.trim_start_matches('>').trim();
        if !para.is_empty() {
            para.push(' ');
        }
        para.push_str(l);
        if para.len() > 400 {
            break;
        }
    }
    truncate_chars(&para.replace("**", ""), 300)
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub project: Project,
    pub git: Option<GitInfo>,
    pub is_new: bool,
}

pub struct Resolver {
    cache: Mutex<HashMap<PathBuf, (i64, Option<Resolved>)>>,
}

impl Default for Resolver {
    fn default() -> Self {
        Self { cache: Mutex::new(HashMap::new()) }
    }
}

pub fn pid_for(root: &Path, remote: &Option<String>) -> String {
    match remote {
        Some(r) => r.clone(),
        None => format!("local{}", root.display()),
    }
}

impl Resolver {
    /// Resolve the project for a working directory. Home dir and `/` are treated as "no project".
    pub fn resolve(&self, store: &Store, cwd: &str) -> Option<Resolved> {
        // Resolve symlinks first so every path spelling of a directory maps to one project.
        let cwd = PathBuf::from(cwd);
        let cwd = cwd.canonicalize().unwrap_or(cwd);
        let now = now_ts();
        if let Some((t, r)) = self.cache.lock().get(&cwd) {
            if now - t < 60 {
                return r.clone().map(|mut r| {
                    r.is_new = false;
                    r
                });
            }
        }
        let r = self.resolve_uncached(store, &cwd);
        self.cache.lock().insert(cwd, (now, r.clone()));
        r
    }

    fn resolve_uncached(&self, store: &Store, cwd: &Path) -> Option<Resolved> {
        let home = dirs::home_dir().map(|h| h.canonicalize().unwrap_or(h));
        let git = git_info(cwd);
        let root = match &git {
            Some(g) => g.root.clone(),
            None => manifest_root(cwd, home.as_deref()).unwrap_or_else(|| cwd.to_path_buf()),
        };
        if Some(&root) == home.as_ref() || root == Path::new("/") || root.as_os_str().is_empty() {
            return None;
        }
        let remote = git.as_ref().and_then(|g| g.remote.clone());
        let pid = pid_for(&root, &remote);
        // Path-based project later gained a remote → keep the old pid as the identity.
        let legacy = format!("local{}", root.display());
        let existing = store.get_project(&pid).or_else(|| if remote.is_some() { store.get_project(&legacy) } else { None });
        if let Some(mut p) = existing {
            let rs = root.display().to_string();
            if p.root != rs {
                if !p.aliases.contains(&p.root) {
                    p.aliases.push(p.root.clone());
                }
                p.root = rs;
                p.updated_at = now_ts();
                let _ = store.upsert_project(&p);
            }
            return Some(Resolved { project: p, git, is_new: false });
        }
        // Without a git repo, only treat directories that look like projects as projects.
        let man = read_manifest(&root);
        if git.is_none() && man.stack.is_empty() && man.readme.is_none() {
            return None;
        }
        let dir_name = root.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let name = remote
            .as_ref()
            .and_then(|r| r.rsplit('/').next().map(String::from))
            .or(man.name.clone())
            .unwrap_or(dir_name.clone());
        let mut aliases = vec![];
        for a in [Some(dir_name), man.name.clone()].into_iter().flatten() {
            if !a.eq_ignore_ascii_case(&name) && !aliases.contains(&a) {
                aliases.push(a);
            }
        }
        let summary = man.description.clone().or(man.readme.clone()).unwrap_or_default();
        let ts = now_ts();
        let p = Project {
            pid,
            root: root.display().to_string(),
            remote,
            name,
            summary: truncate_chars(&summary, 300),
            stack: man.stack.join(", "),
            aliases,
            summary_by: if summary.is_empty() { String::new() } else { "manifest".into() },
            created_at: ts,
            updated_at: ts,
        };
        let _ = store.upsert_project(&p);
        Some(Resolved { project: p, git, is_new: true })
    }

    pub fn invalidate(&self) {
        self.cache.lock().clear();
    }
}

/// Find projects mentioned by name/alias in free text (cross-project references).
pub fn mentioned_projects(store: &Store, text: &str, exclude: Option<&str>) -> Vec<Project> {
    let lower = text.to_lowercase();
    store
        .list_projects()
        .into_iter()
        .filter(|p| Some(p.pid.as_str()) != exclude)
        .filter(|p| {
            std::iter::once(&p.name).chain(p.aliases.iter()).any(|n| {
                let n = n.to_lowercase();
                n.chars().count() >= 3 && contains_word(&lower, &n)
            })
        })
        .collect()
}

/// Projects referred to by a *short form* of a compound name: the leading part of `brine-pipeline` / `harbor_infra`
/// ("brine", "harbor") used as a word. Weaker than a full-name mention, so callers use it only when no project is
/// named in full; parts shorter than 4 characters are ignored.
pub fn short_mentioned_projects(store: &Store, text: &str, exclude: Option<&str>) -> Vec<Project> {
    let lower = text.to_lowercase();
    store
        .list_projects()
        .into_iter()
        .filter(|p| Some(p.pid.as_str()) != exclude)
        .filter(|p| {
            std::iter::once(&p.name).chain(p.aliases.iter()).any(|n| {
                let n = n.to_lowercase();
                let mut parts = n.split(['-', '_', ' ']).filter(|x| !x.is_empty());
                match (parts.next(), parts.next()) {
                    (Some(head), Some(_)) => head.chars().count() >= 4 && !head.starts_with('/') && contains_word(&lower, head),
                    _ => false,
                }
            })
        })
        .collect()
}

/// Does `text` explicitly name `p`: its name or an alias as a word, its root path, or a relative path
/// (from the extracted `entities`) that exists inside its root?
pub fn names_project(p: &Project, text: &str, entities: &[String]) -> bool {
    let lower = text.to_lowercase();
    let by_name = std::iter::once(&p.name).chain(p.aliases.iter()).any(|n| {
        let n = n.to_lowercase();
        n.chars().count() >= 3 && !n.starts_with('/') && contains_word(&lower, &n)
    });
    if by_name || (!p.root.is_empty() && text.contains(&p.root)) {
        return true;
    }
    let root = Path::new(&p.root);
    entities.iter().any(|e| {
        let e = e.trim_start_matches("./");
        (e.contains('/') || e.contains('.')) && !e.contains(' ') && !Path::new(e).is_absolute() && !p.root.is_empty() && root.join(e).exists()
    })
}

fn contains_word(hay: &str, needle: &str) -> bool {
    let mut start = 0;
    while let Some(i) = hay[start..].find(needle) {
        let a = start + i;
        let b = a + needle.len();
        let before = hay[..a].chars().next_back();
        let after = hay[b..].chars().next();
        let ok = |c: Option<char>| c.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'));
        if ok(before) && ok(after) {
            return true;
        }
        start = b;
    }
    false
}

pub fn scope_of(pid: &str) -> String {
    format!("project:{pid}")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_remotes() {
        assert_eq!(normalize_remote("git@github.com:Mapix/Readit.git"), "github.com/mapix/readit");
        assert_eq!(normalize_remote("https://tok@github.com/mapix/readit/"), "github.com/mapix/readit");
        assert_eq!(normalize_remote("ssh://git@gitlab.com:2222/a/b.git"), "gitlab.com/a/b");
    }
    #[test]
    fn word_match() {
        assert!(contains_word("how does readlet deploy?", "readlet"));
        assert!(!contains_word("readlets", "readlet"));
    }
    #[test]
    fn resolves_git_project() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(root.join(".git/config"), "[remote \"origin\"]\n\turl = git@github.com:me/proj.git\n").unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname=\"proj\"\ndescription=\"A thing that does stuff\"\n").unwrap();
        std::fs::create_dir_all(root.join("src/deep")).unwrap();
        let store = Store::open_memory().unwrap();
        let r = Resolver::default().resolve(&store, root.join("src/deep").to_str().unwrap()).unwrap();
        assert_eq!(r.project.pid, "github.com/me/proj");
        assert_eq!(r.project.summary, "A thing that does stuff");
        assert!(r.project.stack.contains("rust"));
        assert_eq!(r.git.unwrap().branch, "main");
    }

    #[test]
    fn non_git_subdirectory_walks_up_to_manifest() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("notes-tool");
        std::fs::create_dir_all(root.join("lib/parsers/inner")).unwrap();
        std::fs::write(root.join("pyproject.toml"), "[project]\nname = \"notes-tool\"\ndescription = \"Parses notes\"\n").unwrap();
        let store = Store::open_memory().unwrap();
        let res = Resolver::default();
        let deep = res.resolve(&store, root.join("lib/parsers/inner").to_str().unwrap()).unwrap();
        let top = res.resolve(&store, root.to_str().unwrap()).unwrap();
        assert_eq!(deep.project.pid, top.project.pid);
        assert_eq!(deep.project.name, "notes-tool");
        // a plain directory with no manifest anywhere above is still not a project
        let plain = d.path().join("scratch/a");
        std::fs::create_dir_all(&plain).unwrap();
        assert!(res.resolve(&store, plain.to_str().unwrap()).is_none());
    }

    #[test]
    fn walk_up_stops_at_home() {
        let d = tempfile::tempdir().unwrap();
        let home = d.path().join("home");
        std::fs::create_dir_all(home.join("Downloads/x")).unwrap();
        std::fs::write(home.join("package.json"), "{}").unwrap();
        assert_eq!(manifest_root(&home.join("Downloads/x"), Some(&home)), None);
        std::fs::write(home.join("Downloads/package.json"), "{}").unwrap();
        assert_eq!(manifest_root(&home.join("Downloads/x"), Some(&home)), Some(home.join("Downloads")));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_path_maps_to_same_project() {
        let d = tempfile::tempdir().unwrap();
        let real = d.path().join("real/widget");
        std::fs::create_dir_all(real.join(".git")).unwrap();
        std::fs::write(real.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::create_dir_all(real.join("src")).unwrap();
        let link = d.path().join("link-to-widget");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let store = Store::open_memory().unwrap();
        let res = Resolver::default();
        let a = res.resolve(&store, real.join("src").to_str().unwrap()).unwrap();
        let b = res.resolve(&store, link.join("src").to_str().unwrap()).unwrap();
        assert_eq!(a.project.pid, b.project.pid);
        assert_eq!(store.list_projects().len(), 1);
    }
}
