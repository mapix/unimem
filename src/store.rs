//! SQLite single-file store: memories, graph edges, projects, episodes, overlays, sessions.
//! FTS uses a pre-tokenized column (our own CJK-aware tokenizer), vectors live in a BLOB column
//! and are mirrored into an in-RAM index for exact cosine search.

use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::util::{now_ts, tokenize};

mod session;
pub use session::{LedgerEntry, SessionState};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Source {
    pub harness: String,
    pub session: String,
    pub cwd: String,
    pub branch: String,
    pub commit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub scope: String,
    pub kind: String,
    pub content: String,
    pub why: Option<String>,
    pub entities: Vec<String>,
    pub source: Source,
    pub created_at: i64,
    pub updated_at: i64,
    pub superseded_by: Option<String>,
    pub salience: f64,
    pub hits: i64,
    pub last_hit_at: Option<i64>,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub src: String,
    pub dst: String,
    pub rel: String,
    pub weight: f64,
    pub judged_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Project {
    pub pid: String,
    pub root: String,
    pub remote: Option<String>,
    pub name: String,
    pub summary: String,
    pub stack: String,
    pub aliases: Vec<String>,
    pub summary_by: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct Episode {
    pub id: i64,
    pub session: String,
    pub harness: String,
    pub pid: Option<String>,
    pub role: String,
    pub text: String,
    pub ts: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRow {
    pub key: String,
    pub pid: Option<String>,
    pub harness: String,
    pub cwd: Option<String>,
    pub first_seen: i64,
    pub last_seen: i64,
}

struct VecEntry {
    scope: String,
    live: bool,
    v: Vec<f32>,
}

/// In-RAM lexical index over live memories (BM25 with weighted query terms, see `lexicon`).
#[derive(Default)]
struct LexIndex {
    /// doc id → (scope, term frequencies, length)
    docs: HashMap<String, (String, HashMap<String, f32>, f32)>,
    /// term → doc ids containing it
    post: HashMap<String, Vec<String>>,
    total_len: f64,
}

impl LexIndex {
    fn add(&mut self, id: &str, scope: &str, text: &str) {
        self.remove(id);
        let mut tf: HashMap<String, f32> = HashMap::new();
        let terms = crate::lexicon::index_terms(text);
        for t in &terms {
            *tf.entry(t.clone()).or_default() += 1.0;
        }
        let len = terms.len() as f32;
        for t in tf.keys() {
            self.post.entry(t.clone()).or_default().push(id.to_string());
        }
        self.total_len += len as f64;
        self.docs.insert(id.to_string(), (scope.to_string(), tf, len));
    }
    fn remove(&mut self, id: &str) {
        if let Some((_, tf, len)) = self.docs.remove(id) {
            self.total_len -= len as f64;
            for t in tf.keys() {
                if let Some(v) = self.post.get_mut(t) {
                    v.retain(|x| x != id);
                    if v.is_empty() {
                        self.post.remove(t);
                    }
                }
            }
        }
    }
    fn set_scope(&mut self, id: &str, scope: &str) {
        if let Some(d) = self.docs.get_mut(id) {
            d.0 = scope.to_string();
        }
    }
}

/// One lexical hit: BM25 score, and the idf-weighted share of the query that the note covers (0..1).
#[derive(Debug, Clone)]
pub struct LexHit {
    pub id: String,
    pub score: f64,
    pub coverage: f64,
}

pub struct Store {
    conn: Mutex<Connection>,
    vecs: RwLock<HashMap<String, VecEntry>>,
    lex: RwLock<LexIndex>,
    /// Write-through cache of `session_state` rows (read on every gateway request).
    sstate: Mutex<HashMap<String, SessionState>>,
}

const SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, scope TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL, why TEXT,
  entities TEXT NOT NULL DEFAULT '[]', source TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, superseded_by TEXT,
  salience REAL NOT NULL DEFAULT 0.5, hits INTEGER NOT NULL DEFAULT 0, last_hit_at INTEGER,
  pinned INTEGER NOT NULL DEFAULT 0, deleted INTEGER NOT NULL DEFAULT 0, embedding BLOB
);
CREATE INDEX IF NOT EXISTS mem_scope ON memories(scope, deleted);
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(id UNINDEXED, tok);
CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL, weight REAL NOT NULL, judged_by TEXT NOT NULL,
  created_at INTEGER NOT NULL, PRIMARY KEY (src, dst, rel)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS projects (
  pid TEXT PRIMARY KEY, root TEXT NOT NULL, remote TEXT, name TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
  stack TEXT NOT NULL DEFAULT '', aliases TEXT NOT NULL DEFAULT '[]', summary_by TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, harness TEXT NOT NULL, pid TEXT,
  role TEXT NOT NULL, text TEXT NOT NULL, ts INTEGER NOT NULL, distilled INTEGER NOT NULL DEFAULT 0,
  hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ep_session ON episodes(session, ts);
CREATE TABLE IF NOT EXISTS overlays (
  anchor TEXT PRIMARY KEY, session TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  key TEXT PRIMARY KEY, pid TEXT, harness TEXT NOT NULL, cwd TEXT, first_seen INTEGER NOT NULL, last_seen INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_state (
  session TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at INTEGER NOT NULL
);
"#;

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        d += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

const MEM_COLS: &str =
    "id, scope, kind, content, why, entities, source, created_at, updated_at, superseded_by, salience, hits, last_hit_at, pinned";

fn row_to_memory(r: &rusqlite::Row) -> rusqlite::Result<Memory> {
    let entities: String = r.get(5)?;
    let source: String = r.get(6)?;
    Ok(Memory {
        id: r.get(0)?,
        scope: r.get(1)?,
        kind: r.get(2)?,
        content: r.get(3)?,
        why: r.get(4)?,
        entities: serde_json::from_str(&entities).unwrap_or_default(),
        source: serde_json::from_str(&source).unwrap_or_default(),
        created_at: r.get(7)?,
        updated_at: r.get(8)?,
        superseded_by: r.get(9)?,
        salience: r.get(10)?,
        hits: r.get(11)?,
        last_hit_at: r.get(12)?,
        pinned: r.get::<_, i64>(13)? != 0,
    })
}

fn row_to_project(r: &rusqlite::Row) -> rusqlite::Result<Project> {
    let aliases: String = r.get(6)?;
    Ok(Project {
        pid: r.get(0)?,
        root: r.get(1)?,
        remote: r.get(2)?,
        name: r.get(3)?,
        summary: r.get(4)?,
        stack: r.get(5)?,
        aliases: serde_json::from_str(&aliases).unwrap_or_default(),
        summary_by: r.get(7)?,
        created_at: r.get(8)?,
        updated_at: r.get(9)?,
    })
}
const PROJ_COLS: &str = "pid, root, remote, name, summary, stack, aliases, summary_by, created_at, updated_at";

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(SCHEMA)?;
        session::migrate(&conn);
        let s = Store { conn: Mutex::new(conn), vecs: RwLock::new(HashMap::new()), lex: RwLock::new(LexIndex::default()), sstate: Mutex::new(HashMap::new()) };
        s.load_vectors()?;
        s.load_lexical()?;
        Ok(s)
    }

    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        session::migrate(&conn);
        Ok(Store { conn: Mutex::new(conn), vecs: RwLock::new(HashMap::new()), lex: RwLock::new(LexIndex::default()), sstate: Mutex::new(HashMap::new()) })
    }

    fn load_vectors(&self) -> Result<()> {
        let c = self.conn.lock();
        let mut st = c.prepare("SELECT id, scope, superseded_by IS NULL, embedding FROM memories WHERE deleted=0 AND embedding IS NOT NULL")?;
        let rows = st.query_map([], |r| {
            let b: Vec<u8> = r.get(3)?;
            Ok((r.get::<_, String>(0)?, VecEntry { scope: r.get(1)?, live: r.get(2)?, v: blob_to_vec(&b) }))
        })?;
        let mut m = self.vecs.write();
        for row in rows {
            let (id, e) = row?;
            m.insert(id, e);
        }
        Ok(())
    }

    fn load_lexical(&self) -> Result<()> {
        let c = self.conn.lock();
        let mut st = c.prepare(&format!("SELECT {MEM_COLS} FROM memories WHERE deleted=0 AND superseded_by IS NULL"))?;
        let rows: Vec<Memory> = st.query_map([], row_to_memory)?.filter_map(|r| r.ok()).collect();
        let mut lx = self.lex.write();
        for m in rows {
            lx.add(&m.id, &m.scope, &Self::lex_text(&m));
        }
        Ok(())
    }

    fn lex_text(m: &Memory) -> String {
        format!("{} {} {}", m.content, m.why.clone().unwrap_or_default(), m.entities.join(" "))
    }

    // ---------- meta ----------
    pub fn meta_get(&self, k: &str) -> Option<String> {
        self.conn.lock().query_row("SELECT v FROM meta WHERE k=?", [k], |r| r.get(0)).optional().ok().flatten()
    }
    pub fn meta_set(&self, k: &str, v: &str) -> Result<()> {
        self.conn.lock().execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", [k, v])?;
        Ok(())
    }

    // ---------- memories ----------
    pub fn insert_memory(&self, m: &Memory, emb: &[f32]) -> Result<()> {
        let c = self.conn.lock();
        c.execute(
            &format!("INSERT INTO memories ({MEM_COLS}, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
            params![
                m.id,
                m.scope,
                m.kind,
                m.content,
                m.why,
                serde_json::to_string(&m.entities)?,
                serde_json::to_string(&m.source)?,
                m.created_at,
                m.updated_at,
                m.superseded_by,
                m.salience,
                m.hits,
                m.last_hit_at,
                m.pinned as i64,
                vec_to_blob(emb)
            ],
        )?;
        let tok = Self::fts_text(m);
        c.execute("INSERT INTO mem_fts(id, tok) VALUES (?,?)", params![m.id, tok])?;
        drop(c);
        self.vecs.write().insert(m.id.clone(), VecEntry { scope: m.scope.clone(), live: m.superseded_by.is_none(), v: emb.to_vec() });
        if m.superseded_by.is_none() {
            self.lex.write().add(&m.id, &m.scope, &Self::lex_text(m));
        }
        Ok(())
    }

    fn fts_text(m: &Memory) -> String {
        let mut s = m.content.clone();
        if let Some(w) = &m.why {
            s.push(' ');
            s.push_str(w);
        }
        for e in &m.entities {
            s.push(' ');
            s.push_str(e);
        }
        tokenize(&s).join(" ")
    }

    pub fn get_memory(&self, id: &str) -> Option<Memory> {
        self.conn
            .lock()
            .query_row(&format!("SELECT {MEM_COLS} FROM memories WHERE id=? AND deleted=0"), [id], row_to_memory)
            .optional()
            .ok()
            .flatten()
    }

    pub fn get_memories(&self, ids: &[String]) -> Vec<Memory> {
        ids.iter().filter_map(|i| self.get_memory(i)).collect()
    }

    pub fn update_content(&self, id: &str, content: &str, why: Option<&str>, emb: &[f32]) -> Result<()> {
        let c = self.conn.lock();
        c.execute(
            "UPDATE memories SET content=?, why=?, updated_at=?, embedding=? WHERE id=?",
            params![content, why, now_ts(), vec_to_blob(emb), id],
        )?;
        let m = c.query_row(&format!("SELECT {MEM_COLS} FROM memories WHERE id=?"), [id], row_to_memory)?;
        c.execute("DELETE FROM mem_fts WHERE id=?", [id])?;
        c.execute("INSERT INTO mem_fts(id, tok) VALUES (?,?)", params![id, Self::fts_text(&m)])?;
        drop(c);
        if let Some(e) = self.vecs.write().get_mut(id) {
            e.v = emb.to_vec();
        }
        if m.superseded_by.is_none() {
            self.lex.write().add(id, &m.scope, &Self::lex_text(&m));
        }
        Ok(())
    }

    pub fn set_scope(&self, id: &str, scope: &str) -> Result<()> {
        self.conn.lock().execute("UPDATE memories SET scope=?, updated_at=? WHERE id=?", params![scope, now_ts(), id])?;
        if let Some(e) = self.vecs.write().get_mut(id) {
            e.scope = scope.to_string();
        }
        self.lex.write().set_scope(id, scope);
        Ok(())
    }

    pub fn supersede(&self, old: &str, new: &str) -> Result<()> {
        self.conn.lock().execute("UPDATE memories SET superseded_by=?, updated_at=? WHERE id=?", params![new, now_ts(), old])?;
        if let Some(e) = self.vecs.write().get_mut(old) {
            e.live = false;
        }
        self.lex.write().remove(old);
        Ok(())
    }

    pub fn set_pinned(&self, id: &str, pinned: bool) -> Result<()> {
        self.conn.lock().execute("UPDATE memories SET pinned=? WHERE id=?", params![pinned as i64, id])?;
        Ok(())
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool> {
        let c = self.conn.lock();
        let n = c.execute("UPDATE memories SET deleted=1 WHERE id=? AND deleted=0", [id])?;
        c.execute("DELETE FROM mem_fts WHERE id=?", [id])?;
        c.execute("DELETE FROM edges WHERE src=? OR dst=?", [id, id])?;
        drop(c);
        self.vecs.write().remove(id);
        self.lex.write().remove(id);
        Ok(n > 0)
    }

    pub fn touch_hits(&self, ids: &[String]) {
        let c = self.conn.lock();
        let ts = now_ts();
        for id in ids {
            let _ = c.execute("UPDATE memories SET hits=hits+1, last_hit_at=? WHERE id=?", params![ts, id]);
        }
    }

    pub fn list_memories(&self, scopes: Option<&[String]>, include_superseded: bool, limit: usize) -> Vec<Memory> {
        let c = self.conn.lock();
        let mut sql = format!("SELECT {MEM_COLS} FROM memories WHERE deleted=0");
        if !include_superseded {
            sql.push_str(" AND superseded_by IS NULL");
        }
        let mut args: Vec<String> = vec![];
        if let Some(sc) = scopes {
            if sc.is_empty() {
                return vec![];
            }
            sql.push_str(&format!(" AND scope IN ({})", vec!["?"; sc.len()].join(",")));
            args.extend(sc.iter().cloned());
        }
        sql.push_str(&format!(" ORDER BY pinned DESC, created_at DESC LIMIT {limit}"));
        let mut st = match c.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        st.query_map(rusqlite::params_from_iter(args.iter()), row_to_memory)
            .map(|it| it.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub fn pinned_and_top(&self, scopes: &[String], n: usize) -> Vec<Memory> {
        if scopes.is_empty() {
            return vec![];
        }
        let c = self.conn.lock();
        let sql = format!(
            "SELECT {MEM_COLS} FROM memories WHERE deleted=0 AND superseded_by IS NULL AND scope IN ({}) \
             AND (pinned=1 OR kind IN ('preference','procedure','gotcha','decision')) \
             ORDER BY pinned DESC, (salience + 0.05*hits) DESC, created_at DESC LIMIT {n}",
            vec!["?"; scopes.len()].join(",")
        );
        let mut st = match c.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        st.query_map(rusqlite::params_from_iter(scopes.iter()), row_to_memory)
            .map(|it| it.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub fn count_memories(&self) -> HashMap<String, i64> {
        let c = self.conn.lock();
        let mut st = c.prepare("SELECT scope, COUNT(*) FROM memories WHERE deleted=0 AND superseded_by IS NULL GROUP BY scope").unwrap();
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).map(|it| it.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// BM25 over the pre-tokenized column. Returns (id, score) with higher = better.
    pub fn fts_search(&self, query: &str, scopes: &[String], k: usize) -> Vec<(String, f64)> {
        let toks: Vec<String> = tokenize(query).into_iter().filter(|t| t.chars().count() > 1 || !t.is_ascii()).collect();
        if toks.is_empty() || scopes.is_empty() {
            return vec![];
        }
        let q = toks.iter().map(|t| format!("\"{}\"", t.replace('"', ""))).collect::<Vec<_>>().join(" OR ");
        let c = self.conn.lock();
        let sql = format!(
            "SELECT f.id, bm25(mem_fts) AS s FROM mem_fts f JOIN memories m ON m.id=f.id \
             WHERE mem_fts MATCH ? AND m.deleted=0 AND m.superseded_by IS NULL AND m.scope IN ({}) ORDER BY s LIMIT {k}",
            vec!["?"; scopes.len()].join(",")
        );
        let mut args = vec![q];
        args.extend(scopes.iter().cloned());
        let mut st = match c.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        st.query_map(rusqlite::params_from_iter(args.iter()), |r| Ok((r.get::<_, String>(0)?, -r.get::<_, f64>(1)?)))
            .map(|it| it.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// BM25 (k1=1.2, b=0.75) over live memories in `scopes`, with weighted query terms from
    /// `lexicon::query_terms` (exact tokens, stems, CJK unigrams, bilingual concepts). Each hit also carries the
    /// idf-weighted fraction of the query it covers, a calibrated "how much of the question does it touch".
    pub fn lex_search(&self, query: &str, scopes: &[String], k: usize) -> Vec<LexHit> {
        let terms = crate::lexicon::query_terms(query);
        let lx = self.lex.read();
        let n = lx.docs.len() as f64;
        if terms.is_empty() || n == 0.0 || scopes.is_empty() {
            return vec![];
        }
        let avgdl = (lx.total_len / n).max(1.0);
        let (k1, b) = (1.2, 0.75);
        let idf = |t: &str| -> f64 {
            let df = lx.post.get(t).map(|v| v.len()).unwrap_or(0) as f64;
            (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
        };
        // Coverage: per query word (group), the best-matching variant's idf × weight, over the query's total mass.
        let mut mass: HashMap<usize, f64> = HashMap::new();
        for t in &terms {
            if let Some(g) = t.group {
                let m = mass.entry(g).or_default();
                *m = m.max(t.weight * idf(&t.term));
            }
        }
        let total: f64 = mass.values().sum();
        let mut acc: HashMap<&str, (f64, HashMap<usize, f64>)> = HashMap::new();
        for t in &terms {
            let i = idf(&t.term);
            let Some(ids) = lx.post.get(&t.term) else { continue };
            for id in ids {
                let (scope, tf, len) = &lx.docs[id];
                if !scopes.iter().any(|s| s == scope) {
                    continue;
                }
                let f = tf[&t.term] as f64;
                let s = t.weight * i * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * *len as f64 / avgdl));
                let e = acc.entry(id.as_str()).or_default();
                e.0 += s;
                if let Some(g) = t.group {
                    let c = e.1.entry(g).or_default();
                    *c = c.max(t.weight * i);
                }
            }
        }
        let mut v: Vec<LexHit> = acc
            .into_iter()
            .map(|(id, (s, c))| LexHit { id: id.to_string(), score: s, coverage: if total > 0.0 { (c.values().sum::<f64>() / total).min(1.0) } else { 0.0 } })
            .collect();
        v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
        v.truncate(k);
        v
    }

    /// Inverse document frequency of an index term over live memories (BM25 form; unseen terms get the maximum).
    pub fn idf(&self, term: &str) -> f64 {
        let lx = self.lex.read();
        let n = lx.docs.len() as f64;
        let df = lx.post.get(term).map(|v| v.len()).unwrap_or(0) as f64;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Exact cosine search over live memories in the given scopes.
    pub fn vec_search(&self, q: &[f32], scopes: &[String], k: usize, include_superseded: bool) -> Vec<(String, f32)> {
        let m = self.vecs.read();
        let mut v: Vec<(String, f32)> = m
            .iter()
            .filter(|(_, e)| (include_superseded || e.live) && scopes.iter().any(|s| s == &e.scope))
            .map(|(id, e)| (id.clone(), cosine(q, &e.v)))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        v.truncate(k);
        v
    }

    pub fn embedding(&self, id: &str) -> Option<Vec<f32>> {
        self.vecs.read().get(id).map(|e| e.v.clone())
    }

    pub fn all_ids_for_reembed(&self) -> Vec<(String, String)> {
        let c = self.conn.lock();
        let mut st = c.prepare("SELECT id, content || ' ' || COALESCE(why,'') FROM memories WHERE deleted=0").unwrap();
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).map(|it| it.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    pub fn set_embedding(&self, id: &str, emb: &[f32]) -> Result<()> {
        self.conn.lock().execute("UPDATE memories SET embedding=? WHERE id=?", params![vec_to_blob(emb), id])?;
        if let Some(e) = self.vecs.write().get_mut(id) {
            e.v = emb.to_vec();
        }
        Ok(())
    }

    pub fn reload_vectors(&self) -> Result<()> {
        self.vecs.write().clear();
        self.load_vectors()?;
        *self.lex.write() = LexIndex::default();
        self.load_lexical()
    }

    // ---------- edges ----------
    pub fn add_edge(&self, e: &Edge) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO edges(src,dst,rel,weight,judged_by,created_at) VALUES (?,?,?,?,?,?) \
             ON CONFLICT(src,dst,rel) DO UPDATE SET weight=excluded.weight, judged_by=excluded.judged_by",
            params![e.src, e.dst, e.rel, e.weight, e.judged_by, now_ts()],
        )?;
        Ok(())
    }

    pub fn neighbors(&self, id: &str, rels: &[&str]) -> Vec<Edge> {
        let c = self.conn.lock();
        let sql = format!(
            "SELECT src,dst,rel,weight,judged_by FROM edges WHERE (src=?1 OR dst=?1) AND rel IN ({})",
            rels.iter().map(|r| format!("'{r}'")).collect::<Vec<_>>().join(",")
        );
        let mut st = match c.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        st.query_map([id], |r| {
            Ok(Edge { src: r.get(0)?, dst: r.get(1)?, rel: r.get(2)?, weight: r.get(3)?, judged_by: r.get(4)? })
        })
        .map(|it| it.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn latest_in_session(&self, session: &str, scope: &str, exclude: &str) -> Option<String> {
        let c = self.conn.lock();
        c.query_row(
            "SELECT id FROM memories WHERE deleted=0 AND scope=? AND id<>? AND json_extract(source,'$.session')=? ORDER BY created_at DESC LIMIT 1",
            params![scope, exclude, session],
            |r| r.get(0),
        )
        .optional()
        .ok()
        .flatten()
    }

    // ---------- projects ----------
    pub fn upsert_project(&self, p: &Project) -> Result<()> {
        self.conn.lock().execute(
            &format!(
                "INSERT INTO projects ({PROJ_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(pid) DO UPDATE SET \
                 root=excluded.root, remote=COALESCE(excluded.remote, projects.remote), name=excluded.name, summary=excluded.summary, \
                 stack=excluded.stack, aliases=excluded.aliases, summary_by=excluded.summary_by, updated_at=excluded.updated_at"
            ),
            params![
                p.pid,
                p.root,
                p.remote,
                p.name,
                p.summary,
                p.stack,
                serde_json::to_string(&p.aliases)?,
                p.summary_by,
                p.created_at,
                p.updated_at
            ],
        )?;
        Ok(())
    }

    pub fn get_project(&self, pid: &str) -> Option<Project> {
        self.conn
            .lock()
            .query_row(&format!("SELECT {PROJ_COLS} FROM projects WHERE pid=?"), [pid], row_to_project)
            .optional()
            .ok()
            .flatten()
    }

    pub fn list_projects(&self) -> Vec<Project> {
        let c = self.conn.lock();
        let mut st = c.prepare(&format!("SELECT {PROJ_COLS} FROM projects ORDER BY updated_at DESC")).unwrap();
        st.query_map([], row_to_project).map(|it| it.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    // ---------- sessions ----------
    pub fn touch_session(&self, key: &str, harness: &str, pid: Option<&str>, cwd: Option<&str>) -> Result<bool> {
        let c = self.conn.lock();
        let ts = now_ts();
        let existed: bool = c.query_row("SELECT 1 FROM sessions WHERE key=?", [key], |_| Ok(true)).optional()?.unwrap_or(false);
        c.execute(
            "INSERT INTO sessions(key,pid,harness,cwd,first_seen,last_seen) VALUES (?,?,?,?,?,?) \
             ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen, pid=COALESCE(excluded.pid, sessions.pid), cwd=COALESCE(excluded.cwd, sessions.cwd)",
            params![key, pid, harness, cwd, ts, ts],
        )?;
        Ok(!existed)
    }

    pub fn idle_sessions(&self, idle_secs: i64) -> Vec<SessionRow> {
        let c = self.conn.lock();
        let cutoff = now_ts() - idle_secs;
        let mut st = c
            .prepare(
                "SELECT s.key,s.pid,s.harness,s.cwd,s.first_seen,s.last_seen FROM sessions s WHERE s.last_seen < ? \
                 AND EXISTS (SELECT 1 FROM episodes e WHERE e.session=s.key AND e.distilled=0)",
            )
            .unwrap();
        st.query_map([cutoff], |r| {
            Ok(SessionRow {
                key: r.get(0)?,
                pid: r.get(1)?,
                harness: r.get(2)?,
                cwd: r.get(3)?,
                first_seen: r.get(4)?,
                last_seen: r.get(5)?,
            })
        })
        .map(|it| it.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    // ---------- episodes ----------
    pub fn add_episode(&self, session: &str, harness: &str, pid: Option<&str>, role: &str, text: &str) -> Result<()> {
        let h = crate::util::hash_hex(&[session, role, text]);
        self.conn.lock().execute(
            "INSERT OR IGNORE INTO episodes(session,harness,pid,role,text,ts,hash) VALUES (?,?,?,?,?,?,?)",
            params![session, harness, pid, role, text, now_ts(), h],
        )?;
        Ok(())
    }

    pub fn undistilled_episodes(&self, session: &str) -> Vec<Episode> {
        let c = self.conn.lock();
        let mut st = c
            .prepare("SELECT id,session,harness,pid,role,text,ts FROM episodes WHERE session=? AND distilled=0 ORDER BY id")
            .unwrap();
        st.query_map([session], |r| {
            Ok(Episode {
                id: r.get(0)?,
                session: r.get(1)?,
                harness: r.get(2)?,
                pid: r.get(3)?,
                role: r.get(4)?,
                text: r.get(5)?,
                ts: r.get(6)?,
            })
        })
        .map(|it| it.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn mark_distilled(&self, session: &str, upto: i64) -> Result<()> {
        self.conn.lock().execute("UPDATE episodes SET distilled=1 WHERE session=? AND id<=?", params![session, upto])?;
        Ok(())
    }

    pub fn episode_count(&self) -> i64 {
        self.conn.lock().query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0)).unwrap_or(0)
    }

    // ---------- overlays ----------
    /// Store an overlay. A re-store under the same anchor by the *same* session replaces it (a retried turn
    /// produced new hidden rounds); an existing overlay owned by another session is never touched.
    pub fn put_overlay(&self, anchor: &str, session: &str, kind: &str, payload: &serde_json::Value) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO overlays(anchor,session,kind,payload,created_at) VALUES (?,?,?,?,?) \
             ON CONFLICT(anchor) DO UPDATE SET kind=excluded.kind, payload=excluded.payload, created_at=excluded.created_at \
             WHERE overlays.session = excluded.session",
            params![anchor, session, kind, payload.to_string(), now_ts()],
        )?;
        Ok(())
    }

    /// Fetch an overlay owned by `session` (overlays are never served across sessions).
    pub fn get_overlay(&self, anchor: &str, session: &str) -> Option<(String, serde_json::Value)> {
        let c = self.conn.lock();
        c.query_row("SELECT kind, payload FROM overlays WHERE anchor=? AND session=?", [anchor, session], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })
        .optional()
        .ok()
        .flatten()
        .and_then(|(k, p)| serde_json::from_str(&p).ok().map(|v| (k, v)))
    }

    pub fn gc_overlays(&self, older_than_secs: i64) -> Result<usize> {
        Ok(self.conn.lock().execute("DELETE FROM overlays WHERE created_at < ?", [now_ts() - older_than_secs])?)
    }

    // ---------- ui helpers ----------
    /// All graph edges (newest first), capped.
    pub fn all_edges(&self, limit: usize) -> Vec<Edge> {
        let c = self.conn.lock();
        let mut st = match c.prepare(&format!("SELECT src,dst,rel,weight,judged_by FROM edges ORDER BY created_at DESC LIMIT {limit}")) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        st.query_map([], |r| Ok(Edge { src: r.get(0)?, dst: r.get(1)?, rel: r.get(2)?, weight: r.get(3)?, judged_by: r.get(4)? }))
            .map(|it| it.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// The most recent raw episode turns, newest first.
    pub fn recent_episodes(&self, n: usize) -> Vec<Episode> {
        let c = self.conn.lock();
        let mut st = match c.prepare(&format!("SELECT id,session,harness,pid,role,text,ts FROM episodes ORDER BY id DESC LIMIT {n}")) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        st.query_map([], |r| Ok(Episode { id: r.get(0)?, session: r.get(1)?, harness: r.get(2)?, pid: r.get(3)?, role: r.get(4)?, text: r.get(5)?, ts: r.get(6)? }))
            .map(|it| it.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    // ---------- traces ----------
    pub fn trace(&self, kind: &str, data: &serde_json::Value) {
        let _ = self.conn.lock().execute(
            "INSERT INTO traces(ts,kind,data) VALUES (?,?,?)",
            params![now_ts(), kind, data.to_string()],
        );
    }

    pub fn recent_traces(&self, n: usize) -> Vec<(i64, String, String)> {
        let c = self.conn.lock();
        let mut st = c.prepare(&format!("SELECT ts,kind,data FROM traces ORDER BY id DESC LIMIT {n}")).unwrap();
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).map(|it| it.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn overlays_are_session_owned() {
        let s = Store::open_memory().unwrap();
        s.put_overlay("anc1", "sess-a", "k", &json!({"v": "a"})).unwrap();
        // another session can neither read nor overwrite it
        assert!(s.get_overlay("anc1", "sess-b").is_none());
        s.put_overlay("anc1", "sess-b", "k", &json!({"v": "b"})).unwrap();
        assert_eq!(s.get_overlay("anc1", "sess-a").unwrap().1, json!({"v": "a"}));
        assert!(s.get_overlay("anc1", "sess-b").is_none());
        // the owning session can update it (a retried turn)
        s.put_overlay("anc1", "sess-a", "k", &json!({"v": "a2"})).unwrap();
        assert_eq!(s.get_overlay("anc1", "sess-a").unwrap().1, json!({"v": "a2"}));
    }
}
