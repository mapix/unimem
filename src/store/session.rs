//! Per-conversation gateway state (`session_state` table): what unimem has put into the live context of a
//! conversation (the de-dup ledger and its token total), compaction-detection signals, and once-per-session flags.
//! One JSON row per session key; read-modify-write from the request handlers (last writer wins, which is fine:
//! every field is a cache of what the next request will re-derive).

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::Store;
use crate::util::now_ts;

/// One note (or a batch of notes, `id` = "") unimem injected, and the message anchor it lives in.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LedgerEntry {
    /// Memory id (`m_…`).
    pub id: String,
    /// The `augment_user` / expansion anchor whose overlay carries it.
    pub anchor: String,
    /// Estimated tokens of the rendered note.
    pub tokens: usize,
    /// Prompt-turn counter when it was injected.
    pub turn: u64,
    /// "passive" | "preamble" | "tool" | "tool_error".
    pub via: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SessionState {
    /// Harness-visible message count of the last request.
    pub last_msg_count: usize,
    /// Hash of the first user message of the last request.
    pub first_user_hash: String,
    /// The preamble is in the live context (cleared on compaction).
    pub preamble_sent: bool,
    /// Tokens unimem added to the live context since the last compaction (overlays are replayed every request).
    pub injected_tokens: usize,
    /// Notes in the live context.
    pub ledger: Vec<LedgerEntry>,
    /// Requests seen / prompt turns / tool-result turns.
    pub requests: u64,
    pub prompt_turns: u64,
    pub tool_turns: u64,
    /// `tool_turns` at the last tool-error-triggered recall.
    pub last_tool_recall_at: Option<u64>,
    /// Error signatures already answered by a tool-triggered recall.
    pub error_sigs: Vec<String>,
    /// Detected compactions.
    pub compactions: u64,
    /// Number of augment overlays that matched in the last request (and how many exist).
    pub aug_hits: usize,
    pub aug_total: usize,
    /// Pre-compaction flush done (nudge sent, background distill requested).
    pub flush_done: bool,
    /// Estimated context tokens of the last request (harness history + replayed overlays).
    pub context_tokens: usize,
    pub window: u64,
    /// Corrections detected on prompt turns.
    pub corrections: u64,
    /// Key of the summary item / message seen last (compaction marker).
    pub summary_key: String,
    /// At least one augment overlay was stored for this conversation.
    pub aug_stored: bool,
    /// Tool-error-triggered recalls that injected notes.
    pub tool_recalls: u64,
    /// Last compaction signal ("msg_count_drop", "first_user_changed", "summary_marker", "zero_anchor_hits").
    pub last_compaction: String,
    /// Session-level row only: the conversation thread (first-user hash) seen last.
    pub last_thread: String,
    /// Real context size from the last upstream usage (input + cache + output of the last round).
    pub last_usage_context: usize,
    /// A compaction request (the harness asking the model to summarize) was passed through.
    pub compaction_requested: bool,
}

impl SessionState {
    /// A compaction (or rewind) dropped the context unimem had added: forget it so the preamble and notes can be
    /// injected again.
    pub fn reset_after_compaction(&mut self) {
        self.preamble_sent = false;
        self.injected_tokens = 0;
        self.ledger.clear();
        self.error_sigs.clear();
        self.flush_done = false;
        self.aug_stored = false;
        self.compactions += 1;
    }
    pub fn in_ledger(&self, id: &str) -> bool {
        self.ledger.iter().any(|e| e.id == id)
    }
}

pub(super) fn migrate(c: &Connection) {
    // v0.4 columns on older databases (ignore "duplicate column" errors)
    let _ = c.execute("ALTER TABLE episodes ADD COLUMN importance INTEGER NOT NULL DEFAULT 0", []);
}

impl Store {
    pub fn session_state(&self, session: &str) -> SessionState {
        if let Some(s) = self.sstate.lock().get(session) {
            return s.clone();
        }
        let s = self.load_session_state(session);
        self.sstate.lock().insert(session.to_string(), s.clone());
        s
    }

    fn load_session_state(&self, session: &str) -> SessionState {
        self.conn
            .lock()
            .query_row("SELECT state FROM session_state WHERE session=?", [session], |r| r.get::<_, String>(0))
            .optional()
            .ok()
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn put_session_state(&self, session: &str, st: &SessionState) -> Result<()> {
        self.sstate.lock().insert(session.to_string(), st.clone());
        self.conn.lock().execute(
            "INSERT INTO session_state(session,state,updated_at) VALUES (?,?,?) ON CONFLICT(session) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
            params![session, serde_json::to_string(st)?, now_ts()],
        )?;
        Ok(())
    }

    /// Apply `f` to the session's state and store it.
    pub fn update_session_state<R>(&self, session: &str, f: impl FnOnce(&mut SessionState) -> R) -> R {
        let mut s = self.session_state(session);
        let r = f(&mut s);
        if let Err(e) = self.put_session_state(session, &s) {
            tracing::warn!("storing session state failed: {e}");
        }
        r
    }

    /// Update the cached state only (not persisted until the next [`Store::put_session_state`]): for values that are
    /// only useful while the gateway runs, written on every upstream round.
    pub fn update_session_state_mem(&self, session: &str, f: impl FnOnce(&mut SessionState)) {
        let mut s = self.session_state(session);
        f(&mut s);
        self.sstate.lock().insert(session.to_string(), s);
    }

    /// Drop session state (and legacy `pre:` flags) not touched for `older_than_secs`.
    pub fn gc_session_state(&self, older_than_secs: i64) -> Result<usize> {
        self.sstate.lock().clear();
        let c = self.conn.lock();
        let n = c.execute("DELETE FROM session_state WHERE updated_at < ?", [now_ts() - older_than_secs])?;
        c.execute("DELETE FROM meta WHERE k LIKE 'pre:%'", [])?;
        Ok(n)
    }

    /// Episode TTL: delete episodes older than `older_than_secs` (distilled or not).
    pub fn gc_episodes(&self, older_than_secs: i64) -> Result<usize> {
        Ok(self.conn.lock().execute("DELETE FROM episodes WHERE ts < ?", [now_ts() - older_than_secs])?)
    }

    /// Mark exactly these episodes as distilled.
    pub fn mark_distilled_ids(&self, ids: &[i64]) -> Result<()> {
        let c = self.conn.lock();
        for id in ids {
            c.execute("UPDATE episodes SET distilled=1 WHERE id=?", [id])?;
        }
        Ok(())
    }

    /// Raise the importance of the session's newest episode of `role` (correction capture).
    pub fn bump_last_episode(&self, session: &str, role: &str, by: i64) -> Result<()> {
        self.conn.lock().execute(
            "UPDATE episodes SET importance=importance+? WHERE id=(SELECT id FROM episodes WHERE session=? AND role=? ORDER BY id DESC LIMIT 1)",
            params![by, session, role],
        )?;
        Ok(())
    }

    /// Ids of the session's episodes with importance ≥ `min` (corrections).
    pub fn important_episodes(&self, session: &str, min: i64) -> std::collections::HashSet<i64> {
        let c = self.conn.lock();
        let Ok(mut st) = c.prepare("SELECT id FROM episodes WHERE session=? AND importance>=?") else { return Default::default() };
        st.query_map(params![session, min], |r| r.get(0)).map(|it| it.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    pub fn episode_importance(&self, session: &str) -> Vec<(String, i64)> {
        let c = self.conn.lock();
        let Ok(mut st) = c.prepare("SELECT role, importance FROM episodes WHERE session=? ORDER BY id") else { return vec![] };
        st.query_map([session], |r| Ok((r.get(0)?, r.get(1)?))).map(|it| it.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// The most recently seen *other* session in the same project and harness (session-end signal).
    pub fn previous_session_in_project(&self, session: &str, harness: &str, pid: &str) -> Option<String> {
        self.conn
            .lock()
            .query_row(
                "SELECT key FROM sessions WHERE pid=? AND harness=? AND key<>? AND EXISTS (SELECT 1 FROM episodes e WHERE e.session=sessions.key AND e.distilled=0) ORDER BY last_seen DESC LIMIT 1",
                params![pid, harness, session],
                |r| r.get(0),
            )
            .optional()
            .ok()
            .flatten()
    }

    /// (documents containing `term`, live documents) in the lexical index.
    pub fn doc_freq(&self, term: &str) -> (usize, usize) {
        let lx = self.lex.read();
        (lx.post.get(term).map(|v| v.len()).unwrap_or(0), lx.docs.len())
    }

    pub fn session_row(&self, key: &str) -> Option<super::SessionRow> {
        self.conn
            .lock()
            .query_row("SELECT key,pid,harness,cwd,first_seen,last_seen FROM sessions WHERE key=?", [key], |r| {
                Ok(super::SessionRow { key: r.get(0)?, pid: r.get(1)?, harness: r.get(2)?, cwd: r.get(3)?, first_seen: r.get(4)?, last_seen: r.get(5)? })
            })
            .optional()
            .ok()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_roundtrip_and_reset() {
        let s = Store::open_memory().unwrap();
        assert!(!s.session_state("a").preamble_sent);
        s.update_session_state("a", |st| {
            st.preamble_sent = true;
            st.injected_tokens = 500;
            st.ledger.push(LedgerEntry { id: "m_1".into(), anchor: "x".into(), tokens: 20, turn: 1, via: "passive".into() });
        });
        let mut st = s.session_state("a");
        assert!(st.preamble_sent && st.in_ledger("m_1"));
        assert!(!s.session_state("b").preamble_sent, "per session");
        st.reset_after_compaction();
        assert!(!st.preamble_sent && st.ledger.is_empty() && st.injected_tokens == 0 && st.compactions == 1);
    }

    #[test]
    fn episode_ttl_and_chunk_marking() {
        let s = Store::open_memory().unwrap();
        s.add_episode("s", "h", None, "user", "one").unwrap();
        s.add_episode("s", "h", None, "user", "two").unwrap();
        let eps = s.undistilled_episodes("s");
        s.mark_distilled_ids(&[eps[0].id]).unwrap();
        assert_eq!(s.undistilled_episodes("s").len(), 1);
        s.bump_last_episode("s", "user", 3).unwrap();
        assert_eq!(s.episode_importance("s"), vec![("user".into(), 0), ("user".into(), 3)]);
        assert_eq!(s.gc_episodes(-10).unwrap(), 2);
        assert_eq!(s.episode_count(), 0);
    }
}
