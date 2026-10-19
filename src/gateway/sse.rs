//! Minimal incremental Server-Sent-Events parser / writer.

use bytes::Bytes;

#[derive(Debug, Clone)]
pub struct Event {
    pub event: Option<String>,
    pub data: String,
}

impl Event {
    pub fn json(&self) -> Option<serde_json::Value> {
        serde_json::from_str(&self.data).ok()
    }
}

#[derive(Default)]
pub struct Parser {
    buf: Vec<u8>,
}

fn find_boundary(b: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] == b'\n' && b[i + 1] == b'\n' {
            return Some((i, 2));
        }
        if i + 3 < b.len() && &b[i..i + 4] == b"\r\n\r\n" {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

impl Parser {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Event> {
        self.buf.extend_from_slice(chunk);
        let mut out = vec![];
        while let Some((pos, len)) = find_boundary(&self.buf) {
            let raw: Vec<u8> = self.buf.drain(..pos + len).collect();
            if let Some(ev) = parse_block(&String::from_utf8_lossy(&raw[..pos])) {
                out.push(ev);
            }
        }
        out
    }

    /// Flush a trailing event without the final blank line.
    pub fn finish(&mut self) -> Option<Event> {
        let raw = std::mem::take(&mut self.buf);
        parse_block(&String::from_utf8_lossy(&raw))
    }
}

fn parse_block(s: &str) -> Option<Event> {
    let mut event = None;
    let mut data: Vec<&str> = vec![];
    for line in s.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(v) = line.strip_prefix("event:") {
            event = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("data:") {
            data.push(v.strip_prefix(' ').unwrap_or(v));
        }
    }
    if event.is_none() && data.is_empty() {
        return None;
    }
    Some(Event { event, data: data.join("\n") })
}

pub fn encode(event: Option<&str>, data: &str) -> Bytes {
    let mut s = String::with_capacity(data.len() + 32);
    if let Some(e) = event {
        s.push_str("event: ");
        s.push_str(e);
        s.push('\n');
    }
    for line in data.split('\n') {
        s.push_str("data: ");
        s.push_str(line);
        s.push('\n');
    }
    s.push('\n');
    Bytes::from(s)
}

pub fn encode_json(event: Option<&str>, v: &serde_json::Value) -> Bytes {
    encode(event, &v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_split_events() {
        let mut p = Parser::default();
        assert!(p.push(b"event: a\nda").is_empty());
        let e = p.push(b"ta: {\"x\":1}\n\nevent: b\r\ndata: 2\r\n\r\n");
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].event.as_deref(), Some("a"));
        assert_eq!(e[0].data, "{\"x\":1}");
        assert_eq!(e[1].data, "2");
    }
}
