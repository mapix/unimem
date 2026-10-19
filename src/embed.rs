//! Embedders. `hash` is offline and dependency-free (signed feature hashing of tokens, stems, bilingual concepts
//! and char trigrams);
//! `openai` calls any `/v1/embeddings` endpoint (OpenAI, Ollama, vLLM, TEI…).

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::config::{resolve_key, EmbedderCfg};
use crate::util::tokenize;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn id(&self) -> String;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[text.to_string()]).await?.pop().unwrap_or_default())
    }
}

pub struct HashEmbedder {
    pub dim: usize,
}

fn h64(s: &str) -> u64 {
    let d = Sha256::digest(s.as_bytes());
    u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

impl HashEmbedder {
    pub fn embed_sync(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let toks = tokenize(text);
        let mut add = |feat: &str, w: f32| {
            let h = h64(feat);
            let idx = (h % self.dim as u64) as usize;
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[idx] += sign * w;
        };
        for t in &toks {
            // stopwords carry no topic; keep them faint so short questions are not dominated by "what is the"
            let w = if crate::screen::is_stopword(t) || crate::screen::is_question_word(t) { 0.25 } else { 1.0 };
            let st = crate::lexicon::stem(t);
            if st != *t {
                add(&format!("w:{st}"), 0.7 * w);
                add(&format!("w:{t}"), 0.5 * w);
            } else {
                add(&format!("w:{t}"), w);
            }
            // char trigrams give some robustness to inflection (deploy/deploys/deployment)
            let cs: Vec<char> = format!("^{t}$").chars().collect();
            if cs.len() > 4 {
                for w in cs.windows(3) {
                    add(&format!("c:{}", w.iter().collect::<String>()), 0.35);
                }
            }
        }
        for w in toks.windows(2) {
            add(&format!("b:{} {}", w[0], w[1]), 0.5);
        }
        // parts of compound identifiers (SyncPeriod → sync, period)
        for w in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            for p in crate::lexicon::identifier_parts(w) {
                add(&format!("w:{}", crate::lexicon::stem(&p)), 0.5);
            }
        }
        // bilingual concepts: an English note and its Chinese restatement share these features
        for c in crate::lexicon::concepts(text) {
            add(&format!("w:{c}"), 0.8);
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn id(&self) -> String {
        format!("hash2-{}", self.dim)
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_sync(t)).collect())
    }
}

pub struct OpenAiEmbedder {
    client: reqwest::Client,
    base_url: String,
    key: Option<String>,
    model: String,
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn id(&self) -> String {
        format!("openai:{}", self.model)
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/').trim_end_matches("/v1"));
        let mut rb = self.client.post(url).json(&serde_json::json!({"model": self.model, "input": texts}));
        if let Some(k) = &self.key {
            rb = rb.bearer_auth(k);
        }
        let r = rb.send().await?;
        if !r.status().is_success() {
            return Err(anyhow!("embeddings {}: {}", r.status(), r.text().await.unwrap_or_default()));
        }
        let v: serde_json::Value = r.json().await?;
        let mut data = v["data"].as_array().ok_or_else(|| anyhow!("bad embeddings response"))?.clone();
        data.sort_by_key(|d| d["index"].as_u64().unwrap_or(0));
        Ok(data
            .iter()
            .map(|d| d["embedding"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default())
            .collect())
    }
}

pub fn build(cfg: &EmbedderCfg) -> Box<dyn Embedder> {
    match cfg.kind.as_str() {
        "openai" if !cfg.base_url.is_empty() && !cfg.model.is_empty() => Box::new(OpenAiEmbedder {
            client: reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap(),
            base_url: cfg.base_url.clone(),
            key: resolve_key(&cfg.api_key, &cfg.api_key_env),
            model: cfg.model.clone(),
        }),
        _ => Box::new(HashEmbedder { dim: cfg.dim.max(64) }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::cosine;
    #[test]
    fn similar_texts_are_closer() {
        let e = HashEmbedder { dim: 512 };
        let a = e.embed_sync("we deploy through ops/deploy.sh");
        let b = e.embed_sync("deployment goes through the ops deploy script");
        let c = e.embed_sync("the month-end job runs at 22:00");
        assert!(cosine(&a, &b) > cosine(&a, &c));
    }
}
