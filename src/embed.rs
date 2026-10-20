//! Embedders. `hash` is offline and dependency-free (signed feature hashing of tokens, stems, bilingual concepts
//! and char trigrams);
//! `openai` calls any `/v1/embeddings` endpoint (OpenAI, Ollama, vLLM, TEI…);
//! `onnx` (cargo feature `onnx`) runs a local transformer such as multilingual-e5-small through ONNX Runtime.

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
    /// Embed a search query. Symmetric embedders use `embed_one`; asymmetric ones (E5: "query: " vs "passage: ")
    /// override this.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(text).await
    }
    /// Raw cosine at or above which two notes are candidates for the write-time near-duplicate shortcut (the
    /// shortcut also requires lexical skeleton agreement). Per embedder family: hash 0.97, transformers 0.95.
    fn near_dup_cos(&self) -> f32 {
        near_dup_for(&self.id())
    }
    /// Typical cosine of two unrelated texts (anisotropy baseline). 0 for the hash embedder.
    fn cos_baseline(&self) -> f32 {
        cos_baseline_for(&self.id())
    }
}

/// One loaded model shared by several engines (a transformer embedder is hundreds of MB; load it once).
#[async_trait]
impl Embedder for std::sync::Arc<dyn Embedder> {
    fn id(&self) -> String {
        (**self).id()
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts).await
    }
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_one(text).await
    }
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text).await
    }
    fn near_dup_cos(&self) -> f32 {
        (**self).near_dup_cos()
    }
    fn cos_baseline(&self) -> f32 {
        (**self).cos_baseline()
    }
}

/// Default near-duplicate cosine by embedder id (override: `embedder.near_dup_cos`).
pub fn near_dup_for(id: &str) -> f32 {
    let id = id.to_lowercase();
    if id.starts_with("hash") {
        0.97
    } else {
        0.95
    }
}

/// Default anisotropy baseline by embedder id (override: `embedder.cos_baseline`). E5: median cosine of random
/// LoCoMo turn pairs is 0.82 (bench/std/RESULTS.md); we use a slightly lower value so that weak relations survive.
pub fn cos_baseline_for(id: &str) -> f32 {
    let id = id.to_lowercase();
    if id.contains("e5") {
        0.75
    } else {
        0.0
    }
}

/// A downloadable model for the `onnx` embedder (`unimem models download <name>`).
pub struct ModelSpec {
    pub name: &'static str,
    /// directory name under `<data_dir>/models/` (contains "e5", which selects the E5 prefixes and calibration)
    pub dir: &'static str,
    pub base_url: &'static str,
    /// (file, sha256 or "" when not pinned)
    pub files: &'static [(&'static str, &'static str)],
}

/// intfloat/multilingual-e5-small, official ONNX export (118M parameters, 384-d, ~100 languages incl. Chinese).
pub const E5_SMALL: ModelSpec = ModelSpec {
    name: "e5-small",
    dir: "multilingual-e5-small",
    base_url: "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx",
    files: &[
        ("model.onnx", "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"),
        ("tokenizer.json", ""),
        ("config.json", ""),
        ("special_tokens_map.json", ""),
        ("tokenizer_config.json", ""),
    ],
};

pub fn model_spec(name: &str) -> Option<&'static ModelSpec> {
    match name {
        "e5-small" | "multilingual-e5-small" | "me5s" => Some(&E5_SMALL),
        _ => None,
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

/// Local transformer embedder through ONNX Runtime (cargo feature `onnx`; `kind = "onnx"`).
///
/// Built for the E5 family (default: `intfloat/multilingual-e5-small`, 384-d, 100 languages incl. Chinese) and other
/// mean-pooled BERT-style encoders: `model` is a directory holding `model.onnx` + `tokenizer.json` (e.g. the `onnx/`
/// folder of the Hugging Face repo). Texts are truncated to 512 tokens, the last hidden state is mean-pooled under
/// the attention mask and L2-normalised. E5 needs asymmetric prefixes: notes are embedded as `"passage: …"`, and
/// [`Embedder::embed_query`] uses `"query: …"`. The ONNX Runtime shared library is loaded at runtime from
/// `ORT_DYLIB_PATH` (e.g. the `libonnxruntime.so.*` that ships in the Python `onnxruntime` wheel), so no binaries
/// are downloaded at build time.
#[cfg(feature = "onnx")]
pub mod onnx {
    use super::*;
    use parking_lot::Mutex;
    use std::path::Path;

    pub struct OnnxEmbedder {
        session: Mutex<ort::session::Session>,
        tokenizer: tokenizers::Tokenizer,
        name: String,
        has_token_types: bool,
        pub query_prefix: String,
        pub passage_prefix: String,
        pub max_len: usize,
    }

    impl OnnxEmbedder {
        pub fn load(dir: &Path, threads: usize) -> Result<Self> {
            Self::load_with(dir, threads, None)
        }

        /// Like [`Self::load`]; `ort_dylib` overrides `$ORT_DYLIB_PATH`.
        pub fn load_with(dir: &Path, threads: usize, ort_dylib: Option<&str>) -> Result<Self> {
            let model = if dir.join("model.onnx").exists() { dir.join("model.onnx") } else { dir.join("onnx/model.onnx") };
            let tok = if dir.join("tokenizer.json").exists() { dir.join("tokenizer.json") } else { dir.join("onnx/tokenizer.json") };
            if !model.exists() {
                return Err(anyhow!("no model.onnx in {} (run `unimem models download e5-small`)", dir.display()));
            }
            let tokenizer = tokenizers::Tokenizer::from_file(&tok).map_err(|e| anyhow!("tokenizer {}: {e}", tok.display()))?;
            match ort_dylib.map(String::from).filter(|p| !p.is_empty()).or_else(|| std::env::var("ORT_DYLIB_PATH").ok()) {
                Some(p) => {
                    let _ = ort::init_from(p).commit();
                }
                None => return Err(anyhow!("ONNX Runtime library not configured: set embedder.ort_dylib or ORT_DYLIB_PATH to libonnxruntime")),
            }
            let session = ort::session::Session::builder()
                .and_then(|b| b.with_intra_threads(threads.max(1)))
                .and_then(|b| b.commit_from_file(&model))
                .map_err(|e| anyhow!("onnx model {}: {e}", model.display()))?;
            let has_token_types = session.inputs.iter().any(|i| i.name == "token_type_ids");
            let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let e5 = dir.to_string_lossy().to_lowercase().contains("e5");
            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                name,
                has_token_types,
                query_prefix: if e5 { "query: ".into() } else { String::new() },
                passage_prefix: if e5 { "passage: ".into() } else { String::new() },
                max_len: 512,
            })
        }

        /// Embed already-prefixed texts in one padded batch.
        pub fn embed_raw(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let encs = self.tokenizer.encode_batch(texts.to_vec(), true).map_err(|e| anyhow!("tokenize: {e}"))?;
            let lens: Vec<usize> = encs.iter().map(|e| e.get_ids().len().min(self.max_len)).collect();
            let (b, l) = (texts.len(), lens.iter().copied().max().unwrap_or(1).max(1));
            let mut ids = vec![0i64; b * l];
            let mut mask = vec![0i64; b * l];
            for (r, e) in encs.iter().enumerate() {
                let mut toks: Vec<u32> = e.get_ids().to_vec();
                if toks.len() > self.max_len {
                    // keep the closing special token, as HF truncation does
                    let last = *toks.last().unwrap();
                    toks.truncate(self.max_len);
                    toks[self.max_len - 1] = last;
                }
                for (c, t) in toks.iter().enumerate() {
                    ids[r * l + c] = *t as i64;
                    mask[r * l + c] = 1;
                }
            }
            let shape = [b as i64, l as i64];
            let t_ids = ort::value::Tensor::from_array((shape, ids.clone())).map_err(|e| anyhow!("{e}"))?;
            let t_mask = ort::value::Tensor::from_array((shape, mask.clone())).map_err(|e| anyhow!("{e}"))?;
            let mut sess = self.session.lock();
            let outputs = if self.has_token_types {
                let t_tt = ort::value::Tensor::from_array((shape, vec![0i64; b * l])).map_err(|e| anyhow!("{e}"))?;
                sess.run(ort::inputs!["input_ids" => t_ids, "attention_mask" => t_mask, "token_type_ids" => t_tt])
            } else {
                sess.run(ort::inputs!["input_ids" => t_ids, "attention_mask" => t_mask])
            }
            .map_err(|e| anyhow!("onnx run: {e}"))?;
            let (oshape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
            let h = *oshape.last().unwrap_or(&0) as usize;
            let mut out = Vec::with_capacity(b);
            for r in 0..b {
                let mut v = vec![0f32; h];
                let mut n = 0f32;
                for c in 0..l {
                    if mask[r * l + c] == 1 {
                        let row = &data[(r * l + c) * h..(r * l + c + 1) * h];
                        v.iter_mut().zip(row).for_each(|(a, x)| *a += x);
                        n += 1.0;
                    }
                }
                v.iter_mut().for_each(|x| *x /= n.max(1e-9));
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter_mut().for_each(|x| *x /= norm);
                out.push(v);
            }
            Ok(out)
        }
    }

    #[async_trait]
    impl Embedder for OnnxEmbedder {
        fn id(&self) -> String {
            format!("onnx:{}", self.name)
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let t: Vec<String> = texts.iter().map(|s| format!("{}{s}", self.passage_prefix)).collect();
            let mut out = vec![];
            for c in t.chunks(16) {
                out.extend(self.embed_raw(c)?);
            }
            Ok(out)
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.embed_raw(&[format!("{}{text}", self.query_prefix)])?.pop().unwrap_or_default())
        }
    }
}

/// Build the configured embedder. An unusable configuration falls back to the hash embedder with a warning; the
/// store records the embedder id, so a later switch is reported (`unimem reembed`).
pub fn build(cfg: &EmbedderCfg) -> Box<dyn Embedder> {
    match cfg.kind.as_str() {
        #[cfg(feature = "onnx")]
        "onnx" => {
            let dir = crate::util::expand_tilde(if cfg.model.is_empty() { "~/.unimem/models/multilingual-e5-small" } else { &cfg.model });
            let threads = if cfg.threads == 0 { 4 } else { cfg.threads };
            match onnx::OnnxEmbedder::load_with(&dir, threads, Some(&cfg.ort_dylib)) {
                Ok(e) => Box::new(e),
                Err(e) => {
                    tracing::warn!("onnx embedder unavailable ({e}); falling back to hash");
                    Box::new(HashEmbedder { dim: cfg.dim.max(64) })
                }
            }
        }
        #[cfg(not(feature = "onnx"))]
        "onnx" => {
            tracing::warn!("embedder.kind = \"onnx\" needs a build with `--features onnx`; falling back to hash");
            Box::new(HashEmbedder { dim: cfg.dim.max(64) })
        }
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

    #[test]
    fn per_embedder_calibration() {
        let h = HashEmbedder { dim: 64 };
        assert_eq!(h.near_dup_cos(), 0.97);
        assert_eq!(h.cos_baseline(), 0.0);
        assert_eq!(near_dup_for("onnx:multilingual-e5-small"), 0.95);
        assert_eq!(cos_baseline_for("onnx:multilingual-e5-small"), 0.75);
    }
}
