//! v0.5: embedder-calibrated relevance prior (cosine z-score + RRF anchor fusion, weights per embedder family),
//! recall depth policy (adaptive cut-off for injection vs ranked top-k when k is asked for), and the onnx embedder
//! configuration. All examples are written for these tests.

use async_trait::async_trait;
use serde_json::json;
use unimem::config::Config;
use unimem::decide::{ChainDecider, HeuristicDecider};
use unimem::embed::{Embedder, HashEmbedder};
use unimem::engine::{relevance_prior, Ctx, Depth, Engine, Evidence, PriorWeights, RecallReq, WriteReq, PRIOR_DENSE, PRIOR_HASH};
use unimem::store::Store;

fn engine_with(emb: Box<dyn Embedder>) -> Engine {
    Engine::with_parts(Config::default(), Store::open_memory().unwrap(), emb, ChainDecider::new(vec![Box::new(HeuristicDecider)]))
}

fn engine() -> Engine {
    engine_with(Box::new(HashEmbedder { dim: 512 }))
}

const NOTES: &[&str] = &[
    "The kestrel ingest worker retries failed uploads three times with exponential backoff starting at 2 s.",
    "Kestrel stores raw uploads in the bucket kestrel-raw and keeps them for 30 days.",
    "Kestrel's ingest queue is Redis streams; the consumer group is called ingest-v2.",
    "The kestrel dashboard is built with SvelteKit and deployed to Cloudflare Pages.",
    "Kestrel API errors are reported to Sentry under the project kestrel-api.",
    "The kestrel nightly export runs at 03:00 UTC and writes parquet files to the analytics bucket.",
    "Kestrel uses Postgres 16; migrations live in db/migrations and run with sqlx migrate.",
    "The kestrel ingest worker is written in Rust with tokio and runs as four replicas.",
];

async fn seed(e: &Engine) -> Ctx {
    let ctx = Ctx { harness: "test".into(), session: "s".into(), ..Default::default() };
    for n in NOTES {
        e.write(&ctx, WriteReq { content: n.to_string(), scope: Some("global".into()), ..Default::default() }).await.unwrap();
    }
    ctx
}

#[tokio::test]
async fn ranking_mode_returns_k_ranked_notes_adaptive_stays_short() {
    let e = engine();
    let ctx = seed(&e).await;
    let q = "how does the kestrel ingest worker handle failed uploads?";
    let adaptive = e.recall(&ctx, RecallReq { query: q.into(), k: 8, ..Default::default() }).await.unwrap();
    let ranked = e.recall(&ctx, RecallReq { query: q.into(), k: 8, depth: Depth::TopK, ..Default::default() }).await.unwrap();
    assert!(adaptive.notes.len() < ranked.notes.len(), "adaptive {} vs ranked {}", adaptive.notes.len(), ranked.notes.len());
    assert_eq!(ranked.notes.len(), 8);
    // same best note, scores non-increasing
    assert_eq!(adaptive.notes[0].id, ranked.notes[0].id);
    assert!(ranked.notes[0].content.contains("retries"), "{:?}", ranked.notes[0]);
    assert!(ranked.notes.windows(2).all(|w| w[0].p >= w[1].p), "{:?}", ranked.notes.iter().map(|n| n.p).collect::<Vec<_>>());
    // k bounds the ranked list
    let r3 = e.recall(&ctx, RecallReq { query: q.into(), k: 3, depth: Depth::TopK, ..Default::default() }).await.unwrap();
    assert_eq!(r3.notes.len(), 3);
    assert_eq!(r3.notes.iter().map(|n| &n.id).collect::<Vec<_>>(), ranked.notes.iter().take(3).map(|n| &n.id).collect::<Vec<_>>());
}

#[tokio::test]
async fn tool_with_explicit_k_ranks_with_scores() {
    let e = engine();
    let ctx = seed(&e).await;
    let q = "kestrel ingest worker";
    let plain = unimem::tools::exec(&e, &ctx, "unimem_recall", &json!({"query": q})).await;
    let ranked = unimem::tools::exec(&e, &ctx, "unimem_recall", &json!({"query": q, "k": 6})).await;
    let count = |s: &str| s.lines().filter(|l| l.contains("] ") && l.contains("(m_")).count();
    assert!(!plain.contains("ranked by relevance"), "{plain}");
    assert!(ranked.contains("6 notes ranked by relevance"), "{ranked}");
    assert_eq!(count(&ranked), 6, "{ranked}");
    assert!(count(&plain) <= count(&ranked));
    // every ranked line starts with its score
    assert!(ranked.lines().filter(|l| l.contains("(m_")).all(|l| l.split_whitespace().next().is_some_and(|t| t.parse::<f64>().is_ok())), "{ranked}");
}

#[tokio::test]
async fn passive_injection_keeps_the_adaptive_cutoff() {
    let e = engine();
    let ctx = seed(&e).await;
    let out = e.recall(&ctx, RecallReq { query: "how does the kestrel ingest worker retry failed uploads?".into(), k: 8, passive: true, ..Default::default() }).await.unwrap();
    assert!(out.notes.len() <= 3, "{:?}", out.notes.len());
}

/// An anisotropic toy embedder: every vector shares a large common component (unrelated texts at cosine ≈ 0.8, as
/// with E5), plus the hash features. Raw cosines are compressed; the z-scored features are not.
struct Aniso(HashEmbedder);

#[async_trait]
impl Embedder for Aniso {
    fn id(&self) -> String {
        "onnx:toy-e5".into()
    }
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = self.0.embed_sync(t);
                v.push(2.0);
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                v
            })
            .collect())
    }
}

#[tokio::test]
async fn cosine_features_are_embedder_normalised() {
    let h = engine();
    let a = engine_with(Box::new(Aniso(HashEmbedder { dim: 512 })));
    seed(&h).await;
    seed(&a).await;
    let q = "kestrel parquet export schedule";
    let scopes = vec!["global".to_string()];
    let ch = h.candidates(q, &scopes, 30).await.unwrap();
    let ca = a.candidates(q, &scopes, 30).await.unwrap();
    // raw cosines differ a lot (anisotropy) ...
    let min_raw = ca.iter().map(|c| c.ev.cos).fold(f64::MAX, f64::min);
    assert!(min_raw > 0.7, "toy embedder should be anisotropic: {min_raw}");
    // ... but the z-scores rank the same note first, with the best note at cos_zrel = 1
    let top = |c: &[unimem::engine::Candidate]| c.iter().max_by(|x, y| x.ev.cos_z.partial_cmp(&y.ev.cos_z).unwrap()).unwrap().id.clone();
    let content = |e: &Engine, id: &str| e.store.get_memory(id).unwrap().content;
    assert_eq!(content(&h, &top(&ch)), content(&a, &top(&ca)));
    assert!(ca.iter().any(|c| (c.ev.cos_zrel - 1.0).abs() < 1e-9));
    assert!(content(&a, &ca[0].id).contains("nightly export"), "{:?}", ca.iter().map(|c| content(&a, &c.id)).collect::<Vec<_>>());
    // RRF feature: rank 1 in both lists is 1
    assert!(ca[0].ev.rrf > 0.9 && ca[0].ev.rrf <= 1.0 + 1e-9, "{}", ca[0].ev.rrf);
}

#[test]
fn prior_weights_follow_the_embedder_family() {
    assert_eq!(PriorWeights::for_embedder("hash2-512"), PRIOR_HASH);
    assert_eq!(PriorWeights::for_embedder("onnx:multilingual-e5-small"), PRIOR_DENSE);
    assert_eq!(PriorWeights::for_embedder("openai:bge-m3"), PRIOR_DENSE);
    // the dense family leans on the normalised cosine more, the hash family on BM25
    const { assert!(PRIOR_DENSE.cos_z + PRIOR_DENSE.cos_zrel > PRIOR_HASH.cos_z + PRIOR_HASH.cos_zrel) };
    const { assert!(PRIOR_HASH.lex_rel > PRIOR_DENSE.lex_rel) };
    // monotone in every feature
    let base = Evidence { coverage: 0.5, lex_rel: 0.5, cov_rel: 0.5, cos_z: 1.0, cos_zrel: 0.5, rrf: 0.5, ..Default::default() };
    for w in [PRIOR_HASH, PRIOR_DENSE] {
        let p0 = relevance_prior(&base, &w);
        for f in [|e: &mut Evidence| e.coverage += 0.1, |e: &mut Evidence| e.lex_rel += 0.1, |e: &mut Evidence| e.cos_z += 0.5, |e: &mut Evidence| e.rrf += 0.1] {
            let mut e = base.clone();
            f(&mut e);
            assert!(relevance_prior(&e, &w) > p0);
        }
        // z is clamped: an extreme outlier cosine does not dominate
        let mut e = base.clone();
        e.cos_z = 1e6;
        assert!(relevance_prior(&e, &w) < 0.99);
    }
}

#[test]
fn onnx_config_parses_and_falls_back_without_the_model() {
    let cfg: Config = toml::from_str("[embedder]\nkind = \"onnx\"\nmodel = \"/nonexistent/multilingual-e5-small\"\nort_dylib = \"/nonexistent/libonnxruntime.so\"\nthreads = 2\n").unwrap();
    assert_eq!(cfg.embedder.kind, "onnx");
    assert_eq!(cfg.embedder.threads, 2);
    // without the model (or without the `onnx` feature) the engine still starts, on the hash embedder
    let e = unimem::embed::build(&cfg.embedder);
    assert!(e.id().starts_with("hash"), "{}", e.id());
    assert_eq!(unimem::embed::model_spec("e5-small").unwrap().dir, "multilingual-e5-small");
    assert!(unimem::embed::model_spec("nope").is_none());
    // per-embedder duplicate thresholds (v0.4) are still in place
    assert_eq!(unimem::embed::near_dup_for("onnx:multilingual-e5-small"), 0.95);
    assert_eq!(unimem::embed::near_dup_for("hash2-512"), 0.97);
}
