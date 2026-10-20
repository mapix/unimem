#!/usr/bin/env python3
"""Precompute multilingual-e5-small embeddings for the standard benchmarks (resumable).

This is the Python twin of unimem's Rust `onnx` embedder backend (`src/embed.rs`, cargo feature `onnx`):
same ONNX graph (intfloat/multilingual-e5-small, `onnx/model.onnx`), same tokenizer.json, "query: " / "passage: "
prefixes, truncation to 512 tokens, attention-masked mean pooling, L2 normalisation. Both write the same cache
format, which `unimem-std-bench` reads:

    <cache>/emb/<set>.texts.jsonl   one JSON string per line (the unit text WITHOUT prefix), fixed order
    <cache>/emb/<set>.e5s.f32       row i = embedding of line i, little-endian f32, dim 384, appended in order

The cache is append-only and resumable: each run embeds until --budget seconds are used, then exits; rerun until
it prints "done". (The sandbox this was developed in kills any process after ~3 minutes.)

Sets (text extraction must match src/bin/unimem-std-bench.rs):
    lme_turn      LongMemEval_S user turns (content as-is)                        prefix "passage: "
    lme_session   LongMemEval_S sessions: ' '.join(user turn contents)            prefix "passage: "
    lme_query     LongMemEval_S questions                                         prefix "query: "
    locomo_turn   LoCoMo turns: "<speaker>: <text>" (+ " [shares <blip_caption>]")  prefix "passage: "
    locomo_query  LoCoMo questions                                                prefix "query: "

usage: python3 bench/std/embed_e5.py --cache DIR --model DIR --set lme_turn [--budget 160] [--threads 4]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

DIM = 384


def lme_items(data_dir, kind):
    d = json.load(open(os.path.join(data_dir, "longmemeval_s_cleaned.json")))
    out = []
    for e in d:
        if kind == "query":
            out.append(e["question"])
            continue
        for sess in e["haystack_sessions"]:
            users = [t["content"] for t in sess if t["role"] == "user"]
            if kind == "turn":
                out.extend(users)
            else:
                out.append(" ".join(users))
    return out


def locomo_turn_text(t):
    s = f"{t['speaker']}: {t.get('text', '')}"
    if t.get("blip_caption"):
        s += f" [shares {t['blip_caption']}]"
    return s


def locomo_items(data_dir, kind):
    d = json.load(open(os.path.join(data_dir, "locomo10.json")))
    out = []
    for conv in d:
        if kind == "query":
            out.extend(q["question"] for q in conv["qa"])
            continue
        c = conv["conversation"]
        for k, v in c.items():
            if k.startswith("session_") and isinstance(v, list):
                out.extend(locomo_turn_text(t) for t in v)
    return out


def texts_for(set_name, data_dir):
    ds, kind = set_name.split("_", 1)
    items = lme_items(data_dir, kind) if ds == "lme" else locomo_items(data_dir, kind)
    uniq = sorted(set(items), key=lambda s: (len(s), s))  # length-sorted: little padding per batch
    return uniq, ("query: " if kind == "query" else "passage: ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True, help="dir with model.onnx + tokenizer.json")
    ap.add_argument("--data", default=None, help="dataset dir (default <cache>/data)")
    ap.add_argument("--set", required=True)
    ap.add_argument("--budget", type=float, default=160.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    data_dir = a.data or os.path.join(a.cache, "data")
    emb_dir = os.path.join(a.cache, "emb")
    os.makedirs(emb_dir, exist_ok=True)
    tpath = os.path.join(emb_dir, f"{a.set}.texts.jsonl")
    vpath = os.path.join(emb_dir, f"{a.set}.e5s.f32")
    if not os.path.exists(tpath):
        texts, prefix = texts_for(a.set, data_dir)
        with open(tpath + ".tmp", "w") as f:
            for s in texts:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        os.replace(tpath + ".tmp", tpath)
    else:
        texts = [json.loads(l) for l in open(tpath)]
        prefix = "query: " if a.set.endswith("query") else "passage: "
    have = os.path.getsize(vpath) // (DIM * 4) if os.path.exists(vpath) else 0
    if os.path.exists(vpath) and os.path.getsize(vpath) != have * DIM * 4:  # torn write: drop partial row
        with open(vpath, "r+b") as f:
            f.truncate(have * DIM * 4)
    print(f"{a.set}: {len(texts)} texts, {have} cached", flush=True)
    if have >= len(texts):
        print("done")
        return

    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    tok.enable_truncation(512)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(os.path.join(a.model, "model.onnx"), so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    i = have
    n_done = 0
    with open(vpath, "ab") as out:
        while i < len(texts) and time.time() - t0 < a.budget:
            # token-budgeted batch of similar-length texts
            j = i
            batch = []
            maxlen = 0
            while j < len(texts):
                enc = tok.encode(prefix + texts[j])
                L = len(enc.ids)
                if batch and max(maxlen, L) * (len(batch) + 1) > a.batch_tokens:
                    break
                batch.append(enc)
                maxlen = max(maxlen, L)
                j += 1
                if len(batch) >= 128:
                    break
            ids = np.zeros((len(batch), maxlen), dtype=np.int64)
            mask = np.zeros((len(batch), maxlen), dtype=np.int64)
            for r, e in enumerate(batch):
                ids[r, : len(e.ids)] = e.ids
                mask[r, : len(e.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = sess.run(None, feed)[0]  # [b, L, 384]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            out.write(v.astype("<f4").tobytes())
            out.flush()
            n_done += len(batch)
            i = j
    dt = time.time() - t0
    print(f"{a.set}: +{n_done} in {dt:.0f}s -> {i}/{len(texts)}", flush=True)
    if i >= len(texts):
        print("done")


if __name__ == "__main__":
    sys.exit(main())
