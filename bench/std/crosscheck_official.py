#!/usr/bin/env python3
"""Cross-check unimem-std-bench against LongMemEval's official retrieval code.

Re-runs the official `flat-bm25` baseline (rank_bm25.BM25Okapi over str.split(" ") tokens, as in
src/retrieval/run_retrieval.py) and the official metric functions (src/retrieval/eval_utils.py, fetched from GitHub
into <cache>/ref/) on every evaluated question, and compares with the bench's `bm25-okapi-ref` run files.

usage: pip install rank_bm25 numpy; python3 bench/std/crosscheck_official.py --cache DIR
"""
import argparse, importlib.util, json, os, urllib.request
import numpy as np

np.asfarray = getattr(np, "asfarray", lambda a: np.asarray(a, dtype=float))  # removed in numpy 2
URL = "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/retrieval/eval_utils.py"


def corpus(e, gran):  # verbatim logic of run_retrieval.process_item_flat_index
    C, I = [], []
    for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
        if gran == "session":
            C.append(" ".join(t["content"] for t in sess if t["role"] == "user"))
            noans = "answer" in sid and all(not t.get("has_answer") for t in sess if t["role"] == "user")
            I.append(sid.replace("answer", "noans") if noans else sid)
        else:
            for i, t in enumerate(sess):
                if t["role"] == "user":
                    C.append(t["content"])
                    d = f"{sid}_{i + 1}"
                    I.append(d if ("answer" not in sid or t.get("has_answer")) else d.replace("answer", "noans"))
    return C, I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    from rank_bm25 import BM25Okapi
    ref = os.path.join(a.cache, "ref", "eval_utils.py")
    if not os.path.exists(ref):
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        urllib.request.urlretrieve(URL, ref)
    spec = importlib.util.spec_from_file_location("lme_eval_utils", ref)
    eu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eu)
    data = {e["question_id"]: e for e in json.load(open(os.path.join(a.cache, "data/longmemeval_s_cleaned.json")))}
    for gran in ["session", "turn"]:
        recs = [json.loads(l) for l in open(os.path.join(a.cache, f"runs/lme-{gran}-bm25-okapi-ref.jsonl"))]
        off, ours = {}, {}
        for r in recs:
            e = data[r["qid"]]
            C, I = corpus(e, gran)
            rank = np.argsort(BM25Okapi([d.split(" ") for d in C]).get_scores(e["question"].split(" ")))[::-1]
            gold = list(set(d for d in I if "answer" in d))
            for k in [5, 10]:
                fns = [("", eu.evaluate_retrieval)] + ([("session_from_turn ", eu.evaluate_retrieval_turn2session)] if gran == "turn" else [])
                for tag, fn in fns:
                    _, l, n = fn(rank, gold, I, k=k)
                    lvl = "session" if (gran == "session" or tag) else "turn"
                    for name, v in [(f"{tag}recall_all@{k}", l), (f"{tag}ndcg_any@{k}", n)]:
                        off.setdefault(name, []).append(v)
                        ours.setdefault(name, []).append(r["metrics"][lvl][name.replace(tag, "")])
        print(f"LongMemEval_S {gran} index, {len(recs)} questions:")
        for k in off:
            print(f"  {k:32s} official {np.mean(off[k]):.4f}   unimem-std-bench {np.mean(ours[k]):.4f}   max |diff| {np.max(np.abs(np.array(off[k]) - np.array(ours[k]))):.2e}")


if __name__ == "__main__":
    main()
