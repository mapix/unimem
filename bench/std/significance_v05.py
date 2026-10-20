#!/usr/bin/env python3
"""v0.5 paired bootstrap (2000 resamples, 95% CI, seed 0) of every unimem system against hybrid-rrf (and a few other
pairs), on the TEST split (LongMemEval: sha256(question_id)[0] odd; LoCoMo: all of it, nothing was fitted on it)
and on the full set.
usage: python3 -I bench/std/significance_v05.py CACHE_DIR [RUNS_SUBDIR=runs-v05]"""
import hashlib, json, os, random, sys

CACHE = sys.argv[1] if len(sys.argv) > 1 else "bench-cache"
RUNS = sys.argv[2] if len(sys.argv) > 2 else "runs-v05"


def split(ds, qid):
    if ds != "lme":
        return "test"
    return "dev" if hashlib.sha256(qid.encode()).digest()[0] % 2 == 0 else "test"


def load(name, lvl, key):
    p = os.path.join(CACHE, RUNS, name + ".jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        if name.startswith("locomo") and r["cat"] == "5":
            continue  # headline LoCoMo numbers are categories 1-4
        out[r["qid"]] = r["metrics"][lvl][key]
    return out


def comp(ds, g, a, b, lvl, key, which):
    A, B = load(f"{g}-{a}", lvl, key), load(f"{g}-{b}", lvl, key)
    if A is None or B is None:
        return None
    q = sorted(x for x in A if x in B and (which == "all" or split(ds, x) == which))
    d = [A[x] - B[x] for x in q]
    rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    return {"a": a, "b": b, "group": g, "split": which, "metric": f"{lvl}/{key}", "n": len(d), "mean_a": sum(A[x] for x in q) / len(q), "mean_b": sum(B[x] for x in q) / len(q),
            "delta": sum(d) / len(d), "ci95": [bs[50], bs[1949]], "win": sum(x > 0 for x in d), "loss": sum(x < 0 for x in d)}


UNI = ["unimem-prior-hash", "unimem-prior-e5", "unimem-topk-hash", "unimem-topk-e5", "unimem-e5", "unimem-hash"]
rows = []
for which in ("test", "all"):
    for ds, g, lvl, key in [("lme", "lme-session", "session", "recall_all@10"), ("lme", "lme-session", "session", "recall_all@5"),
                            ("lme", "lme-turn", "turn", "recall_all@10"), ("lme", "lme-turn", "turn", "recall_all@5"),
                            ("locomo", "locomo-turn", "turn", "recall_frac@10"), ("locomo", "locomo-turn", "turn", "recall_frac@5")]:
        for a in UNI:
            r = comp(ds, g, a, "hybrid-rrf", lvl, key, which)
            if r:
                rows.append(r)
        for a, b in [("unimem-prior-e5", "e5-small"), ("unimem-prior-hash", "bm25"), ("unimem-prior-e5", "unimem-prior-hash"), ("unimem-topk-e5", "unimem-prior-e5")]:
            r = comp(ds, g, a, b, lvl, key, which)
            if r:
                rows.append(r)
for r in rows:
    print(f"{r['split']:4s} {r['group']:12s} {r['metric']:22s} {r['a']:18s} - {r['b']:17s} n={r['n']:4d} {r['mean_a']:.3f} vs {r['mean_b']:.3f}  "
          f"d={r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] w/l {r['win']}/{r['loss']}")
if len(sys.argv) > 3:
    json.dump(rows, open(sys.argv[3], "w"), indent=1)
