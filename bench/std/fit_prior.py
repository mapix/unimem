#!/usr/bin/env python3
"""Fit unimem's relevance prior (src/engine.rs `relevance_prior`) on DEVELOPMENT data only.

Inputs: evidence files written by
  * `UNIMEM_BENCH_EVIDENCE=F unimem-bench e1 --data bench/data/{devmem,heldout}.json`  (product-style notes)
  * `unimem-std-bench evidence --dataset lme --gran {session,turn} --embedder {hash,e5}` (dev half of LongMemEval_S:
    sha256(question_id)[0] even; the command never writes test-split questions)
Each line: {"qid", "cat", "n_gold", "cands": [{"y": 0|1, "ev": Evidence}]}.

Model: one logistic regression per embedder family (evidence file suffix -hash / -e5) on normalised features (coverage, lex_rel, cov_rel, cos_z, cos_zrel, rrf,
reason_match). Every source (file) gets the same total weight and every query the same weight inside its source, so
neither the 245-candidate LongMemEval haystacks nor one embedder dominate. L2 = 1e-3. Then the intercept alone is
re-fitted on the product-style sources (devmem + heldout), because the recall thresholds (T_RELEVANT, KEEP_RATIO,
T_SUFFICIENT) were calibrated on that scale.

Prints dev ranking metrics (unit recall_all@k / recall_any@k over the candidate pool) for the old v0.4 prior and the
new fit, per source, and writes the weights as JSON.
usage: python3 -I bench/std/fit_prior.py OUT.json EVIDENCE.jsonl...
"""
import json, math, os, sys
import numpy as np

FEATS = ["coverage", "lex_rel", "cov_rel", "cos_z", "cos_zrel", "rrf", "reason_match"]
Z_MIN, Z_MAX = -3.0, 8.0
PRODUCT = ("devmem", "heldout")


def feat(e):
    return [1.0, e["coverage"], e["lex_rel"], e["cov_rel"], min(max(e["cos_z"], Z_MIN), Z_MAX), e["cos_zrel"], e["rrf"], 1.0 if e["reason_match"] else 0.0]


def old_prior(e):  # v0.4 weights (fitted on hash-embedder raw cosines)
    z = -5.1 + 1.2 * e["coverage"] + 1.5 * e["lex_rel"] + 0.25 * max(e["cos"], 0) + 0.6 * e["cos_rel"] + 1.7 * e["cov_rel"] + (1.0 if e["reason_match"] else 0.0)
    return z


def load(paths):
    srcs = {}
    for p in paths:
        name = os.path.basename(p).replace(".jsonl", "")
        qs = [json.loads(l) for l in open(p)]
        srcs[name] = [q for q in qs if q["cands"] and q["n_gold"] > 0]
    return srcs


def design(srcs, names):
    X, y, w = [], [], []
    for n in names:
        qs = srcs[n]
        for q in qs:
            m = len(q["cands"])
            for c in q["cands"]:
                X.append(feat(c["ev"]))
                y.append(c["y"])
                w.append(1.0 / (len(qs) * m))
    w = np.array(w)
    return np.array(X), np.array(y, dtype=float), w / w.sum()


def fit(X, y, w, l2=1e-3, iters=20000, lr=2.0, fixed=None):
    b = np.zeros(X.shape[1]) if fixed is None else fixed.copy()
    mask = np.ones(X.shape[1]) if fixed is None else np.r_[1.0, np.zeros(X.shape[1] - 1)]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (w * (p - y)) + l2 * np.r_[0.0, b[1:]] * (fixed is None)
        b -= lr * g * mask
    return b


def metrics(qs, score):
    out = {}
    for k in (1, 5, 10):
        a = l = 0
        for q in qs:
            s = sorted(q["cands"], key=lambda c: -score(c["ev"]))[:k]
            hit = sum(c["y"] for c in s)
            a += hit > 0
            l += hit >= q["n_gold"]
        out[f"R_any@{k}"] = round(a / len(qs), 4)
        out[f"R_all@{k}"] = round(l / len(qs), 4)
    return out


def fit_family(srcs, names):
    X, y, w = design(srcs, names)
    b = fit(X, y, w)
    prod = [n for n in names if n.startswith(PRODUCT)]
    if prod:
        Xp, yp, wp = design(srcs, prod)
        b = fit(Xp, yp, wp, fixed=b, iters=5000)
    return b


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    srcs = load(paths)
    # one weight set per embedder family (file name suffix -hash / -e5): the engine picks it by embedder id
    fams = {}
    for n in sorted(srcs):
        fams.setdefault(n.rsplit("-", 1)[1], []).append(n)
    rep = {"features": FEATS, "families": {}, "n_queries": {n: len(srcs[n]) for n in srcs}}
    for fam, names in fams.items():
        b = fit_family(srcs, names)
        W = dict(zip(["bias"] + FEATS, [round(float(v), 3) for v in b]))
        print(fam, "weights", W)
        new = lambda e, b=b: float(np.dot(feat(e), b))
        res = {}
        for n in names:
            r = {"v0.4-prior": metrics(srcs[n], old_prior), "rrf-only": metrics(srcs[n], lambda e: e["rrf"]), "v0.5-prior": metrics(srcs[n], new)}
            res[n] = r
            print(f"  {n:20s} n={len(srcs[n]):4d}", " | ".join(f"{k}: R_all@5 {v['R_all@5']:.3f} R_all@10 {v['R_all@10']:.3f} R_any@1 {v['R_any@1']:.3f}" for k, v in r.items()))
        rep["families"][fam] = {"weights": W, "dev_metrics": res}
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
