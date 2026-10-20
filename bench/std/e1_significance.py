#!/usr/bin/env python3
"""Paired bootstrap (2000 resamples, 95% CI, seed 0) of per-query R@5 hit and reciprocal rank on E1 answerable queries.
usage: python3 -I bench/std/e1_significance.py e1_memory.json [label]"""
import json, random, sys

d = json.load(open(sys.argv[1]))
lab = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
Q = [q for q in d["per_query"] if q["gold"]]


def hit(q, s):
    return float(any(g in q["systems"][s]["ret"][:5] for g in q["gold"]))


def rr(q, s):
    return q["systems"][s]["rr"]


def boot(a, b, f):
    x = [f(q, a) - f(q, b) for q in Q]
    rng = random.Random(0)
    bs = sorted(sum(x[rng.randrange(len(x))] for _ in x) / len(x) for _ in range(2000))
    return sum(x) / len(x), bs[50], bs[1949], sum(v > 0 for v in x), sum(v < 0 for v in x)


pairs = [("unimem-full", "vector-top5-cwd"), ("unimem-full", "bm25-top5-cwd"), ("unimem-top5", "vector-top5-cwd"), ("unimem-top5", "vector-top5"), ("unimem-top5", "bm25-top5-cwd"), ("unimem-top5", "unimem-full")]
for a, b in pairs:
    for name, f in (("R@5", hit), ("MRR", rr)):
        m, lo, hi, w, l = boot(a, b, f)
        print(f"{lab:14s} {a:12s} - {b:16s} {name:4s} n={len(Q)} d={m:+.3f} [{lo:+.3f}, {hi:+.3f}] w/l {w}/{l}")
