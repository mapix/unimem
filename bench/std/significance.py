#!/usr/bin/env python3
"""Paired bootstrap (2000 resamples, 95% CI) and win/loss counts for key system pairs, from the run files.
usage: python3 bench/std/significance.py CACHE_DIR"""
import os, sys
CACHE = sys.argv[1] if len(sys.argv) > 1 else "bench-cache"
import json, random
def load(f, lvl, key):
    return {r['qid']: r['metrics'][lvl][key] for r in map(json.loads, open(os.path.join(CACHE, 'runs', f + '.jsonl')))}
def comp(a, b, lvl, key, tag):
    A, B = load(a, lvl, key), load(b, lvl, key)
    q = sorted(A); d = [A[x] - B[x] for x in q]
    rng = random.Random(0); bs = []
    for _ in range(2000):
        s = [d[rng.randrange(len(d))] for _ in d]; bs.append(sum(s) / len(s))
    bs.sort(); w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
    print(f"{tag:10s} {a.split('-',2)[2]:18s} - {b.split('-',2)[2]:16s} {key:16s} {sum(d)/len(d):+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}] win/loss {w}/{l}")
for g in ['lme-session', 'lme-turn']:
    lvl = 'session' if g == 'lme-session' else 'turn'
    for a, b in [('unimem-prior-e5', 'hybrid-rrf'), ('unimem-prior-e5', 'e5-small'), ('unimem-prior-hash', 'bm25'), ('unimem-prior-e5', 'unimem-prior-hash'), ('hybrid-rrf', 'e5-small'), ('e5-small', 'bm25')]:
        comp(f'{g}-{a}', f'{g}-{b}', lvl, 'recall_all@10', g)
for a, b in [('unimem-prior-e5', 'hybrid-rrf'), ('unimem-prior-hash', 'bm25'), ('unimem-prior-e5', 'e5-small'), ('hybrid-rrf', 'e5-small')]:
    comp(f'locomo-turn-{a}', f'locomo-turn-{b}', 'turn', 'recall_frac@10', 'locomo')
