#!/usr/bin/env python3
"""Matched-budget comparison: unimem's recall returns an adaptive number of notes (often 1-3), so Recall@10 penalises
it against baselines that always return 10. For every question, cut each baseline's ranking at the number of notes
unimem returned for that question and compare recall_any / evidence recall / precision at that same depth.
usage: python3 bench/std/matched_budget.py CACHE_DIR"""
import json, os, sys

CACHE = sys.argv[1] if len(sys.argv) > 1 else "bench-cache"


def load(name):
    return {r["qid"]: r for r in map(json.loads, open(os.path.join(CACHE, "runs", name + ".jsonl")))}


def gold_of(r, gold_runs):
    return gold_runs[r["qid"]]


def main():
    data = json.load(open(os.path.join(CACHE, "data/longmemeval_s_cleaned.json")))
    lme_gold = {}
    for e in data:
        g_t, g_s = set(), set()
        for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
            for i, t in enumerate(sess):
                if t["role"] == "user" and t.get("has_answer") and "answer" in sid:
                    g_t.add(f"{sid}_{i + 1}")
                    g_s.add(sid)
        lme_gold[e["question_id"]] = {"turn": g_t, "session": g_s}
    loc = json.load(open(os.path.join(CACHE, "data/locomo10.json")))
    for setting, unimem in [("lme-session", "unimem-e5"), ("lme-turn", "unimem-e5"), ("locomo-turn", "unimem-e5"), ("lme-turn", "unimem-hash"), ("locomo-turn", "unimem-hash")]:
        U = load(f"{setting}-{unimem}")
        print(f"\n{setting}: depth = #notes returned by {unimem} per question (mean {sum(len(r['top']) for r in U.values()) / len(U):.2f})")
        print(f"  {'system':18s} {'hit (any gold)':>15s} {'evidence recall':>16s} {'precision':>10s}")
        for sysname in [unimem, "bm25", "e5-small", "hybrid-rrf", unimem.replace("unimem-", "unimem-prior-")]:
            R = load(f"{setting}-{sysname}")
            hit = rec = prec = 0.0
            for q, u in U.items():
                n = max(len(u["top"]), 1)
                top = R[q]["top"][:n]
                if setting.startswith("lme"):
                    gold = lme_gold[q]["session" if setting == "lme-session" else "turn"]
                else:
                    gold = None
                if gold is None:  # LoCoMo: gold from the run's metrics is not stored; recompute from evidence
                    sample, i = q.split("#")
                    conv = next(c for c in loc if c["sample_id"] == sample)
                    raw = conv["qa"][int(i)]["evidence"]
                    gold = set()
                    for ev in raw:
                        for p in ev.replace(";", " ").split():
                            p = p.replace("D:", "D")
                            if ":" in p:
                                a, b = p.lstrip("D").split(":", 1)
                                gold.add(f"D{a.lstrip('0')}:{b.lstrip('0')}")
                got = [x for x in top if x in gold]
                hit += bool(got)
                rec += len(set(got)) / max(len(gold), 1)
                prec += len(got) / n
            N = len(U)
            print(f"  {sysname:18s} {hit / N:15.3f} {rec / N:16.3f} {prec / N:10.3f}")


if __name__ == "__main__":
    main()
