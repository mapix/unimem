"""Generate the data figures of the unimem paper from committed result files.
Run from the repo root: python3 -I docs/paper/figs/make_figs.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "legend.fontsize": 6.5, "legend.frameon": False})
OUT = "docs/paper/figs"
e5 = json.load(open("bench/results/v0.4/e5_long_session.json"))["variants"]

def series(name, key):
    pr = e5[name]["per_request"]
    return [r["request"] for r in pr], [r[key] for r in pr]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
styles = [("heavy/no-budget-no-dedup", "no budget, no ledger", "#c0392b", "-"),
          ("heavy/no-budget", "ledger only", "#e67e22", "--"),
          ("heavy/v0.4", "budgets + ledger (v0.4)", "#1f4e79", "-"),
          ("no-budget-no-dedup", "normal notes, no ledger", "#888888", ":"),
          ("v0.4", "normal notes, v0.4", "#2e8b57", "-.")]
for name, lab, c, ls in styles:
    x, y = series(name, "unimem_live_tokens")
    axs[0].plot(x, [v / 1000 for v in y], color=c, ls=ls, lw=1.0, label=lab)
    x, y = series(name, "cumulative_new_tokens")
    axs[1].plot(x, [v / 1000 for v in y], color=c, ls=ls, lw=1.0, label=lab)
for a in axs:
    a.axvline(60, color="k", lw=0.5, ls=(0, (2, 2)))
    a.set_xlabel("client request")
    a.grid(alpha=0.25, lw=0.4)
axs[0].text(61, axs[0].get_ylim()[1] * 0.92, "compaction", fontsize=6.5)
axs[0].set_ylabel("unimem tokens in live context (k)")
axs[1].set_ylabel("cumulative new injected tokens (k)")
axs[1].legend(loc="upper left")
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/e5_session.pdf")

# R@k curves, test split (v0.5)
lme = json.load(open("bench/results/v0.5/std/longmemeval_s_retrieval.test.json"))
loc = json.load(open("bench/results/v0.5/std/locomo_retrieval.test.json"))
ks = [1, 3, 5, 10, 25, 50]
systems = [("bm25", "BM25 (stemmed)", "#888888", "--"), ("e5-small", "e5-small", "#e67e22", ":"),
           ("hybrid-rrf", "BM25+e5 RRF", "#c0392b", "-."), ("unimem-topk-hash", "unimem top-k (hash)", "#2e8b57", "--"),
           ("unimem-topk-e5", "unimem top-k (e5)", "#1f4e79", "-")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
for s, lab, c, ls in systems:
    g = lme[f"lme-turn-{s}"]["groups"]["overall"]
    axs[0].plot(ks, [g[f"turn/recall_all@{k}"] for k in ks], marker="o", ms=2.5, color=c, ls=ls, lw=1.0, label=lab)
    g = loc[f"locomo-turn-{s}"]["groups"]["overall(1-4)"]
    axs[1].plot(ks, [g[f"turn/recall_frac@{k}"] for k in ks], marker="o", ms=2.5, color=c, ls=ls, lw=1.0, label=lab)
# adaptive recall as a single point at its mean depth
for ax, d, key, grp, metric in [(axs[0], lme, "lme-turn-unimem-e5", "overall", "turn/recall_all@10"),
                                (axs[1], loc, "locomo-turn-unimem-e5", "overall(1-4)", "turn/recall_frac@10")]:
    r = d[key]
    ax.plot([r["notes_returned_mean"]], [r["groups"][grp][metric]], marker="*", ms=7, color="#6a1b9a", ls="none",
            label="unimem adaptive (e5)")
axs[0].set_title("LongMemEval$_S$ turn index, test (n=203)", fontsize=8)
axs[1].set_title("LoCoMo evidence recall, cat. 1-4 (n=1536)", fontsize=8)
for a in axs:
    a.set_xscale("log"); a.set_xticks(ks); a.set_xticklabels([str(k) for k in ks])
    a.set_xlabel("k (notes returned)"); a.grid(alpha=0.25, lw=0.4)
axs[0].set_ylabel("Recall@k")
axs[0].legend(loc="upper left")
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/rk_curves.pdf")
print("ok")
