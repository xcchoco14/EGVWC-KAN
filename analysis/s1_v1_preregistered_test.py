"""Pre-registered test of the v1 benchmark on shard B (preregistration/preregistration_v1_shardB.md).

Primary: EGVW-KAN with the standardised score vs EGVW-KAN (gamma = 1), mean MCC per dataset, two-sided Wilcoxon,
alpha = 0.05. Secondary (Holm, reported only): standardised score vs uniform weights (Ens+CW+Boot+Guards), vs the tuned
single KAN, and vs the classical baseline with the highest mean MCC over the 24 datasets.
"""
import numpy as np, pandas as pd
from scipy import stats
from common import results, save, holm, V1_BASELINES

P1B = results("v1_part1_B")
B = results(*V1_BASELINES)
STD, G1 = "EGVW-KAN [standardised score]", "EGVW-KAN"

# a dataset counts only if both primary rows have all 15 folds
n = P1B[P1B.method.isin([STD, G1])].groupby(["dataset", "method"]).size().unstack()
ds = sorted(n.index[(n[STD] == 15) & (n[G1] == 15)])
assert len(ds) == 14, f"expected 14 complete shard-B datasets, got {len(ds)}"
P = P1B[P1B.dataset.isin(ds)]

best = B.groupby(["dataset", "method"]).mcc.mean().unstack().mean().idxmax()
cells = P[["dataset", "repeat", "fold"]].drop_duplicates()
Bb = B[B.method == best].merge(cells, on=["dataset", "repeat", "fold"])
M = pd.concat([P, Bb]).groupby(["dataset", "method"]).mcc.mean().unstack()


def wil(a, b, data=M):
    d = data[a] - data[b]
    t = stats.wilcoxon(data[a], data[b], alternative="two-sided")
    return dict(a=a, b=b, mean_a=round(float(data[a].mean()), 4), mean_b=round(float(data[b].mean()), 4),
                delta=round(float(d.mean()), 4), median_delta=round(float(d.median()), 4),
                wins=int((d > 1e-12).sum()), losses=int((d < -1e-12).sum()), ties=int((d.abs() <= 1e-12).sum()),
                W=float(t.statistic), p=float(t.pvalue))


primary = wil(STD, G1)
decision = "standardised score adopted" if (primary["delta"] > 0 and primary["p"] < 0.05) else "original definition kept"
sec = [wil(STD, c) for c in ("Ens+CW+Boot+Guards", "KAN (tuned)", best)]
for s, a in zip(sec, holm([s["p"] for s in sec])): s["p_holm"] = float(a)
other = {}
for met in ("balanced_accuracy", "auroc", "brier"):
    X = P[P.method.isin([STD, G1])].groupby(["dataset", "method"])[met].mean().unstack()
    other[met] = wil(STD, G1, X)

per_ds = M[[STD, G1, "Ens+CW+Boot+Guards", "KAN (tuned)", best]].round(4)
save(dict(datasets=ds, n_datasets=len(ds), folds_per_dataset=15, best_baseline=best, primary=primary, decision=decision,
          secondary=sec, other_metrics=other, per_dataset_mcc=per_ds.to_dict()), "v1_preregistered_test.json")
print("primary:", primary["delta"], f"{primary['wins']}/{primary['losses']}", "p =", round(primary["p"], 4), "->", decision)
