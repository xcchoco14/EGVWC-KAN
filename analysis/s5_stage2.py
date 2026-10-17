"""Stage 2 (confirmation) analysis as fixed in preregistration/preregistration_v2.md, section 4 and Amendment 1.

Primary: v2-final ("v2") vs EGVW-KAN v1, mean MCC per dataset, two-sided Wilcoxon signed-rank, alpha = 0.05, over the
datasets where both rows have all 15 folds.
Secondary: v2-final vs Ens+CW, vs KAN (tuned), vs CatBoost (MCC, Holm over the three); balanced accuracy, AUROC and Brier
for the primary pair (reported only).
Descriptive (Amendment 1): v2-final against every other method, collapse rates, mean ranks.
"""
import numpy as np, pandas as pd
from scipy import stats
from common import results, pred_files, save, holm, collapsed, STAGE2_KAN, STAGE2_COMP

FOLDS = 15
KAN = results(*STAGE2_KAN); COMP = results(*STAGE2_COMP)
ALL = pd.concat([KAN, COMP])

# all units present, and the three notebooks used the same test folds
cnt = ALL.assign(group=ALL.key.str.rsplit("|", n=1).str[1]).drop_duplicates("key").group.value_counts().to_dict()
h = ALL.drop_duplicates(["dataset", "repeat", "fold", "fold_hash"]).groupby(["dataset", "repeat", "fold"]).fold_hash.nunique()
assert (h == 1).all(), "fold hashes differ between notebooks"
assert cnt == {"kan": 195, "baselines": 195, "variants": 195}, cnt

M = ALL.groupby(["dataset", "method"]).mcc.mean().unstack()
N = ALL.groupby(["dataset", "method"]).size().unstack()
PRIMARY = ("v2", "EGVW-KAN")
full = N.index[(N[list(PRIMARY)] == FOLDS).all(1)]


def wil(a, b, metric="mcc", higher=True, data=None):
    X = (data if data is not None else M).loc[full]
    d = (X[a] - X[b]) if higher else (X[b] - X[a])            # positive: a is better
    p = float(stats.wilcoxon(X[a], X[b], zero_method="zsplit").pvalue) if (d.abs() > 1e-12).any() else 1.0
    return dict(a=a, b=b, metric=metric, mean_a=round(float(X[a].mean()), 4), mean_b=round(float(X[b].mean()), 4),
                delta=round(float(d.mean()), 4), median_delta=round(float(d.median()), 4),
                wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), ties=int((d.abs() <= 1e-9).sum()), p=round(p, 4))


primary = wil(*PRIMARY)
confirmed = primary["delta"] > 0 and primary["p"] < 0.05
sec = [wil("v2", c) for c in ("Ens+CW", "KAN (tuned)", "CatBoost")]
for s, a in zip(sec, holm([s["p"] for s in sec])): s["p_holm"] = round(float(a), 4)
other = [wil("v2", "EGVW-KAN", met, hb, ALL.groupby(["dataset", "method"])[met].mean().unstack())
         for met, hb in (("balanced_accuracy", True), ("auroc", True), ("brier", False))]

methods = [m for m in M.columns if (N.loc[full, m] == FOLDS).all()]
main_rows = ["v2", "EGVW-KAN", "Ens+CW", "Ens-uniform", "KAN (tuned)"]
comps = [m for m in methods if m in COMP.method.unique()]
desc = [wil("v2", c) for c in main_rows[1:] + sorted(comps, key=lambda m: -M.loc[full, m].mean())]
rank = M.loc[full, main_rows + comps].rank(axis=1, ascending=False).mean().round(2).sort_values().to_dict()
mean = M.loc[full, main_rows + comps].mean().round(4).sort_values(ascending=False).to_dict()

col = {}
for f in pred_files(STAGE2_KAN, "kan"):
    z = np.load(f, allow_pickle=True); ms = list(z["methods"])
    for m in ["v2", "EGVW-KAN", "Ens+CW", "Ens+CW+Boot+Guards", "KAN (tuned)"]:
        if m in ms: col.setdefault(m, []).append(collapsed(z["probs"][ms.index(m)]))
col = {m: round(float(np.mean(v)), 4) for m, v in col.items()}

save(dict(n_datasets=int(len(full)), datasets=list(full), units=cnt, primary=primary, confirmed=bool(confirmed),
          secondary=sec, primary_other_metrics=other, descriptive=desc, mean_mcc=mean, mean_rank=rank, collapse=col,
          per_dataset=M.loc[full, ["v2", "EGVW-KAN", "Ens+CW", "KAN (tuned)", "CatBoost"]].round(4).to_dict()), "stage2.json")
print("primary:", primary, "-> confirmed" if confirmed else "-> not confirmed"); print("collapse:", col)
