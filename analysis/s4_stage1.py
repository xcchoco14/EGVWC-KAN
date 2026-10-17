"""Stage 1 (development) analysis as fixed in preregistration/preregistration_v2.md, section 3.

1. Collapse rate of the v2 rows against EGVW-KAN v1 (row H), and the number of gated members.
2. v2 vs H and v2 vs C (Ens+CW): mean MCC per dataset, two-sided Wilcoxon, datasets with all folds.
3. Freeze rule: best mean MCC among v2, v2 - gate, v2 - shrink and the 15 (gamma, C) settings; the default is kept
   unless the best one gains at least 0.005.
4. Go rule for Stage 2: v2-final beats H with p < 0.05.
Also reported (exploratory): v2-final against all 18 comparators of the v1 benchmark.
"""
import numpy as np, pandas as pd
from scipy import stats
from common import results, info, pred_files, save, collapsed, V1_KAN, V1_BASELINES, V1_VARIANTS, STAGE1

V2 = results(*STAGE1); V2 = V2[V2.key.str.endswith("|v2")]
P1 = results(*V1_KAN)
B = results(*V1_BASELINES, *V1_VARIANTS)
FOLDS = 15

# every v2 unit must have the same test-fold hash as the Part 1 unit it is compared with
h1 = P1.drop_duplicates(["dataset", "repeat", "fold"]).set_index(["dataset", "repeat", "fold"]).fold_hash
h2 = V2.drop_duplicates(["dataset", "repeat", "fold"]).set_index(["dataset", "repeat", "fold"]).fold_hash
assert (h2 == h1.reindex(h2.index)).all(), "fold hashes differ between Stage 1 and Part 1"

ALL = pd.concat([V2, P1, B])
M = ALL.groupby(["dataset", "method"]).mcc.mean().unstack()
n = ALL.groupby(["dataset", "method"]).size().unstack()
core = ["v2", "v2 − gate", "v2 − shrink", "v2 uniform", "v2 uniform + gate", "v2 [v1 score]", "EGVW-KAN", "Ens+CW"]
full = n.index[(n[core] == FOLDS).all(1)]
M = M.loc[full]


def wil(a, b):
    d = M[a] - M[b]
    p = float(stats.wilcoxon(M[a], M[b], zero_method="zsplit").pvalue) if (d.abs() > 1e-12).any() else 1.0
    return dict(a=a, b=b, mean_a=round(float(M[a].mean()), 4), mean_b=round(float(M[b].mean()), 4), delta=round(float(d.mean()), 4),
                wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), ties=int((d.abs() <= 1e-9).sum()), p=round(p, 4))


def collapse(files, methods):
    out = {m: [] for m in methods}
    for f in files:
        z = np.load(f, allow_pickle=True); ms = list(z["methods"])
        for m in methods:
            if m in ms: out[m].append(collapsed(z["probs"][ms.index(m)]))
    return {m: round(float(np.mean(v)), 4) for m, v in out.items() if v}


col = {**collapse(pred_files(STAGE1, "v2"), core[:6]), **collapse(pred_files(V1_KAN, "kan"), ["EGVW-KAN", "Ens+CW"])}
I = info(*STAGE1); I = I[I.group == "v2"]
ng = pd.Series([i["n_gated"] for i in I["info"]])
gated = dict(folds_with_gate=int((ng > 0).sum()), of=int(len(ng)), members_gated=int(ng.sum()), of_members=int(5 * len(ng)),
             all_gated=int((ng == 5).sum()))

dev = [wil("v2", "EGVW-KAN"), wil("v2", "Ens+CW")]
abl = [wil(m, "v2") for m in core[1:6]]

cands = ["v2", "v2 − gate", "v2 − shrink"] + [c for c in M.columns if c.startswith("v2 [γ=")]
means = M[cands].mean().sort_values(ascending=False)
best = means.index[0]
final = best if means[best] - means["v2"] >= 0.005 else "v2"
go = wil(final, "EGVW-KAN"); proceed = go["delta"] > 0 and go["p"] < 0.05

comps = ["KAN (tuned)", "Ens-uniform", "CatBoost", "Random forest", "Logistic regression", "XGBoost", "LightGBM", "SVM (RBF)",
         "KNN", "MLP", "FastKAN", "EfficientKAN", "SKAN", "BSRBF-KAN", "FasterKAN", "WavKAN", "FC-KAN", "GottliebKAN"]
expl = [wil(final, c) for c in comps]
rank = M[comps + ["EGVW-KAN", "Ens+CW", final]].rank(axis=1, ascending=False).mean().round(2).to_dict()

save(dict(n_datasets=int(len(full)), collapse=col, gated=gated, development=dev, ablation_vs_v2=abl,
          freeze=dict(candidates_mean=means.round(4).to_dict(), best=best, v2_final=final), go_rule=go,
          proceed_to_stage2=bool(proceed), exploratory_vs_comparators=expl, mean_rank_21=rank,
          per_dataset=M[["v2", final, "EGVW-KAN", "Ens+CW", "CatBoost"]].round(4).to_dict()), "stage1.json")
print("collapse:", col); print("v2 vs v1:", dev[0]); print("go rule met:", proceed)
