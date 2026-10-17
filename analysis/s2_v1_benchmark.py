"""v1 benchmark on 24 datasets: comparators, ablation ladder, collapse rates and Part 4 (imbalance alternatives).

Needs s1_v1_preregistered_test.py to have run first (its result is copied into the output).
"""
import numpy as np, pandas as pd
from scipy import stats
from common import (results, inclusion, pred_files, load, save, holm, collapsed,
                    V1_KAN, V1_BASELINES, V1_VARIANTS, V1_RESAMPLING)

N = {"inclusion": inclusion("v1_part1_A").fillna("").to_dict("records")}

# comparators: 8 baselines and 8 KAN variants, 24 datasets x 15 folds
B = results(*V1_BASELINES); V = results(*V1_VARIANTS)
C = pd.concat([B.assign(family="baseline"), V.assign(family="kan_variant")])
assert C.groupby(["dataset", "method"]).size().eq(15).all()
summ = {}
for m in ["mcc", "balanced_accuracy", "auroc", "auprc", "brier", "f1", "accuracy"]:
    M = C.groupby(["dataset", "method"])[m].mean().unstack()
    summ[m] = dict(mean=M.mean().round(4).to_dict(), rank=M.rank(axis=1, ascending=(m == "brier")).mean().round(2).to_dict())
N["comp_summary"] = summ
N["comp_time"] = C.groupby("method").time_sec.median().round(1).to_dict()

# ablation ladder, each row against its reference row
P1 = pd.concat([results("v1_part1_A").assign(shard="A"), results("v1_part1_B").assign(shard="B")])
cells = P1[["dataset", "repeat", "fold"]].drop_duplicates()
LADDER = ["KAN (tuned)", "Ens-uniform", "Ens+CW", "Ens+CW+Boot", "Ens+CW+Boot+Guards", "EGVW loss-only",
          "EGVW dispersion-only", "EGVW-KAN", "EGVW-KAN fixed-CW", "EGVW-KAN no-refit", "EGVW-KAN double-CW"]
L = P1[P1.method.isin(LADDER)]
N["p1a_ladder"] = {m: L.groupby(["dataset", "method"])[m].mean().unstack()[LADDER].mean().round(4).to_dict()
                   for m in ["mcc", "balanced_accuracy", "auroc", "brier"]}
T = L.groupby(["dataset", "method"]).mcc.mean().unstack()[LADDER]
REF = {"Ens-uniform": "KAN (tuned)", "Ens+CW": "Ens-uniform", "Ens+CW+Boot": "Ens+CW", "Ens+CW+Boot+Guards": "Ens+CW+Boot",
       "EGVW loss-only": "Ens+CW+Boot+Guards", "EGVW dispersion-only": "Ens+CW+Boot+Guards", "EGVW-KAN": "Ens+CW+Boot+Guards",
       "EGVW-KAN fixed-CW": "EGVW-KAN", "EGVW-KAN no-refit": "EGVW-KAN", "EGVW-KAN double-CW": "EGVW-KAN"}
steps = []
for m, r in REF.items():
    d = T[m] - T[r]
    try: p = float(stats.wilcoxon(T[m], T[r], zero_method="zsplit").pvalue)
    except ValueError: p = 1.0
    steps.append(dict(row=m, ref=r, delta=round(float(d.mean()), 4), wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), p=round(p, 3)))
N["p1a_steps"] = steps

# all 19 methods on the same folds: Friedman, Nemenyi CD, Holm-corrected Wilcoxon against v1
key = lambda D: D.merge(cells, on=["dataset", "repeat", "fold"])
S = pd.concat([key(B), key(V), P1[P1.method.isin(["EGVW-KAN", "KAN (tuned)", "Ens-uniform"])]])
h = S.drop_duplicates(["dataset", "repeat", "fold", "fold_hash"]).groupby(["dataset", "repeat", "fold"]).fold_hash.nunique()
assert (h == 1).all(), "fold hashes differ"
M = S.groupby(["dataset", "method"]).mcc.mean().unstack()
fr = stats.friedmanchisquare(*[M[c] for c in M.columns]); k, n = M.shape[1], M.shape[0]
cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
N["final_friedman"] = dict(chi2=round(float(fr.statistic), 2), p=float(fr.pvalue), cd=round(float(cd), 2), k=k, n=n)
others = [c for c in M.columns if c != "EGVW-KAN"]
raw = [float(stats.wilcoxon(M["EGVW-KAN"], M[c], zero_method="zsplit").pvalue) for c in others]
N["final_holm"] = {c: dict(delta=round(float((M["EGVW-KAN"] - M[c]).mean()), 4), p=round(p, 4), p_holm=round(float(a), 4),
                           wins=int((M["EGVW-KAN"] - M[c] > 1e-9).sum()), losses=int((M["EGVW-KAN"] - M[c] < -1e-9).sum()))
                   for c, p, a in zip(others, raw, holm(raw))}
N["p1_time_min"] = int(round(P1[P1.method == "EGVW-KAN"].time_sec.median() / 60))
N["p1a_vs_all"] = dict(mean=M.mean().round(4).to_dict(), rank=M.rank(axis=1, ascending=False).mean().round(2).to_dict(),
                       per_dataset=M.round(3).to_dict())

# share of folds in which every test row gets the same class
col = {}
for f in pred_files(V1_KAN, "kan"):
    z = np.load(f, allow_pickle=True); ms = list(z["methods"])
    for m in ["KAN (tuned)", "Ens-uniform", "Ens+CW", "Ens+CW+Boot", "Ens+CW+Boot+Guards", "EGVW-KAN"]:
        if m in ms: col.setdefault(m, []).append(collapsed(z["probs"][ms.index(m)]))
N["collapse"] = {m: round(float(np.mean(v)), 3) for m, v in col.items()}
N["collapse_n"] = len(col["EGVW-KAN"])

# Part 4: the imbalance alternatives on the same pipeline, then each one against the tiered correction (Holm over 4)
ROWS = ["EGVW-KAN", "EGVW-KAN + SMOTE-NC", "EGVW-KAN + RUS", "EGVW-KAN (no correction)", "EGVW-KAN (no correction) + prior threshold"]
p4 = {}
for sh, run in zip("AB", V1_RESAMPLING):
    R = results(run)
    c = R[R.method == "EGVW-KAN + RUS"][["dataset", "repeat", "fold"]].drop_duplicates()
    S4 = R.merge(c, on=["dataset", "repeat", "fold"]); S4 = S4[S4.method.isin(ROWS)]
    p4[sh] = dict(n_datasets=int(c.dataset.nunique()), n_folds=int(len(c)),
                  **{m: S4.groupby(["dataset", "method"])[m].mean().unstack()[ROWS].mean().round(4).to_dict()
                     for m in ["mcc", "balanced_accuracy", "brier", "sensitivity", "specificity"]})
    p4[sh]["per_dataset_mcc"] = S4.groupby(["dataset", "method"]).mcc.mean().unstack()[ROWS].round(3).to_dict()
N["p4"] = p4
X4 = pd.concat([pd.DataFrame(p4[s]["per_dataset_mcc"]) for s in "AB"])
raw4 = [float(stats.wilcoxon(X4[m], X4["EGVW-KAN"], zero_method="zsplit").pvalue) for m in ROWS[1:]]
N["p4_tests"] = {m: dict(delta=round(float((X4[m] - X4["EGVW-KAN"]).mean()), 4), p=round(p, 4), p_holm=round(float(a), 4),
                         wins=int((X4[m] - X4["EGVW-KAN"] > 1e-9).sum()), losses=int((X4[m] - X4["EGVW-KAN"] < -1e-9).sum()))
                 for m, p, a in zip(ROWS[1:], raw4, holm(raw4))}
N["prereg"] = load("v1_preregistered_test.json")
save(N, "v1_benchmark.json")
print("Friedman:", N["final_friedman"]); print("collapse:", N["collapse"])
