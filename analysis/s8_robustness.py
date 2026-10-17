"""Robustness checks: Nemenyi critical difference, Bayesian signed-rank test with Cliff's delta, and calibration (ECE).

1. Critical difference for the v1 benchmark (19 methods, 24 datasets) and Stage 2 (21 methods, 13 datasets).
2. Bayesian signed-rank test (Benavoli et al., 2017) with a region of practical equivalence of 0.01 MCC, and the
   paired Cliff's delta, for the key pairs of both stages.
3. Top-label expected calibration error (10 equal-width bins), averaged over datasets.
Everything is computed from the stored per-fold results and predictions; nothing is retrained.
"""
import os
import numpy as np, pandas as pd
from scipy import stats
from common import (results, pred_files, save, V1_KAN, V1_BASELINES, V1_VARIANTS, STAGE1, STAGE2_KAN, STAGE2_COMP)

P1 = results(*V1_KAN)
B1 = results(*V1_BASELINES, *V1_VARIANTS)
V2 = results(*STAGE1); V2 = V2[V2.method == "v2"]
T2 = results(*STAGE2_KAN, *STAGE2_COMP)
COMP = sorted(B1.method.unique())
M1 = pd.concat([P1, B1, V2]).groupby(["dataset", "method"]).mcc.mean().unstack()
M2 = T2.groupby(["dataset", "method"]).mcc.mean().unstack()
SETS = {"cd_v1": (M1, COMP + ["EGVW-KAN", "KAN (tuned)", "Ens-uniform"]),
        "cd_t2": (M2, COMP + ["EGVW-KAN", "KAN (tuned)", "Ens-uniform", "Ens+CW", "v2"])}
P = {}

for key, (M, methods) in SETS.items():
    R = M[methods].rank(axis=1, ascending=False).mean().sort_values()
    k, n = len(methods), len(M)
    cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    rs, cl = R.values, []                                   # largest groups whose rank range is below CD
    for i in range(len(rs)):
        j = i
        while j + 1 < len(rs) and rs[j + 1] - rs[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl): cl.append((i, j))
    best = R.min()
    P[key] = dict(k=k, n=n, cd=round(float(cd), 2), ranks=R.round(2).to_dict(), cliques=len(cl),
                  within_cd_of_best=[m for m in R.index if R[m] - best < cd],
                  sig_worse_than_best=[m for m in R.index if R[m] - best >= cd])


def bayes_signed_rank(x, y, rope=0.01, prior=0.5, nsamples=50000, seed=0):
    """Returns P(x better), P(practically equal), P(y better)."""
    z = np.concatenate([[0.0], np.asarray(x) - np.asarray(y)]); n = len(z)
    S = z[:, None] + z[None, :]
    regions = [(S < -2 * rope), ((S >= -2 * rope) & (S <= 2 * rope)), (S > 2 * rope)]
    W = np.random.default_rng(seed).dirichlet(np.r_[prior, np.ones(n - 1)], nsamples)
    mass = np.stack([np.einsum("si,ij,sj->s", W, A.astype(float), W) for A in regions], 1)
    win = mass.argmax(1)
    return [round(float((win == c).mean()), 3) for c in (2, 1, 0)]


def cliff(x, y):
    """Paired Cliff's delta: (datasets where x > y minus datasets where x < y) / n."""
    d = np.asarray(x) - np.asarray(y)
    return float(((d > 1e-9).sum() - (d < -1e-9).sum()) / len(d))


def size_label(d):
    a = abs(d); return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


P["bayes"] = []
for stage, M in (("Stage 1", M1), ("Stage 2", M2)):
    for a, b in (("v2", "EGVW-KAN"), ("v2", "Ens+CW"), ("v2", "KAN (tuned)"), ("v2", "CatBoost"), ("EGVW-KAN", "CatBoost")):
        X = M[[a, b]].dropna()
        pw, pe, pl = bayes_signed_rank(X[a], X[b]); c = cliff(X[a], X[b])
        P["bayes"].append(dict(stage=stage, a=a, b=b, n=int(len(X)), p_better=pw, p_rope=pe, p_worse=pl,
                               cliff=round(c, 3), cliff_label=size_label(c), delta=round(float((X[a] - X[b]).mean()), 4)))


def load_preds(files):
    out = {}
    for f in files:
        z = np.load(f, allow_pickle=True); key = os.path.basename(f).rsplit("__", 1)[0]
        order = np.argsort(z["te_idx"])
        for m, pr in zip(list(z["methods"]), z["probs"]):
            out.setdefault(m, {})[key] = (z["y"][order], pr[order])
    return out


def ece(y, Pm, bins=10):
    conf = Pm.max(1); acc = (Pm.argmax(1) == y).astype(float); e = 0.0
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    for b in range(bins):
        s = idx == b
        if s.any(): e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return e


CAL = ["EGVW-KAN", "v2", "Ens+CW", "CatBoost", "Logistic regression"]
P["ece"] = {}
for stage, files in (("Stage 1", pred_files(V1_KAN, "kan") + pred_files(STAGE1, "v2") + pred_files(V1_BASELINES, "baselines")),
                     ("Stage 2", pred_files(STAGE2_KAN, "kan") + pred_files(STAGE2_COMP, "baselines"))):
    pr = load_preds(files); P["ece"][stage] = {}
    for m in CAL:
        if m not in pr: continue
        per_ds = {}
        for key, (y, Pm) in pr[m].items(): per_ds.setdefault(key.split("__")[0], []).append(ece(y, Pm))
        P["ece"][stage][m] = round(float(np.mean([np.mean(v) for v in per_ds.values()])), 4)
save(P, "robustness.json")
print("CD:", P["cd_v1"]["cd"], P["cd_t2"]["cd"]); print("ECE:", P["ece"])
