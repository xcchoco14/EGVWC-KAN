"""Diagnosis of the v1 weighting on all 1,800 ensemble members, and the combined summary used by s7-s9.

Reads the member logs of Part 1 (weights, validation loss and logit dispersion of every member) and the outputs of
s1-s5. Writes combined.json and v1_members.csv.
"""
import os
import numpy as np, pandas as pd
from scipy import stats
from common import OUT, info, inclusion, load, save, V1_KAN

N = {"v1": load("v1_benchmark.json"), "v1_prereg": load("v1_preregistered_test.json"), "interp": load("interpretability.json"),
     "t1": load("stage1.json"), "t2": load("stage2.json")}

# one row per member: 24 datasets x 15 folds x 5 members
I = info(*V1_KAN)
I = I[(I.group == "kan") & I["info"].apply(lambda d: isinstance(d, dict) and "egvw_weights" in d)].drop_duplicates("key", keep="last")
rows = []
for r in I.itertuples():
    d = r.info; w, L, D = (np.asarray(d[k], float) for k in ("egvw_weights", "val_loss", "val_disp"))
    for m in range(len(w)):
        rows.append(dict(key=r.key, dataset=r.dataset, w=w[m], L=L[m], D=D[m], guard=d["guards"][m]))
X = pd.DataFrame(rows); X.to_csv(os.path.join(OUT, "v1_members.csv"), index=False)

top = X.loc[X.groupby("key").w.idxmax()].set_index("key")
lowD = X.loc[X.groupby("key").D.idxmin()].set_index("key"); lowL = X.loc[X.groupby("key").L.idxmin()].set_index("key")
sp = lambda a, b: float(X.groupby("key").apply(lambda g: stats.spearmanr(g[a], g[b])[0] if g[a].std() > 0 and g[b].std() > 0 else np.nan).mean())
K_of = {r["dataset"]: int(r["K"]) for r in N["v1"]["inclusion"] if r["decision"] == "include"}
X["logK"] = np.log(X.dataset.map(K_of))
N["diag"] = dict(
    n_members=int(len(X)), n_folds=int(X.key.nunique()),
    share_D=round(float((X.D / (X.L + X.D)).median()), 3),          # share of the dispersion in the score s = L + D
    spearman_wD=round(sp("w", "D"), 3), spearman_wL=round(sp("w", "L"), 3),
    top_is_lowD=round(float((top.D == lowD.D).mean()), 3), top_is_lowL=round(float((top.L == lowL.L).mean()), 3),
    D_median=round(float(X.D.median()), 3), D_gt_logK=round(float((X.D > X.logK).mean()), 3),
    guard_counts=X.guard.value_counts().to_dict(),
    by_guard={g: dict(D=round(float(G.D.median()), 3), L=round(float(G.L.median()), 3), w=round(float(G.w.mean()), 3), n=int(len(G)))
              for g, G in X.groupby("guard")},
    folds_with_overfit_guard=round(float(X.groupby("key").guard.apply(lambda s: (s == "overfit").any()).mean()), 3),
)

N["t2_inclusion"] = inclusion("v2_stage2_comp").fillna("").to_dict("records")

# per-dataset difference v2 - v1 against dataset size, both stages
n1 = {r["dataset"]: int(r["n"]) for r in N["v1"]["inclusion"] if r["decision"] == "include"}
n2 = {r["dataset"]: int(r["n"]) for r in N["t2_inclusion"] if r["decision"] == "include"}
eff = [dict(stage=stage, dataset=ds, n=nn.get(ds), delta=round(v - R["EGVW-KAN"][ds], 4))
       for stage, R, nn in (("Stage 1", N["t1"]["per_dataset"], n1), ("Stage 2", N["t2"]["per_dataset"], n2))
       for ds, v in R["v2"].items()]
N["effects"] = eff
E = pd.DataFrame(eff)
N["effect_size_corr"] = {s: round(float(stats.spearmanr(G.n, G.delta)[0]), 3) for s, G in E.groupby("stage")}
save(N, "combined.json")
print("diagnosis:", {k: N["diag"][k] for k in ("share_D", "spearman_wD", "spearman_wL", "top_is_lowD")})
