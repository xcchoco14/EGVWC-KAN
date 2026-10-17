"""Quantitative checks for the Discussion section.

1. Propositions: flat members (P2), prior shift of the double correction (P4), log K bound of the OOB score (P5),
   shrinkage (P6).
2. Nested CV: inner-vs-outer optimism, stability of the selected hyperparameters, fold and repeat variability, rank
   stability across repeats, Nadeau-Bengio corrected intervals for v2 - v1 per dataset.
3. Collapse of v1 against the improvement of v2 per dataset.
Needs combined.json (s6).
"""
import json, os, re
import numpy as np, pandas as pd
from scipy import stats
from common import (OUT, results, info, pred_files, load, save, collapsed,
                    V1_KAN, V1_BASELINES, V1_VARIANTS, STAGE1, STAGE2_KAN)

NF = load("combined.json")
inc = [r for r in NF["v1"]["inclusion"] + NF["t2_inclusion"] if r["decision"] == "include"]
K_of = {r["dataset"]: int(r["K"]) for r in inc}
n_of = {r["dataset"]: int(r["n"]) for r in inc}
D = {}
P1 = results(*V1_KAN)
I1 = info(*V1_KAN)
I1 = I1[(I1.group == "kan") & I1["info"].apply(lambda d: isinstance(d, dict) and "params" in d)].drop_duplicates("key", keep="last")

# P2: folds that contain a flat member (D < 0.01 and L <= log K)
X = pd.read_csv(os.path.join(OUT, "v1_members.csv")); X["logK"] = np.log(X.dataset.map(K_of))
X["flat"] = (X.D < 0.01) & (X.L <= X.logK)
has_flat = X.groupby("key").flat.any(); flat_folds = has_flat[has_flat].index
top_flat = X.loc[X.groupby("key").w.idxmax()].set_index("key").flat
D["p2"] = dict(n_flat_members=int(X.flat.sum()), n_folds_with_flat=int(len(flat_folds)),
               flat_gets_max=round(float(top_flat.loc[flat_folds].mean()), 3),
               mean_w_flat=round(float(X[X.flat].w.mean()), 3), mean_w_other=round(float(X[~X.flat].w.mean()), 3))

# P4: log-odds shift between the single (H) and double (K) correction on binary datasets with a balanced bootstrap
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
p4 = []
for f in pred_files(V1_KAN, "kan"):
    z = np.load(f, allow_pickle=True); ms = list(z["methods"])
    if "EGVW-KAN double-CW" not in ms or z["probs"][ms.index("EGVW-KAN")].shape[1] != 2: continue
    ps, pd_ = z["probs"][ms.index("EGVW-KAN")][:, 1], z["probs"][ms.index("EGVW-KAN double-CW")][:, 1]
    if np.allclose(ps, pd_): continue                      # no bootstrap: row K equals row H
    y = z["y"]; pi1 = float(np.mean(y)); ds = os.path.basename(f).split("__")[0]
    p4.append(dict(ds=ds, theory=float(np.log((1 - pi1) / pi1)), obs_shift=float(np.median(lg(pd_) - lg(ps))),
                   pos_single=float((ps > .5).mean()), pos_double=float((pd_ > .5).mean()),
                   pos_pred=float((1 / (1 + np.exp(-(lg(ps) + np.log((1 - pi1) / pi1)))) > .5).mean()), prev=pi1))
P4 = pd.DataFrame(p4)
D["p4"] = dict(per_dataset=P4.groupby("ds").mean().round(3).to_dict(orient="index"), n_folds=int(len(P4)),
               spearman_shift_theory=round(float(stats.spearmanr(P4.theory, P4.obs_shift)[0]), 3),
               slope=round(float(np.polyfit(P4.theory, P4.obs_shift, 1)[0]), 3),
               pos_double=round(float(P4.pos_double.mean()), 3), pos_pred=round(float(P4.pos_pred.mean()), 3),
               pos_single=round(float(P4.pos_single.mean()), 3), prev=round(float(P4.prev.mean()), 3))

# P5, P6: every v2 member of both stages
rows = []
for runs, stage in ((STAGE1, "Stage 1"), (STAGE2_KAN, "Stage 2")):
    for r in info(*runs).itertuples():
        inf = r.info if isinstance(r.info, dict) else {}
        v = inf.get("v2", inf) if stage == "Stage 2" else inf
        if not isinstance(v, dict) or "oob_balanced_nll" not in v: continue
        n_o = float(np.median(v["n_oob"])); ge = 1.0 * n_o / (n_o + 100)
        w = np.asarray(v["weights"]["v2"]); H = float(-(w[w > 0] * np.log(w[w > 0])).sum() / np.log(len(w)))
        for b, gt in zip(v["oob_balanced_nll"], v["gated"]):
            rows.append(dict(stage=stage, key=r.key, dataset=r.dataset, B=b, gated=gt, logK=np.log(K_of[r.dataset]),
                             n_oob=n_o, gamma_eff=ge, entropy=H))
V = pd.DataFrame(rows).replace([np.inf], np.nan)
F = V.drop_duplicates("key")
D["p5"] = dict(n_members=int(len(V)), n_gated=int(V.gated.sum()),
               gated_B_ge_logK=round(float((V[V.gated].B >= V[V.gated].logK).mean()), 3) if V.gated.any() else None,
               share_B_below_logK=round(float((V.B < V.logK).mean()), 3),
               median_B_over_logK=round(float((V.B / V.logK).median()), 3))
small = F.dataset.map(n_of) < 200
D["p6"] = dict(gamma_eff_range=[round(float(F.gamma_eff.min()), 3), round(float(F.gamma_eff.max()), 3)],
               gamma_eff_small=round(float(F[small].gamma_eff.median()), 3), gamma_eff_large=round(float(F[~small].gamma_eff.median()), 3),
               entropy_small=round(float(F[small].entropy.median()), 3), entropy_large=round(float(F[~small].entropy.median()), 3),
               spearman_gamma_entropy=round(float(stats.spearmanr(F.gamma_eff, F.entropy)[0]), 3),
               v1_entropy_median=round(float(np.median([d["weight_entropy"] for d in I1["info"]])), 3))

# optimism: best inner-CV MCC of the tuned KAN against its MCC on the outer test fold
o = []
outer = P1[P1.method == "KAN (tuned)"].set_index("key").mcc
for r in I1.itertuples():
    if not isinstance(r.search, list) or r.key not in outer.index: continue
    sc = [c["inner_mcc"] for c in r.search]
    o.append(dict(dataset=r.dataset, best_inner=max(sc), mean_inner=float(np.mean(sc)), outer=float(outer.loc[r.key])))
O = pd.DataFrame(o); O["gap"] = O.best_inner - O.outer; O["sel"] = O.best_inner - O.mean_inner
Og = O.groupby("dataset")[["best_inner", "outer", "gap", "sel"]].mean(); Og["n"] = Og.index.map(n_of)
D["optimism"] = dict(n_units=int(len(O)), best_inner=round(float(O.best_inner.mean()), 4), outer=round(float(O.outer.mean()), 4),
                     gap=round(float(O.gap.mean()), 4), sel=round(float(O.sel.mean()), 4),
                     p=float(stats.wilcoxon(Og.best_inner, Og.outer).pvalue), datasets_positive=int((Og.gap > 0).sum()),
                     n_datasets=int(len(Og)), spearman_n_gap=round(float(stats.spearmanr(Og.n, Og.gap)[0]), 3),
                     largest=Og.gap.sort_values(ascending=False).head(3).round(3).to_dict())

# stability of the selected hyperparameters over the 15 outer folds
cfg = I1.assign(c=I1["info"].apply(lambda d: json.dumps(d["params"], sort_keys=True)))
st = cfg.groupby("dataset").c.agg(lambda s: (s.nunique(), s.value_counts().iloc[0] / len(s)))
D["hp_stability"] = dict(median_distinct=float(np.median([a for a, _ in st])), max_distinct=int(max(a for a, _ in st)),
                         median_modal_share=round(float(np.median([b for _, b in st])), 3),
                         datasets_one_config=int(sum(a == 1 for a, _ in st)))

# variability between folds and between repeats; rank stability of the 19 v1 methods across repeats
B = results(*V1_BASELINES, *V1_VARIANTS)
V2 = results(*STAGE1)
ALL = pd.concat([P1, B, V2[V2.method == "v2"]])
D["variability"] = {m: dict(fold_sd=round(float(ALL[ALL.method == m].groupby("dataset").mcc.std().mean()), 4),
                            repeat_sd=round(float(ALL[ALL.method == m].groupby(["dataset", "repeat"]).mcc.mean()
                                                  .groupby("dataset").std().mean()), 4))
                    for m in ("v2", "EGVW-KAN", "Ens+CW", "CatBoost", "Logistic regression")}
R = ALL[ALL.method.isin(list(NF["v1"]["p1a_vs_all"]["mean"]))].groupby(["repeat", "method"]).mcc.mean().unstack()
D["rank_stability"] = dict(r01=round(float(stats.spearmanr(R.loc[0], R.loc[1])[0]), 3),
                           r02=round(float(stats.spearmanr(R.loc[0], R.loc[2])[0]), 3),
                           r12=round(float(stats.spearmanr(R.loc[1], R.loc[2])[0]), 3),
                           v1_rank_by_repeat=[int(R.loc[r].rank(ascending=False)["EGVW-KAN"]) for r in (0, 1, 2)],
                           n_methods=int(R.shape[1]))

# Nadeau-Bengio corrected 95% interval of v2 - v1 per dataset
T2 = results(*STAGE2_KAN)
nb = {}
for stage, Rr in (("Stage 1", pd.concat([P1, V2])), ("Stage 2", T2)):
    W = Rr[Rr.method.isin(["v2", "EGVW-KAN"])].pivot_table(index=["dataset", "repeat", "fold"], columns="method", values="mcc")
    nt = Rr.drop_duplicates(["dataset", "repeat", "fold"]).set_index(["dataset", "repeat", "fold"])[["n_train", "n_test"]]
    W = W.join(nt); W["d"] = W["v2"] - W["EGVW-KAN"]
    up = dn = 0; per = []
    for ds, G in W.groupby(level=0):
        J = len(G); s2 = G.d.var(ddof=1); ratio = float((G.n_test / G.n_train).mean())
        se = np.sqrt((1 / J + ratio) * s2) if s2 > 0 else 0.0
        t = stats.t.ppf(0.975, J - 1); lo, hi = G.d.mean() - t * se, G.d.mean() + t * se
        up += lo > 0; dn += hi < 0; per.append(dict(dataset=ds, mean=G.d.mean(), lo=lo, hi=hi))
    nb[stage] = dict(n=int(W.index.get_level_values(0).nunique()), ci_above_0=int(up), ci_below_0=int(dn),
                     per_dataset=pd.DataFrame(per).set_index("dataset").round(3).to_dict(orient="index"))
D["nadeau_bengio"] = nb


# collapse of v1 per dataset against the gain of v2
def collapse_by_ds(files, method):
    out = {}
    for f in files:
        z = np.load(f, allow_pickle=True); ms = list(z["methods"])
        if method in ms: out.setdefault(os.path.basename(f).split("__")[0], []).append(collapsed(z["probs"][ms.index(method)]))
    return {k: float(np.mean(v)) for k, v in out.items()}


cv = []
for stage, runs, per in (("Stage 1", V1_KAN, NF["t1"]["per_dataset"]), ("Stage 2", STAGE2_KAN, NF["t2"]["per_dataset"])):
    c = collapse_by_ds(pred_files(runs, "kan"), "EGVW-KAN")
    for ds in per["v2"]:
        cv.append(dict(stage=stage, dataset=ds, collapse_v1=c.get(re.sub(r"[^A-Za-z0-9]+", "_", ds).strip("_"), np.nan),
                       delta=per["v2"][ds] - per["EGVW-KAN"][ds]))
C = pd.DataFrame(cv).dropna(); withc = C.collapse_v1 > 0
D["collapse_vs_gain"] = dict(spearman=round(float(stats.spearmanr(C.collapse_v1, C.delta)[0]), 3),
                             n_with_collapse=int(withc.sum()), n_total=int(len(C)),
                             gain_with=round(float(C[withc].delta.mean()), 4), gain_without=round(float(C[~withc].delta.mean()), 4),
                             share_of_positive_gain=round(float(C[withc & (C.delta > 0)].delta.sum() / C[C.delta > 0].delta.sum()), 3))

# Myocardial Infarction: single (H) vs double (K) correction on the same 15 folds
mi = P1[P1.dataset.str.startswith("Myocardial")].groupby("method").mcc.mean()
mp = D["p4"]["per_dataset"]["Myocardial_Infarction_Complications_AF"]
D["mi_final"] = dict(pos_single=mp["pos_single"], pos_double=mp["pos_double"], prev=mp["prev"],
                     mcc_single=round(float(mi["EGVW-KAN"]), 3), mcc_double=round(float(mi["EGVW-KAN double-CW"]), 3))
save(D, "discussion.json")
print("optimism gap:", D["optimism"]["gap"], "| collapse vs gain:", D["collapse_vs_gain"])
