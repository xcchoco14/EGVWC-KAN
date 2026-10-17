# ═════════════════════════════════════════════════════════════════════════════
# CELL 7c — PART 5: EGVWC-KAN v2 (development stage, see preregistration_v2.md)
# Diagnosis from v1 (Parts 1/4): the over-fitting guard (λ × 5) produces flat members on small datasets; a flat
# member has near-zero logit dispersion, so the v1 score (validation loss + dispersion) gives it the LARGEST weight;
# and the score is computed on a 20 % internal split (≈ 20 rows on the smallest datasets). v2 changes exactly this:
#   members       5 KANs, each trained on a stratified bootstrap of the outer TRAINING fold (group bootstrap when
#                 the dataset has patient/duplicate groups) with the full imbalance-tiered class weights.
#                 No balanced bootstrap (it leaves almost no out-of-bag minority rows), no fitting guards, no refit.
#   score         balanced log-loss of each member on ITS OWN out-of-bag (OOB) rows (mean of per-class mean NLL).
#   gate          a member that predicts a single class on its OOB rows (or failed to train) gets weight 0.
#   shrinkage     γ_eff = γ · n_oob / (n_oob + C_SHRINK): small OOB sets → weights close to uniform.
#   weights       w_m ∝ exp(−γ_eff · (s_m / min s − 1)) over the non-gated members; all gated → uniform.
# Same folds, same tuned KAN configuration (read from Part 1), same member seeds as v1. The outer test fold is used
# once, for prediction. OOB rows are training-fold rows.
# Rows (all from the SAME five members, so differences isolate the weighting):
#   v2                     gate + shrinkage + OOB balanced loss                        (the candidate)
#   v2 − gate              no gate
#   v2 − shrink            γ_eff = γ
#   v2 uniform             bagged ensemble, uniform weights
#   v2 uniform + gate      uniform weights over non-gated members
#   v2 [v1 score]          v1 score (class-weighted loss + logit dispersion) on the OOB rows, no gate / shrinkage
# Post-hoc sensitivity (no retraining): γ ∈ V2_GAMMA_GRID × C_SHRINK ∈ V2_SHRINK_GRID, with the gate.
# ═════════════════════════════════════════════════════════════════════════════
import time
import numpy as np

V2_METHODS = ["v2", "v2 − gate", "v2 − shrink", "v2 uniform", "v2 uniform + gate", "v2 [v1 score]"]


def v2_bootstrap(y, groups, seed):
    """Stratified bootstrap of the training rows (each class resampled to its own size); with groups, whole
    groups are resampled so that rows of one patient are never split between in-bag and out-of-bag."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True) for c in np.unique(y)])
    else:
        ug = np.unique(groups); pick = rng.choice(ug, size=len(ug), replace=True)
        pos = {g: np.where(groups == g)[0] for g in ug}
        idx = np.concatenate([pos[g] for g in pick])
    oob = np.setdiff1d(np.arange(len(y)), idx)
    return idx, oob


def balanced_nll(logits, y, K):
    """Mean over the classes present of the mean negative log-likelihood within the class."""
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    nll = -logp[np.arange(len(y)), y]
    return float(np.mean([nll[y == c].mean() for c in np.unique(y)]))


def v2_weights(score, gated, n_oob, gamma, c_shrink, use_gate=True, shrink=True):
    s = np.asarray(score, float); keep = ~np.asarray(gated) if use_gate else np.ones(len(s), bool)
    if not keep.any(): return np.full(len(s), 1 / len(s))
    g = gamma * (n_oob / (n_oob + c_shrink) if shrink else 1.0)
    w = np.zeros(len(s)); ref = max(s[keep].min(), 1e-8)
    w[keep] = np.exp(-g * (s[keep] / ref - 1.0))
    return w / w.sum()


def run_v2(bundle, tr_idx, te_idx, seed, params=None):
    """One outer fold → probabilities for the v2 rows. `params` = the configuration Part 1 selected on this fold."""
    _torch_single_thread()
    K = bundle.K
    Xtr, ytr, Xte, yte = fold_arrays(bundle, tr_idx, te_idx)
    gtr = None if bundle.groups is None else bundle.groups[tr_idx]
    info, times, probs = {}, {}, {}
    t_search = 0.0
    if params is None:
        t0 = time.time(); p, _ = kan_search(bundle, tr_idx, seed); t_search = time.time() - t0
        info["params_source"] = "searched here (no Part 1 output)"
    else:
        p = {**params, "_K": K}; info["params_source"] = "Part 1"
    info["params"] = {k: v for k, v in p.items() if k != "_K"}
    cw = class_weights(ytr, K, "tiered")                      # full tiered exponent (no balanced bootstrap)
    t0 = time.time()
    Lte, bal, v1s, gated, n_oob, fails = [], [], [], [], [], 0
    for m in range(CONFIG["N_MEMBERS"]):
        s = seed * 100 + m
        idx, oob = v2_bootstrap(ytr, gtr, s)
        mod = train_kan(Xtr[idx], ytr[idx], p, s, cw); fails += mod is None
        zo = kan_logits(mod, Xtr[oob], K); yo = ytr[oob]
        Lte.append(kan_logits(mod, Xte, K))
        bal.append(balanced_nll(zo, yo, K) if len(yo) else np.inf)
        v1s.append((ce(zo, yo, cw) + float(np.var(zo, axis=1).mean())) if len(yo) else np.inf)
        one_class = len(np.unique(yo)) > 1 and len(np.unique(zo.argmax(1))) == 1
        gated.append(bool(mod is None or one_class or not len(yo))); n_oob.append(len(yo))
    t_fit = time.time() - t0 + t_search
    n = float(np.median(n_oob)); G, C = CONFIG["GAMMA"], CONFIG["V2_C_SHRINK"]
    W = {"v2": v2_weights(bal, gated, n, G, C),
         "v2 − gate": v2_weights(bal, gated, n, G, C, use_gate=False),
         "v2 − shrink": v2_weights(bal, gated, n, G, C, shrink=False),
         "v2 uniform": np.full(len(bal), 1 / len(bal)),
         "v2 uniform + gate": v2_weights(np.ones(len(bal)), gated, n, 0.0, C),
         "v2 [v1 score]": member_weights(v1s, np.zeros(len(v1s)), "loss", G)}
    for name, w in W.items():
        probs[name] = ens_proba(Lte, w); times[name] = t_fit
    sens = {}
    for gam in CONFIG["V2_GAMMA_GRID"]:
        for c in CONFIG["V2_SHRINK_GRID"]:
            if (gam, c) != (G, C):
                sens[f"v2 [γ={gam}, C={c}]"] = ens_proba(Lte, v2_weights(bal, gated, n, gam, c))
    info.update(weights={k: v.tolist() for k, v in W.items()}, oob_balanced_nll=bal, v1_score=v1s,
                gated=gated, n_oob=n_oob, n_gated=int(sum(gated)), train_failures_total=int(fails))
    return dict(yte=yte, probs=probs, sens=sens, times=times, info=info, search=None)


def analyse_v2():
    """Stage 4 for Part 5: v2 rows vs EGVW-KAN (H) and Ens+CW (C) of Part 1 on identical folds (complete datasets)."""
    import pandas as pd
    from scipy import stats
    R = pd.read_json(F_RES, lines=True)
    ref = ["EGVW-KAN", "Ens+CW", "Ens-uniform", "KAN (tuned)"]
    S = R[R.method.isin(V2_METHODS + ref)]
    n = S.groupby(["dataset", "method"]).size().unstack()
    full = n.index[(n.reindex(columns=V2_METHODS + ref) == CONFIG["REPEATS"] * CONFIG["OUTER_FOLDS"]).all(1)]
    M = S[S.dataset.isin(full)].groupby(["dataset", "method"]).mcc.mean().unstack()
    print(f"  {len(full)} datasets complete for every row")
    if len(full) < 2: return M
    rows = []
    for m in V2_METHODS:
        for r in ("EGVW-KAN", "Ens+CW"):
            d = M[m] - M[r]
            p = stats.wilcoxon(M[m], M[r], zero_method="zsplit").pvalue if (d != 0).any() else 1.0
            rows.append(dict(row=m, vs=r, mcc=M[m].mean(), ref_mcc=M[r].mean(), delta=d.mean(),
                             wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), p=p))
    T = pd.DataFrame(rows).round(4)
    T.to_csv(os.path.join(CONFIG["SAVE_DIR"], "v2_vs_v1.csv"), index=False)
    M.round(4).to_csv(os.path.join(CONFIG["SAVE_DIR"], "v2_mcc_by_dataset.csv"))
    print(T.to_string(index=False))
    return T
