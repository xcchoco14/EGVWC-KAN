"""Result figures of the paper (PNG and PDF in analysis/figures).

fig_weight_diagnosis  v1 member weight against logit dispersion and against validation loss
fig_collapse          share of collapsed folds per ladder row, Stage 1 and Stage 2
fig_effect_size       MCC difference v2 - v1 per dataset against dataset size
fig_cd_v1, fig_cd_t2  critical difference diagrams
fig_calibration       reliability diagrams pooled over all folds
Needs combined.json (s6) and robustness.json (s8).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
from common import (OUT, FIG, results, pred_files, load, V1_KAN, V1_BASELINES, V1_VARIANTS, STAGE1, STAGE2_KAN, STAGE2_COMP)

N = load("combined.json"); PR = load("robustness.json")
X = pd.read_csv(os.path.join(OUT, "v1_members.csv"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
NAME = {"EGVW-KAN": "EGVWC-KAN v1", "v2": "EGVWC-KAN v2", "Ens+CW": "Ens+CW (C)", "KAN (tuned)": "Single KAN (A)",
        "Ens-uniform": "Uniform ensemble (B)"}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig); print("saved", name)


# weights against dispersion and loss
COL = {"none": "#4C78A8", "overfit": "#E45756", "underfit": "#F2A541"}
LAB = {"none": "no guard", "overfit": "overfitting guard (λ × 5)", "underfit": "underfitting guard"}
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
for g in ("none", "overfit", "underfit"):
    G = X[X.guard == g]
    ax[0].scatter(np.clip(G.D, 1e-3, None), G.w, s=6, alpha=0.35, color=COL[g], label=f"{LAB[g]} (n = {len(G)})", edgecolor="none")
    ax[1].scatter(G.L, G.w, s=6, alpha=0.35, color=COL[g], edgecolor="none")
for a in ax: a.axhline(0.2, color="#555", lw=0.8, ls="--"); a.set_ylabel("member weight wₘ")
ax[0].set_xscale("log"); ax[0].set_xlabel("logit dispersion Dₘ (log scale)")
ax[0].set_title(f"(a) weight vs dispersion, within-fold Spearman = {N['diag']['spearman_wD']:.2f}", fontsize=9, loc="left")
ax[1].set_xlabel("validation loss Lₘ")
ax[1].set_title(f"(b) weight vs loss, within-fold Spearman = {N['diag']['spearman_wL']:.2f}", fontsize=9, loc="left")
ax[1].axvline(np.log(2), color="#999", lw=0.8, ls=":"); ax[1].text(np.log(2), ax[1].get_ylim()[1] * 0.97, " ln 2", fontsize=7, color="#666", va="top")
ax[0].legend(fontsize=7, frameon=False, loc="upper right", markerscale=2.5)
ax[0].text(0.012, 0.21, "uniform weight = 0.2", fontsize=7, color="#555")
save(fig, "fig_weight_diagnosis")

# collapse rates
rows = ["KAN (tuned)", "Ens+CW", "Ens+CW+Boot+Guards", "EGVW-KAN", "v2"]
lab = ["A  single KAN", "C  Ens + CW", "E  + fitting guards", "H  EGVWC-KAN v1", "v2"]
t1 = {**N["v1"]["collapse"], **{k: v for k, v in N["t1"]["collapse"].items() if k in ("v2", "EGVW-KAN", "Ens+CW")}}
t2 = N["t2"]["collapse"]
fig, ax = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(rows)); w = 0.38
b1 = ax.bar(x - w / 2, [100 * t1[r] for r in rows], w, color="#9ECAE9", label="Stage 1, 24 datasets (360 folds)")
b2 = ax.bar(x + w / 2, [100 * t2[r] for r in rows], w, color="#4C78A8", label="Stage 2, 13 new datasets (195 folds)")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("collapsed folds (%)")
ax.legend(fontsize=8, frameon=False); save(fig, "fig_collapse")

# effect against dataset size
E = pd.DataFrame(N["effects"])
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for s, c, m in (("Stage 1", "#9ECAE9", "o"), ("Stage 2", "#4C78A8", "s")):
    G = E[E.stage == s]
    ax.scatter(G.n, G.delta, s=26, color=c, marker=m, edgecolor="#333", lw=0.4,
               label=f"{s} (Spearman n vs Δ = {N['effect_size_corr'][s]:.2f})")
ax.axhline(0, color="#555", lw=0.8); ax.set_xscale("log")
ax.set_xlabel("dataset size n (log scale)"); ax.set_ylabel("Δ MCC (v2 − v1)")
for _, r in E[E.delta.abs() > 0.15].iterrows():
    ax.annotate(r.dataset.split(" (")[0], (r.n, r.delta), fontsize=6.5, xytext=(4, 2), textcoords="offset points", color="#333")
ax.legend(fontsize=8, frameon=False, loc="upper right"); save(fig, "fig_effect_size")


# critical difference diagrams (ranks recomputed from the results so the groups use unrounded ranks)
def cd_plot(M, methods, title, name):
    R = M[methods].rank(axis=1, ascending=False).mean().sort_values(); k, n = len(methods), len(M)
    cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    half = (k + 1) // 2; y0 = 0.0; bottom = y0 - 0.45 - (half - 1) * 0.42 - 0.3; top = y0 + 1.15
    fig, ax = plt.subplots(figsize=(10.5, (top - bottom) * 0.55))
    ax.set_xlim(0.5, k + 0.5); ax.set_ylim(bottom, top); ax.axis("off")
    ax.plot([1, k], [y0, y0], color="k", lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.12], color="k", lw=0.8); ax.text(t, y0 + 0.2, str(t), ha="center", fontsize=7)
    ax.plot([1, 1 + cd], [y0 + 0.65, y0 + 0.65], color="#C0392B", lw=2)
    ax.text(1 + cd / 2, y0 + 0.75, f"CD = {cd:.2f}", ha="center", fontsize=7.5, color="#C0392B")
    for i, (m, r) in enumerate(R.items()):
        left = i < half; row = i if left else k - 1 - i
        yy = y0 - 0.45 - row * 0.42; xx = 0.6 if left else k + 0.4; bold = m in ("v2", "EGVW-KAN")
        ax.plot([r, r, xx], [y0, yy, yy], color="#1F3864" if bold else "#555", lw=1.1 if bold else 0.7)
        ax.text(xx + (-0.05 if left else 0.05), yy, f"{NAME.get(m, m)} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7.2, fontweight="bold" if bold else "normal")
    rs, cl = R.values, []
    for i in range(len(rs)):
        j = i
        while j + 1 < len(rs) and rs[j + 1] - rs[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl): cl.append((i, j))
    for c, (i, j) in enumerate(cl):
        ax.plot([rs[i] - 0.03, rs[j] + 0.03], [y0 - 0.18 - (c % 4) * 0.1] * 2, color="#7F8C8D", lw=2.4, solid_capstyle="round")
    ax.set_title(title, fontsize=9, loc="left"); save(fig, name)


P1 = results(*V1_KAN); B1 = results(*V1_BASELINES, *V1_VARIANTS)
V2 = results(*STAGE1); V2 = V2[V2.method == "v2"]; T2 = results(*STAGE2_KAN, *STAGE2_COMP)
COMP = sorted(B1.method.unique())
cd_plot(pd.concat([P1, B1, V2]).groupby(["dataset", "method"]).mcc.mean().unstack(), COMP + ["EGVW-KAN", "KAN (tuned)", "Ens-uniform"],
        "(a) v1 benchmark, 19 methods, 24 datasets", "fig_cd_v1")
cd_plot(T2.groupby(["dataset", "method"]).mcc.mean().unstack(), COMP + ["EGVW-KAN", "KAN (tuned)", "Ens-uniform", "Ens+CW", "v2"],
        "(b) Stage 2, 21 methods, 13 new datasets", "fig_cd_t2")

# reliability diagrams
CCOL = {"EGVW-KAN": "#E45756", "v2": "#1F3864", "Ens+CW": "#54A24B", "CatBoost": "#F2A541", "Logistic regression": "#9D755D"}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, (stage, files) in zip(axes, (("Stage 1", pred_files(V1_KAN, "kan") + pred_files(STAGE1, "v2") + pred_files(V1_BASELINES, "baselines")),
                                     ("Stage 2", pred_files(STAGE2_KAN, "kan") + pred_files(STAGE2_COMP, "baselines")))):
    pooled = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for m, pr in zip(list(z["methods"]), z["probs"]):
            if m in CCOL: pooled.setdefault(m, []).append((pr.max(1), (pr.argmax(1) == z["y"]).astype(float)))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=0.8)
    for m in CCOL:
        if m not in pooled: continue
        conf = np.concatenate([c for c, _ in pooled[m]]); acc = np.concatenate([a for _, a in pooled[m]])
        idx = np.clip(np.digitize(conf, np.linspace(0, 1, 11)[1:-1]), 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [acc[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=CCOL[m], label=f"{NAME.get(m, m)} (ECE {PR['ece'][stage][m]:.3f})")
    ax.set_xlabel("confidence (probability of the predicted class)"); ax.set_ylabel("observed accuracy")
    ax.set_title(f"({'a' if stage == 'Stage 1' else 'b'}) {stage}", loc="left", fontsize=9); ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.2, 1.0)
save(fig, "fig_calibration")
