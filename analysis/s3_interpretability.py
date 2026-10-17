"""Agreement between ensemble members on two binary datasets (Pima from shard A, WDBC from shard B).

Reads the member-agreement tables and plots written by Stage 5 of the Part 1 notebooks, stacks the two plots into one
figure and summarises the agreement of features with a large and a small effect.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import RESULTS, FIG, save

SETS = [("Pima Indians Diabetes", "v1_part1_A", "Pima_Indians_Diabetes"),
        ("Breast Cancer Wisconsin (Diagnostic)", "v1_part1_B", "Breast_Cancer_Wisconsin_Diagnostic")]
out, imgs = {}, []
for name, run, slug in SETS:
    A = pd.read_csv(os.path.join(RESULTS, run, "tables", f"member_agreement_{slug}.csv"))
    big, small = A[A.ensemble_effect_range >= 0.5], A[A.ensemble_effect_range < 0.5]   # effect range in logits
    out[name] = dict(n_features=len(A), n_big=len(big), big_min_r=round(float(big.member_agreement_r.min()), 3),
                     small_median_r=round(float(small.member_agreement_r.median()), 3) if len(small) else None,
                     small_min_r=round(float(small.member_agreement_r.min()), 3) if len(small) else None,
                     top=A.head(4)[["feature", "ensemble_effect_range", "member_agreement_r"]].round(3).values.tolist())
    imgs.append(plt.imread(os.path.join(RESULTS, run, "figures", f"member_agreement_{slug}.png")))

fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.0))
for ax, im, lab in zip(axes, imgs, ("(a) Pima Indians Diabetes", "(b) Breast Cancer Wisconsin (Diagnostic)")):
    ax.imshow(im); ax.axis("off"); ax.set_title(lab, loc="left", fontsize=11, fontweight="bold")
fig.tight_layout(h_pad=0.6)
for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"fig_interpretability.{ext}"), dpi=200)
save(out, "interpretability.json")
