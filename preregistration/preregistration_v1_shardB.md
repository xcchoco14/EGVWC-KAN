# Pre-registration — EGVW member weighting, confirmation on shard B

Written 2026-10-04, **before** any shard-B result of Part 1 (EGVW-KAN) was opened.
Shard-B outputs of Part 1 (`part1-b`) had been downloaded but not read at the time of writing.

## Background (shard A = development)
On shard A (9 datasets, repeat 0, 43 outer folds) the pre-specified member weighting of EGVW-KAN
(score = validation loss + logit dispersion, γ = 1) did not improve on uniform weights of the same members:
mean MCC 0.590 (γ = 1) vs 0.602 (uniform, ablation row E); sharper weighting was worse (γ = 4: 0.571),
driven by Heart Disease (Cleveland) and Maternal Health Risk. The standardised-score variant
(z-scored loss + z-scored dispersion, computed post hoc on the same members, no retraining) gave 0.600.

## Decision
The candidate replacement weighting is **"EGVW-KAN [standardised score]"** (`member_weights(..., "egvw_std", γ = 1)`),
already computed for every outer fold by Part 1 as a post-hoc variant. No code or run is changed.

## Confirmation analysis (shard B only)
- Data: the 14 shard-B datasets, Part 1-B, all R × K = 3 × 5 outer folds per dataset; a dataset enters only when
  all 15 folds are finished (no partial datasets).
- Primary comparison: EGVW-KAN [standardised score] vs EGVW-KAN (γ = 1), MCC, dataset means,
  two-sided Wilcoxon signed-rank test, α = 0.05.
- Secondary (reported, Holm-corrected, not used to decide): standardised score vs uniform weights
  (Ens+CW+Boot+Guards), vs KAN (tuned), vs the best classical baseline; balanced accuracy, AUROC, Brier.
- Decision rule: if the primary comparison favours the standardised score (mean ΔMCC > 0 and p < 0.05),
  the paper presents the standardised score as the method's weighting, reporting shard A as development and
  shard B as confirmation. Otherwise the paper keeps the original definition (loss + dispersion, γ = 1)
  and reports the weighting result as a negative finding, with the γ sensitivity analysis.
- Either way, both rows are reported for all 24 datasets, and this document is cited in the Methods.
