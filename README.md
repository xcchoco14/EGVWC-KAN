# EGVWC-KAN

Code, results and pre-registration documents for the paper

> E. A. Nurrohmah et al. *Failure Analysis and Pre-Registered Repair of Reliability-Weighted Kolmogorov–Arnold Network
> Ensembles for Imbalanced Clinical Tabular Data.* Manuscript in preparation, 2026.

The paper evaluates EGVWC-KAN, an ensemble of five Kolmogorov–Arnold Networks (KANs) whose members are weighted by
validation loss and logit dispersion, against 18 other methods on 24 tabular clinical datasets. The weighting fails, and
the paper shows why. A repaired version (v2) is then tested in two pre-registered stages: development on the same 24
datasets and confirmation on 13 new ones.

All experiments ran on Kaggle (CPU). The notebooks here are the ones that produced the stored results.

## Contents

```
notebooks/
  v1/            v1 benchmark, Parts 1-4, each split into shard A and shard B
  v2_stage1/     v2, development stage on the 24 v1 datasets
  v2_stage2/     v2, confirmation on 13 new datasets
results/         outputs of every Kaggle run (per-fold metrics, logs, test-fold predictions)
analysis/        scripts that compute the numbers, tables and figures of the paper from results/
preregistration/ pre-registration documents, their SHA-256 hashes and the files they lock
```

| Notebook | What it runs |
|---|---|
| `v1/part1_egvw_kan_{A,B}` | EGVW-KAN (v1), tuned single KAN, ablation rows A to K |
| `v1/part2_baselines_{A,B}` | logistic regression, SVM, KNN, random forest, XGBoost, LightGBM, CatBoost, MLP |
| `v1/part3_kan_variants_{A,B}` | eight KAN variants from the FC-KAN repository, and the main v1 analysis |
| `v1/part4_resampling_{A,B}` | SMOTE-NC, undersampling and threshold moving in place of class weights |
| `v2_stage1/v2_stage1_{A,B}` | v2 and its ablation rows on the v1 folds |
| `v2_stage2/stage2_kan_{A,B}` | v1, ablation rows and v2-final on the Stage 2 datasets |
| `v2_stage2/stage2_comparators` | baselines and KAN variants on the Stage 2 datasets |

## Data

All datasets are public. The notebooks download them at run time from the UCI Machine Learning Repository (through
`ucimlrepo` or the dataset's UCI page) and from scikit-learn, then apply the corrections listed in Cell 3 (Cell 3b for
Stage 2). `results/*/dataset_inclusion.csv` records which candidates were included and the rule that excluded the others.

## Reproducing the numbers in the paper

The stored results are enough; nothing has to be retrained.

```
pip install -r requirements.txt
python analysis/run_all.py
```

This writes JSON files to `analysis/output/` and figures to `analysis/figures/`. The scripts run in this order:

| Script | Output |
|---|---|
| `s1_v1_preregistered_test.py` | pre-registered test of the v1 benchmark (shard B) |
| `s2_v1_benchmark.py` | v1 benchmark: means, ranks, Friedman test, Holm-Wilcoxon tests, ablation ladder, collapse rates, Part 4 |
| `s3_interpretability.py` | agreement between ensemble members (Pima, WDBC) |
| `s4_stage1.py` | Stage 1: v2 vs v1 and vs Ens+CW, freeze rule, go rule |
| `s5_stage2.py` | Stage 2: primary and secondary pre-registered tests |
| `s6_diagnosis.py` | diagnosis of the v1 weights over 1,800 ensemble members |
| `s7_discussion_checks.py` | checks of the propositions, nested-CV optimism, stability, Nadeau-Bengio intervals |
| `s8_robustness.py` | critical difference, Bayesian signed-rank test, Cliff's delta, calibration error |
| `s9_figures.py` | result figures |

We tested the scripts with Python 3.8, numpy 1.24, pandas 2.0, scipy 1.10 and matplotlib 3.7.

## Running the experiments again

Each notebook is meant for a Kaggle CPU session with internet access; Cell 1 installs the packages it needs (pykan,
XGBoost, LightGBM, CatBoost, imbalanced-learn and others) and clones the FC-KAN repository. A notebook stops itself
before Kaggle's 12-hour limit and resumes when you attach its previous output and run it again. The first cell of each
notebook explains its inputs.

Order: run the four v1 parts first (Part 4 needs Part 1 of the same shard). Stage 1 needs the output of Part 1 of the
same shard. Stage 2 needs no earlier output. Notebooks of one stage build identical folds from `BASE_SEED`, and the
analysis checks the test-fold hashes before it merges their results.

pykan trains with L-BFGS on CPU, which is not bit-reproducible across machines, so a rerun gives slightly different
numbers for the KAN models. The baselines and KAN variants reproduce exactly.

## Results format

Each folder in `results/` holds one Kaggle run:

- `results.jsonl`: one line per dataset, repeat, outer fold and method, with MCC, balanced accuracy, accuracy, precision,
  sensitivity, specificity, F1, AUROC, AUPRC, Brier score, training time and a hash of the test fold.
- `info.jsonl`: per-unit logs, such as the hyperparameters chosen by the inner CV, inner-CV scores and member weights.
- `preds/*.npz`: test-fold labels, row indices and predicted probabilities of every method.
- `dataset_inclusion.csv`, `protocol.json`, `done.txt` and the leakage self-test record.

## Pre-registration

`preregistration/` contains the pre-registration of the v1 confirmation test (shard B), the v2 pre-registration as
locked on 8 October 2026, and its Amendment 1, locked before Stage 2. The files under `preregistration/locked_files/` are
unchanged since their hashes were recorded. The notebooks in `notebooks/` contain the same code as the locked ones; we
edited only comments, markdown and the text of printed messages afterwards. You can check both points with

```
python preregistration/verify.py
```

## Citation

If you use this code or the results, please cite the paper above. Until it is published:

```
@unpublished{nurrohmah2026egvwckan,
  author = {Nurrohmah, Erna Alfi and others},
  title  = {Failure Analysis and Pre-Registered Repair of Reliability-Weighted {Kolmogorov--Arnold} Network
            Ensembles for Imbalanced Clinical Tabular Data},
  note   = {Manuscript in preparation. Code and data: https://github.com/xcchoco14/EGVWC-KAN},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The datasets themselves come from the UCI Machine Learning
Repository and scikit-learn and keep their original licenses.

## Contact

Erna Alfi Nurrohmah, ernaalfin14@gmail.com
