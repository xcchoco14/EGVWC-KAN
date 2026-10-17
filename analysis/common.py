"""Paths and small helpers shared by the analysis scripts."""
import glob, json, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "analysis", "output")
FIG = os.path.join(REPO, "analysis", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# result folders, one per Kaggle run
V1_KAN = ("v1_part1_A", "v1_part1_B")             # EGVW-KAN, tuned single KAN, ablation rows
V1_BASELINES = ("v1_part2_A", "v1_part2_B")       # 8 classical baselines
V1_VARIANTS = ("v1_part3_A", "v1_part3_B")        # 8 KAN variants
V1_RESAMPLING = ("v1_part4_A", "v1_part4_B")      # Part 4
STAGE1 = ("v2_stage1_A", "v2_stage1_B")           # v2, development stage
STAGE2_KAN = ("v2_stage2_kan_A", "v2_stage2_kan_B")
STAGE2_COMP = ("v2_stage2_comp",)


def results(*runs):
    """Per-fold metrics of the given runs (one row per dataset, repeat, fold and method)."""
    return pd.concat([pd.read_json(os.path.join(RESULTS, r, "results.jsonl"), lines=True) for r in runs], ignore_index=True)


def info(*runs):
    """Per-unit logs (selected hyperparameters, member weights, inner-CV scores, ...)."""
    return pd.concat([pd.read_json(os.path.join(RESULTS, r, "info.jsonl"), lines=True) for r in runs], ignore_index=True)


def inclusion(run):
    return pd.read_csv(os.path.join(RESULTS, run, "dataset_inclusion.csv"))


def pred_files(runs, suffix):
    """Stored test-fold probabilities, e.g. suffix 'kan' for the KAN units."""
    return sorted(f for r in runs for f in glob.glob(os.path.join(RESULTS, r, "preds", f"*__{suffix}.npz")))


def load(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))


def save(obj, name):
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("wrote", os.path.join("analysis", "output", name))


def holm(p):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(p, float); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def collapsed(probs):
    """True when every test row of a fold is predicted as the same class."""
    return len(np.unique(probs.argmax(1))) < probs.shape[1]
