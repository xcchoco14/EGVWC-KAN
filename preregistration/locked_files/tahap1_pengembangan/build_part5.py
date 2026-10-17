"""Builds the Part 5 (EGVWC-KAN v2, development stage) notebooks from the Part 4 notebooks of the v1 benchmark.
Everything that defines the data, folds, preprocessing, KAN training and the runner is inherited unchanged; only
the configuration, the new Cell 7c (cell7c_v2.py), the runner hooks for group "v2" and Stage 4 differ.
Output: egvw_part5_v2_A.ipynb, egvw_part5_v2_B.ipynb (next to this script)."""
import copy, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CELL7C = open(os.path.join(HERE, "cell7c_v2.py"), encoding="utf-8").read()


def src(c): return "".join(c["source"])
def setsrc(c, s): c["source"] = s.splitlines(keepends=True)


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:70], s.count(old))
    return s.replace(old, new)


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


for shard in ("A", "B"):
    nb = json.load(open(os.path.join(ROOT, f"egvw_part4_resampling_{shard}.ipynb"), encoding="utf-8"))
    C = nb["cells"]
    for c in C:
        if c["cell_type"] == "code": c["outputs"], c["execution_count"] = [], None

    # ── header ──
    other = "B" if shard == "A" else "A"
    setsrc(C[0], f"""# EGVWC-KAN v2 — Part 5 — development stage (Tahap 1) — shard {shard}

> **Shard {shard} of 2.** Twin notebook: `egvw_part5_v2_{other}`. Folds, seeds and the protocol fingerprint are those of the
> v1 benchmark (Parts 1–4), so every v2 row pairs exactly with EGVW-KAN (v1) and the ablation rows of Part 1.

**Why this part exists.** In the v1 benchmark EGVW-KAN did not beat its comparators. The ablation located the cause:
the over-fitting guard (λ × 5) produces flat members on small datasets, a flat member has near-zero logit dispersion,
and the v1 score (validation loss + dispersion) therefore gives it the largest weight (single-class collapse rose from
1.7 % of folds to 15.8 %). v2 replaces exactly this part of the method — see Cell 7c and `preregistration_v2.md`:
bagged members with the tiered class weights, member scores from the balanced log-loss on each member's out-of-bag
rows, a gate that gives a collapsed member weight 0, and shrinkage of the weights towards uniform when the
out-of-bag sets are small. No fitting guards, no internal 80/20 split, no refit.

**This is the development stage.** It runs on the same 24 datasets as v1, so it can show whether the diagnosis was
right and fix v2's remaining settings, but it is not the confirmation. The frozen v2 is confirmed afterwards on
datasets never used before (Tahap 2).

**Inputs to attach (Add Input → Your Work):** the latest output of Part 1 of the same shard (`part1-{shard.lower()}`),
which provides the KAN configuration selected on every fold, and — from the second session on — this notebook's own
output (resume). Units whose Part 1 configuration is missing wait.

Cells 1–8 are inherited unchanged from the v1 notebooks except: Cell 2 (`NOTEBOOK = "v2"`, v2 settings), the new
Cell 7c, the runner hooks for the unit group `"v2"`, and Stage 4 (`analyse_v2()`).
""")

    # ── Cell 2: configuration ──
    s = src(C[4])
    s = rep(s, 'NOTEBOOK = "resampling"', 'NOTEBOOK = "v2"')
    s = rep(s, '    RUN_RESAMPLING = NOTEBOOK == "resampling",          # Part 4: SMOTE-NC / undersampling / threshold moving\n',
            '    RUN_RESAMPLING = NOTEBOOK == "resampling",          # Part 4: SMOTE-NC / undersampling / threshold moving\n'
            '    RUN_V2         = NOTEBOOK == "v2",                  # Part 5: EGVWC-KAN v2 (Cell 7c)\n'
            '    # ── EGVWC-KAN v2 (Part 5, preregistration_v2.md). Not in the fingerprint: v2 must pair with Part 1. ──\n'
            '    V2_VERSION     = "v2.0-dev",       # v2 units of another version are never merged\n'
            '    V2_C_SHRINK    = 100,              # γ_eff = γ · n_oob / (n_oob + C)\n'
            '    V2_GAMMA_GRID  = [0.5, 1.0, 2.0, 4.0],   # post-hoc sensitivity (no retraining)\n'
            '    V2_SHRINK_GRID = [0, 25, 100, 400],\n')
    setsrc(C[4], s)

    # ── runner hooks ──
    s = src(C[18])
    s = rep(s, '("resampling", CONFIG.get("RUN_RESAMPLING", False))) if on]',
            '("resampling", CONFIG.get("RUN_RESAMPLING", False)), ("v2", CONFIG.get("RUN_V2", False))) if on]')
    s = rep(s, '"resampling": run_resampling}[t["group"]]', '"resampling": run_resampling, "v2": run_v2}[t["group"]]')
    s = rep(s, '    if group == "resampling": return "resampling"\n',
            '    if group == "resampling": return "resampling"\n    if group == "v2": return "v2"\n')
    s = rep(s, '(grp == "variants" and r.get("variant_version") != CONFIG["VARIANT_VERSION"]):',
            '(grp == "variants" and r.get("variant_version") != CONFIG["VARIANT_VERSION"]) or \\\n'
            '               (grp == "v2" and r.get("v2_version") != CONFIG.get("V2_VERSION")):')
    s = rep(s, 'method_version=CONFIG["METHOD_VERSION"], variant_version=CONFIG["VARIANT_VERSION"],',
            'method_version=CONFIG["METHOD_VERSION"], variant_version=CONFIG["VARIANT_VERSION"],\n'
            '                v2_version=CONFIG.get("V2_VERSION"),')
    n = s.count('t["group"] == "resampling"')
    s = s.replace('t["group"] == "resampling"', 't["group"] in ("resampling", "v2")')
    assert n >= 3, n
    s = rep(s, 'def leakage_self_test(bundle, groups=("kan", "baselines", "variants", "resampling")):',
            'def leakage_self_test(bundle, groups=("kan", "baselines", "variants", "resampling", "v2")):')
    s = rep(s, '               "resampling": run_resampling}\n', '               "resampling": run_resampling, "v2": run_v2}\n')
    s = rep(s, '"variants": "RUN_VARIANTS", "resampling": "RUN_RESAMPLING"}[g], False)]',
            '"variants": "RUN_VARIANTS", "resampling": "RUN_RESAMPLING", "v2": "RUN_V2"}[g], False)]')
    setsrc(C[18], s)

    # ── Stage 2 marker, Stage 4 ──
    s = src(C[26]).replace("leakage_selftest_passed_resampling_", "leakage_selftest_passed_v2_")
    s = s.replace('{CONFIG["VARIANT_VERSION"]}.txt', '{CONFIG["VARIANT_VERSION"]}_{CONFIG["V2_VERSION"]}.txt')
    setsrc(C[26], s)
    assert src(C[30]).strip() == "analyse()"
    setsrc(C[30], "analyse_v2()")
    setsrc(C[29], "# Stage 4 — v2 vs v1\n`analyse_v2()` compares every v2 row with EGVW-KAN (v1, row H) and Ens+CW (row C) of Part 1 on "
                  "identical folds, over the datasets on which every row has all 15 folds. Writes `v2_vs_v1.csv` and "
                  "`v2_mcc_by_dataset.csv` to the output folder. The pre-registered decision is taken locally from these files.")

    # ── new Cell 7c after Cell 7b ──
    C[17:17] = [md("## Cell 7c — Part 5: EGVWC-KAN v2"), code(CELL7C)]
    out = os.path.join(HERE, f"egvw_part5_v2_{shard}.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out, "| cells:", len(C))
