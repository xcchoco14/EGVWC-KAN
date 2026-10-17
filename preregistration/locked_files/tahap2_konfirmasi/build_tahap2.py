"""Builds the Tahap 2 (confirmation) notebooks from the Part 5 notebook. All model, split, preprocessing and runner
code is inherited unchanged; Tahap 2 adds: the new candidate datasets (cell3b_datasets_tahap2.py, replacing the v1
list), a separate pipeline version (so no v1 / Tahap 1 result can ever be merged), and a wrapper that adds the frozen
v2-final rows to every Part-1-style unit, so v2 uses the KAN configuration selected on the same fold in the same unit.
    t2-kan-a / t2-kan-b : EGVW-KAN v1 + ablation ladder (A–K) + v2-final rows, shard A / B
    t2-comp             : 8 classical baselines + 8 published KAN variants, all datasets
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "tahap1_pengembangan", "egvw_part5_v2_A.ipynb")
CELL3B = open(os.path.join(HERE, "cell3b_datasets_tahap2.py"), encoding="utf-8").read()
SHARD_A = ["AIDS Clinical Trials (ACTG 175)", "Cervical Cancer Risk Factors", "Diabetic Retinopathy Debrecen", "HCV Data",
           "Haberman's Survival", "Fertility", "Parkinson's (Little 2008)"]
WRAP = '''# ═════════════════════════════════════════════════════════════════════════════
# CELL 7d — TAHAP 2: v2-final rows inside every KAN unit
# v2-final = the v2 definition frozen in preregistration_v2.md (Tahap 1 kept the defaults: γ = 1, C = 100, gate on).
# It runs on the KAN configuration that the inner search of THIS unit selected (out["info"]["params"]), i.e. exactly
# the configuration EGVW-KAN v1 and the ablation rows use, on the same outer fold.
# ═════════════════════════════════════════════════════════════════════════════
if "_run_kan_family_v1" not in globals():
    _run_kan_family_v1 = run_kan_family


def run_kan_family(bundle, tr_idx, te_idx, seed):
    out = _run_kan_family_v1(bundle, tr_idx, te_idx, seed)
    v = run_v2(bundle, tr_idx, te_idx, seed, params=out["info"]["params"])
    for name, P in v["probs"].items():
        out["probs"][name] = P; out["times"][name] = out["times"].get("KAN (tuned)", 0.0) + v["times"][name]
    out["sens"].update(v["sens"])
    out["info"]["v2"] = v["info"]
    out["info"]["train_failures_total"] = int(out["info"].get("train_failures_total", 0) + v["info"]["train_failures_total"])
    return out
'''


def src(c): return "".join(c["source"])
def setsrc(c, s): c["source"] = s.splitlines(keepends=True)
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old)); return s.replace(old, new)


for slug, nbname, shard, kan in (("t2-kan-a", "kan", "A", True), ("t2-kan-b", "kan", "B", True), ("t2-comp", "compare", "", False)):
    nb = json.load(open(SRC, encoding="utf-8")); C = nb["cells"]
    for c in C:
        if c["cell_type"] == "code": c["outputs"], c["execution_count"] = [], None
    what = ("EGVW-KAN v1, ablation ladder A–K and the frozen v2-final rows" if kan
            else "8 classical baselines and 8 published KAN variants")
    setsrc(C[0], f"""# EGVWC-KAN v2 — Tahap 2 (confirmation on new datasets) — `{slug}`

This notebook runs **{what}** on the **Tahap 2 datasets** — {'shard ' + shard if shard else 'all of them'}.
None of these datasets was used in the v1 benchmark or in Tahap 1 (development). Their corrections and the inclusion
audit (rules R1–R5, unchanged from v1) were fixed in the amendment to `preregistration_v2.md` before any model was
trained on them. v2-final is the definition frozen after Tahap 1 (γ = 1, C = 100, gate on), unchanged.

The three Tahap 2 notebooks (`t2-kan-a`, `t2-kan-b`, `t2-comp`) build identical folds from `BASE_SEED`; their outputs merge
exactly in the analysis. `PIPELINE_VERSION` differs from v1 and Tahap 1, so their results can never be mixed in.
**No input is needed for the first session**; from the second session on, attach this notebook's own output (resume).
""")
    s = src(C[4])
    s = rep(s, 'NOTEBOOK = "v2"', f'NOTEBOOK = "{nbname}"')
    s = rep(s, 'SHARD = "A"', f'SHARD = "{shard}"')
    s = rep(s, '    PIPELINE_VERSION  = "2026-09-29c",   # results from other versions are never merged',
            '    PIPELINE_VERSION  = "2026-10-09-tahap2",   # Tahap 2: never merged with v1 / Tahap 1 results')
    s = rep(s, '    V2_VERSION     = "v2.0-dev",       # v2 units of another version are never merged',
            '    V2_VERSION     = "v2.0-final",     # frozen after Tahap 1 (defaults kept)')
    s = rep(s, '    RUN_V2         = NOTEBOOK == "v2",                  # Part 5: EGVWC-KAN v2 (Cell 7c)',
            '    RUN_V2         = False,           # Tahap 2: v2-final rows are produced inside each KAN unit (Cell 7d)')
    if not kan:
        s = rep(s, '    RUN_BASELINES  = NOTEBOOK in ("baselines", "all"),', '    RUN_BASELINES  = True,')
        s = rep(s, '    RUN_VARIANTS   = NOTEBOOK in ("variants", "all"),', '    RUN_VARIANTS   = True,')
    a = s.index("    SHARD_A_DATASETS = ["); b = s.index("],", a) + 2
    s = s[:a] + "    SHARD_A_DATASETS = " + json.dumps(SHARD_A, ensure_ascii=False) + "," + s[b:]
    setsrc(C[4], s)
    s = src(C[28])
    assert "leakage_selftest_passed_v2_" in s
    setsrc(C[28], s.replace("leakage_selftest_passed_v2_", f"leakage_selftest_passed_{slug}_"))
    setsrc(C[31], "# Stage 4 — quick look\n`analyse_v2()` compares the v2 rows with EGVW-KAN v1 and Ens+CW in this notebook's output "
                  "(KAN notebooks only). The pre-registered Tahap 2 analysis is run locally on the merged outputs of all three notebooks.")
    # new cells: Cell 3b after Cell 3 (index 6), Cell 7d after Cell 7c (index 18 before insertion)
    C[19:19] = [md("## Cell 7d — Tahap 2: v2-final inside every KAN unit"), code(WRAP)]
    C[7:7] = [md("## Cell 3b — Tahap 2 datasets (replaces the v1 candidate list)"), code(CELL3B)]
    out = os.path.join(HERE, f"{slug}.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out, "| cells:", len(C))
