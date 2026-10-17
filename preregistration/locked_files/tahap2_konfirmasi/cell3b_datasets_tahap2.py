# ═════════════════════════════════════════════════════════════════════════════
# CELL 3b — TAHAP 2 (confirmation): NEW CANDIDATE DATASETS — replaces the v1 candidate list
# None of these datasets was used in the v1 benchmark or in Tahap 1. Corrections are fixed here, before any model is
# trained on them, by the same principles as v1 (A3/B6): outcome-derived or post-baseline columns are removed, other
# targets of the same record are removed, ambiguous labels are removed, repeated measurements of one patient are
# grouped. The same inclusion rules R1–R5 (Cell 3, audit_datasets) decide inclusion.
# ═════════════════════════════════════════════════════════════════════════════
CANDIDATES.clear()


def _dedupe_columns(X):
    """ucimlrepo returns two columns with the same name for a few datasets (e.g. 'MDVP:Jitter(%)' and
    'MDVP:Jitter(Abs)' both as 'MDVP:Jitter'); make every name unique, order kept."""
    seen, cols = {}, []
    for c in X.columns:
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    X = X.copy(); X.columns = cols
    return X


@candidate("Heart Failure Clinical Records", "hf_faisalabad", positive=["1"], nominal=[])
def _():
    X, T = _uci(519)
    return X.drop(columns=["time"]), _pick_target(T, "death_event"), None, \
        ["dropped 'time' (follow-up period: patients who die early have short follow-up → outcome-derived)"]


@candidate("HCV Data", "hcv_hannover", positive=["disease"], nominal=["Sex"])
def _():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=571); X = d.data.features.copy(); t = d.data.targets["Category"].astype(str)
    keep = ~t.str.startswith("0s")                               # '0s=suspect Blood Donor': ambiguous label
    y = t[keep].map(lambda v: "donor" if v.startswith("0=") else "disease")
    return X[keep].reset_index(drop=True), y.reset_index(drop=True), None, \
        [f"removed {int((~keep).sum())} 'suspect blood donor' rows (ambiguous label)",
         "label = liver disease (hepatitis, fibrosis or cirrhosis) vs blood donor"]


@candidate("Haberman's Survival", "haberman", positive=["2"])
def _():
    X, T = _uci(43); return X, _pick_target(T, "survival_status"), None, ["2 = died within 5 years of surgery"]


@candidate("Diabetic Retinopathy Debrecen", "dr_debrecen", positive=["1"])
def _():
    X, T = _uci(329); return _dedupe_columns(X), _pick_target(T, "Class"), None, ["1 = signs of diabetic retinopathy"]


@candidate("Thoracic Surgery", "thoracic_wroclaw", positive=["T"], nominal=["DGN", "PRE6", "PRE14"])
def _():
    X, T = _uci(277); return X, _pick_target(T, "Risk1Yr"), None, ["T = death within one year after surgery"]


@candidate("Breast Cancer (Ljubljana)", "breast_ljubljana", positive=["recurrence-events"],
           nominal=["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad", "irradiat"])
def _():
    X, T = _uci(14); return X, _pick_target(T, "Class"), None, ["positive = recurrence"]


@candidate("Fertility", "fertility", positive=["O"])
def _():
    X, T = _uci(244); return X, _pick_target(T, "diagnosis"), None, ["O = altered semen diagnosis"]


@candidate("Glioma Grading", "glioma_tcga", positive=["1"], nominal=["Race"])
def _():
    X, T = _uci(759); return X, _pick_target(T, "Grade"), None, ["1 = GBM (high grade), 0 = LGG"]


@candidate("Cervical Cancer Risk Factors", "cervical_caracas", positive=["1"])
def _():
    from ucimlrepo import fetch_ucirepo
    X = fetch_ucirepo(id=383).data.features.copy()
    y = X["Biopsy"]
    X = X.drop(columns=["Biopsy", "Hinselmann", "Schiller", "Citology"])
    return X, y, None, ["target = Biopsy; dropped Hinselmann, Schiller, Citology (other diagnostic tests of the same visit)"]


@candidate("Parkinson's (Little 2008)", "parkinsons_little", positive=["1"])
def _():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=174)
    X = _dedupe_columns(d.data.features)
    subj = d.data.ids["name"].astype(str).str.extract(r"(S\d+)")[0].to_numpy()   # phon_R01_S01_1 → S01
    return X, _pick_target(d.data.targets, "status"), subj, \
        [f"A4: grouped by subject ({len(set(subj))} subjects, several recordings each)"]


@candidate("Chronic Kidney Disease Risk (Bangladesh)", "ckd_bangladesh", positive=["ckd"],
           nominal=["bp (Diastolic)", "bp limit", "sg", "al", "rbc", "su", "pc", "pcc", "ba", "bgr", "bu", "sod", "sc", "pot",
                    "hemo", "pcv", "rbcc", "wbcc", "htn", "dm", "cad", "appet", "pe", "ane", "grf", "age"])
def _():
    X, T = _uci(857)
    return X.drop(columns=["stage", "affected"]), _pick_target(T, "class"), None, \
        ["dropped 'stage' (CKD stage, derived from the outcome) and 'affected'; features are recorded as ranges"]


@candidate("AIDS Clinical Trials (ACTG 175)", "actg175", positive=["1"], nominal=["trt", "race", "strat"])
def _():
    X, T = _uci(890)
    X = X.drop(columns=["time", "cd420", "cd820", "offtrt", "treat"])
    return X, _pick_target(T, "cid"), None, \
        ["dropped time (follow-up), cd420 / cd820 (CD4/CD8 at week 20, post-baseline), offtrt (post-baseline event), "
         "treat (duplicate of trt); baseline variables only"]


@candidate("Pediatric Appendicitis (Regensburg)", "appendicitis_regensburg", positive=["appendicitis"],
           nominal=["Sex", "Appendix_on_US", "Migratory_Pain", "Lower_Right_Abd_Pain", "Contralateral_Rebound_Tenderness",
                    "Coughing_Pain", "Nausea", "Loss_of_Appetite", "Neutrophilia", "Ketones_in_Urine", "RBC_in_Urine",
                    "WBC_in_Urine", "Dysuria", "Stool", "Peritonitis", "Psoas_Sign", "Ipsilateral_Rebound_Tenderness",
                    "US_Performed", "Free_Fluids", "Appendix_Wall_Layers", "Target_Sign", "Appendicolith", "Perfusion",
                    "Perforation", "Surrounding_Tissue_Reaction", "Appendicular_Abscess", "Abscess_Location",
                    "Pathological_Lymph_Nodes", "Lymph_Nodes_Location", "Bowel_Wall_Thickening",
                    "Conglomerate_of_Bowel_Loops", "Ileus", "Coprostasis", "Meteorism", "Enteritis", "Gynecological_Findings"])
def _():
    X, T = _uci(938)
    return X.drop(columns=["Length_of_Stay"]), _pick_target(T, "Diagnosis"), None, \
        ["target = Diagnosis; dropped Length_of_Stay (after the outcome) and the other targets Management, Severity"]
