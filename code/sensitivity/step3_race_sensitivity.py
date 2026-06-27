#!/usr/bin/env python3
# =============================================================================
# step3_race_sensitivity_nonsmote.py   (Route B, Step 3 -- sensitivity tail)
#
# REQUIRED:
#   * Race / ethnicity subgroups (sect 141, INTERNAL MIMIC-IV OOF only):
#       White / Black / Hispanic-Latino / Asian / Other / Unrecorded
#       AUROC + 1000-boot CI + events. Prints event counts to verify the
#       grouping reproduces 249/12/14/4/5/58 (sum 342).
#
# OPTIONAL de-SMOTE of the scattered sensitivity numbers (use if you want the
# whole paper SMOTE-free; otherwise keep the old values):
#   * no-RRT subgroup        (received_rrt == 0), internal OOF AUROC + CI
#   * death-excluded         internal OOF AUROC after removing patients whose
#                            ONLY qualifying event was death (re-EVALUATION on
#                            the fixed OOF; flag if your original was a refit)
#   * mv handling-choice     eICU full AUROC under 3 scoring strategies:
#                            retain-invalid / native-missing / median-impute
#
#   conda activate extub
#   python ~/Downloads/step3_race_sensitivity_nonsmote.py
# =============================================================================
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA_DIR  = __import__("os").path.expanduser("~/Documents/xgb_extubation_failure/data")
PKL       = os.path.join(DATA_DIR, "xgb_extubation_failure_v2_nonsmote.pkl")
MIMIC_CSV = os.path.join(DATA_DIR, "MIMIC-IVdata-1775367119727.csv")
EICU_CSV  = os.path.join(DATA_DIR, "eICUdata-1775370213861.csv")
TARGET, MV_COL, RNG = "extubation_failure", "mv_duration_hours", 42
LEAK_COLS = [
    "reintubated_48h", "hours_to_reintubation", "death_within_48h_of_extubation",
    "n_invasive_episodes", "hospital_mortality", "icu_mortality", "hospital_los_days",
    "icu_los_days", "source", "mortality_28d", "mortality_90d", "first_extubation_time",
    "subject_id", "hadm_id", "stay_id", "icu_intime", "icu_outtime", "admittime",
    "dischtime", "rrt_first_time",
]
XGB_HP = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric="logloss", use_label_encoder=False,
              n_jobs=-1, random_state=RNG, verbosity=0)

def encode_and_filter(df, feat_template):
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    for c in feat_template:
        if c not in d.columns: d[c] = np.nan
    return d[feat_template]

def auroc_ci(y, p, B=1000, seed=RNG):
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed); n = len(y); a = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n: a.append(roc_auc_score(y[idx], p[idx]))
    return roc_auc_score(y, p), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def show(label, y, p, n=None):
    if 0 < int(np.sum(y)) < len(y):
        a, lo, hi = auroc_ci(y, p)
        extra = f"  n={n}" if n is not None else ""
        print(f"    {label:18s} AUROC={a:.3f} ({lo:.3f}, {hi:.3f})  ev={int(np.sum(y))}{extra}")
    else:
        print(f"    {label:18s} (insufficient events: {int(np.sum(y))})")

def race_group(v):
    u = "" if pd.isna(v) else str(v).upper()
    if u.strip() == "": return "Unrecorded"
    if "WHITE" in u or "EUROPEAN" in u or "PORTUGUESE" in u: return "White"
    if "BLACK" in u or "AFRICAN" in u or "CARIBBEAN" in u: return "Black"
    if "HISPANIC" in u or "LATINO" in u or "SOUTH AMERICAN" in u: return "Hispanic/Latino"
    if "ASIAN" in u: return "Asian"
    if any(k in u for k in ["UNKNOWN", "UNABLE", "DECLINED", "NOT SPECIFIED", "PATIENT DECLINED"]):
        return "Unrecorded"
    return "Other"

# ---- data + matrices --------------------------------------------------------
bundle = pickle.load(open(PKL, "rb"))
feats, model_dep, imp_dep = list(bundle["features"]), bundle["model"], bundle["imputer"]
mimic = pd.read_csv(MIMIC_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
eicu  = pd.read_csv(EICU_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
yM = mimic[TARGET].astype(int).values
XM = encode_and_filter(mimic.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in mimic.columns]), feats).values

# XGBoost OOF on MIMIC (no SMOTE, RNG=42; identical to part 1/3)
oof = np.zeros(len(yM))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=RNG).split(XM, yM):
    imp = SimpleImputer(strategy="median").fit(XM[tr])
    m = XGBClassifier(**XGB_HP).fit(imp.transform(XM[tr]), yM[tr])
    oof[va] = m.predict_proba(imp.transform(XM[va]))[:, 1]
print(f">> MIMIC OOF ready: overall AUROC={roc_auc_score(yM, oof):.3f}  (events={int(yM.sum())})")

# ---- (1) RACE / ETHNICITY subgroups  (REQUIRED) -----------------------------
print("\n=== (1) RACE / ETHNICITY subgroups -- internal MIMIC-IV OOF ===")
racecol = next((c for c in ["race", "ethnicity", "Race", "Ethnicity"] if c in mimic.columns), None)
print(f"  race_col = {racecol!r}")
if racecol is not None:
    grp = mimic[racecol].map(race_group).values
    order = ["White", "Black", "Hispanic/Latino", "Asian", "Other", "Unrecorded"]
    print(f"  group event counts (target sum 342): "
          + ", ".join(f"{g} {int(yM[grp==g].sum())}/{int((grp==g).sum())}n" for g in order))
    for g in order:
        mask = (grp == g)
        if mask.sum() > 0:
            show(g, yM[mask], oof[mask], n=int(mask.sum()))

# ---- (2) no-RRT subgroup  (optional de-SMOTE) -------------------------------
print("\n=== (2) no-RRT subgroup (received_rrt==0) -- internal OOF [old 0.820] ===")
if "received_rrt" in mimic.columns:
    nr = (pd.to_numeric(mimic["received_rrt"], errors="coerce").fillna(0).to_numpy() == 0)
    show("no_rrt", yM[nr], oof[nr], n=int(nr.sum()))
else:
    print("    received_rrt column not found")

# ---- (3) death-excluded  (optional de-SMOTE; re-EVALUATION) -----------------
print("\n=== (3) death-excluded internal discrimination [old 0.840] ===")
if {"death_within_48h_of_extubation", "reintubated_48h"}.issubset(mimic.columns):
    death = pd.to_numeric(mimic["death_within_48h_of_extubation"], errors="coerce").fillna(0).to_numpy()
    reint = pd.to_numeric(mimic["reintubated_48h"], errors="coerce").fillna(0).to_numpy()
    death_only = (death == 1) & (reint == 0)
    keep = ~death_only
    print(f"    death-only-event patients removed: {int(death_only.sum())}  (old paper: 63)")
    show("death-excluded", yM[keep], oof[keep], n=int(keep.sum()))
    print("    NOTE: re-EVALUATION on the fixed OOF. If your original 0.840 was a model REFIT")
    print("          on the reduced cohort, tell me and I will refit instead.")
else:
    print("    death/reintubation columns not found -- cannot compute")

# ---- (4) eICU mv handling-choice  (optional de-SMOTE) -----------------------
print("\n=== (4) eICU full mv handling-choice [old retain 0.547 / native 0.713 / median 0.615] ===")
yE = eicu[TARGET].astype(int).values
XE_raw = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]), feats)
# (a) retain invalid mv as-is (no <=0 -> NaN rule); other missing median-imputed by deployed imputer
p_retain = model_dep.predict_proba(imp_dep.transform(XE_raw.values))[:, 1]
# (b) + (c): apply the v2 rule (mv<=0 -> NaN)
XE_nan = XE_raw.copy()
XE_nan.loc[pd.to_numeric(XE_nan[MV_COL], errors="coerce") <= 0, MV_COL] = np.nan
p_median = model_dep.predict_proba(imp_dep.transform(XE_nan.values))[:, 1]      # deployed pipeline
p_native = model_dep.predict_proba(XE_nan.values)[:, 1]                          # XGBoost native NaN, no imputer
for lab, p in [("retain-invalid", p_retain), ("native-missing", p_native), ("median-impute", p_median)]:
    a, lo, hi = auroc_ci(yE, p)
    print(f"    {lab:15s} AUROC={a:.3f} ({lo:.3f}, {hi:.3f})")

print("\nDONE. Paste blocks (1)-(4). Block (1) race is the one the manuscript needs;")
print("(2)-(4) only if you want the sensitivity numbers de-SMOTEd too.")
