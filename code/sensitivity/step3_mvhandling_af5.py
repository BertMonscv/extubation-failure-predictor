#!/usr/bin/env python3
# =============================================================================
# step3_mvhandling_af5_nonsmote.py   (Route B -- AF5 Table C)
#
# eICU FULL-cohort AUROC as a function of how the mechanical-ventilation-duration
# data-quality issue is handled, scored with the FROZEN non-SMOTE deployed model
# (xgb_extubation_failure_v2_nonsmote.pkl). Five scoring strategies (model fixed,
# only the eICU mv preprocessing changes) + a missingness-indicator RETRAIN pair
# (29 features, refit on full MIMIC, non-SMOTE). All AUROCs with 1000x stratified
# bootstrap 95% CI.
#
#   conda activate extub
#   cd ~/Documents/xgb_extubation_failure/data
#   python ~/Downloads/step3_mvhandling_af5_nonsmote.py
# =============================================================================
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA_DIR  = "/Users/ilizyue/Documents/xgb_extubation_failure/data"
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
              colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
              use_label_encoder=False, n_jobs=-1, random_state=RNG, verbosity=0)

def encode_and_filter(df, feat_template):
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    for c in feat_template:
        if c not in d.columns: d[c] = np.nan
    return d[feat_template]

def auroc_ci(y, p, B=1000, seed=RNG):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    point = roc_auc_score(y, p)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed); a = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        a.append(roc_auc_score(y[idx], p[idx]))
    return point, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def line(tag, y, p, note=""):
    pt, lo, hi = auroc_ci(y, p)
    print(f"  {tag:26s} AUROC={pt:.3f} ({lo:.3f}, {hi:.3f}){note}")

# ---- load frozen non-SMOTE deployed model ----------------------------------
bundle = pickle.load(open(PKL, "rb"))
feats, model_dep, imp_dep = list(bundle["features"]), bundle["model"], bundle["imputer"]
eicu = pd.read_csv(EICU_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
yE = eicu[TARGET].astype(int).values
XE = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]), feats)
eicu_pos_med = float(XE.loc[XE[MV_COL] > 0, MV_COL].median())
n_nonpos = int((XE[MV_COL] <= 0).sum()); n_nan = int(XE[MV_COL].isna().sum())
print(f">> eICU full n={len(yE)} ev={int(yE.sum())} | mv<=0: {n_nonpos}  mv NaN: {n_nan}  "
      f"eICU positive-mv median={eicu_pos_med:.1f}")

def frozen(Xdf, impute=True):
    arr = Xdf.values
    return (model_dep.predict_proba(imp_dep.transform(arr)) if impute
            else model_dep.predict_proba(arr))[:, 1]

# ---- five frozen-model strategies ------------------------------------------
print("\n=== TABLE C  --  eICU full AUROC by mv handling (frozen non-SMOTE deployed model) ===")
# retain invalid values as-is
line("retain-invalid", yE, frozen(XE.copy(), True), "   [old SMOTE 0.547]")
# set non-positive to 0
X = XE.copy(); X.loc[X[MV_COL] <= 0, MV_COL] = 0.0
line("set-to-0", yE, frozen(X, True))
# non-positive -> NaN -> impute with eICU's OWN positive-mv median (others MIMIC median)
X = XE.copy(); X.loc[X[MV_COL] <= 0, MV_COL] = np.nan; X[MV_COL] = X[MV_COL].fillna(eicu_pos_med)
line("eICU-positive-median", yE, frozen(X, True), "   [old SMOTE 0.589]")
# non-positive -> NaN -> deployed MIMIC-median imputer (the published pipeline)
X = XE.copy(); X.loc[X[MV_COL] <= 0, MV_COL] = np.nan
line("MIMIC-median (deployed)", yE, frozen(X, True), "   [= Table 2; old SMOTE 0.615]")
# non-positive -> NaN, no imputation (XGBoost native missingness)
X = XE.copy(); X.loc[X[MV_COL] <= 0, MV_COL] = np.nan
line("native-missing", yE, frozen(X, False), "   [old SMOTE 0.713]")

# ---- missingness-indicator retrain pair (29 features, non-SMOTE) -----------
print("\n=== indicator-retrain (28 features + mv_missing flag, refit full MIMIC, non-SMOTE) ===")
mimic = pd.read_csv(MIMIC_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
yM = mimic[TARGET].astype(int).values
XM = encode_and_filter(mimic.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in mimic.columns]), feats)
mvflag_M = ((XM[MV_COL] <= 0) | XM[MV_COL].isna()).astype(int)
print(f"  MIMIC mv_missing=1 count: {int(mvflag_M.sum())} of {len(XM)}  "
      f"(if ~0, the indicator is near-constant in training and cannot help)")
XM.loc[XM[MV_COL] <= 0, MV_COL] = np.nan
XM29 = XM.copy(); XM29["mv_missing"] = mvflag_M.values
imp29 = SimpleImputer(strategy="median").fit(XM29.values)
clf29 = XGBClassifier(**XGB_HP).fit(imp29.transform(XM29.values), yM)

mvflag_E = ((XE[MV_COL] <= 0) | XE[MV_COL].isna()).astype(int)
XE_ind = XE.copy(); XE_ind.loc[XE_ind[MV_COL] <= 0, MV_COL] = np.nan
XE_ind["mv_missing"] = mvflag_E.values
line("indicator + MIMIC-median", yE, clf29.predict_proba(imp29.transform(XE_ind.values))[:, 1], "   [old SMOTE 0.633]")
line("indicator + native",       yE, clf29.predict_proba(XE_ind.values)[:, 1],                  "   [old SMOTE 0.633]")

print("\nDONE. Paste the TABLE C block + the indicator pair (incl. the mv_missing count).")
