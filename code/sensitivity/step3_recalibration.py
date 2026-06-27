#!/usr/bin/env python3
# =============================================================================
# step3_recalibration_nonsmote.py   (Route B, Step 3 -- part 2: Table 6)
#
# Recalibration of the NON-SMOTE deployed model within eICU, mirroring Methods 2.5.8:
#   repeated stratified 5-fold cross-fitting (20 repeats); on each training partition
#   a recalibration map is fitted and applied to the held-out partition, so every
#   recalibrated prediction is out-of-sample. Three maps:
#     - intercept-only  (single logit shift; calibration-in-the-large)
#     - Platt / logistic (2-parameter logit-linear; primary)
#     - isotonic        (non-parametric monotone)
#   reported before vs after as median (2.5-97.5 pct) across repeats.
#   Full cohort: all three maps. Clean subset (underpowered): intercept + Platt only.
#
#   conda activate extub
#   python ~/Downloads/step3_recalibration_nonsmote.py
# =============================================================================
import os, pickle, warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")

DATA_DIR   = "/Users/ilizyue/Documents/xgb_extubation_failure/data"
PKL        = os.path.join(DATA_DIR, "xgb_extubation_failure_v2_nonsmote.pkl")
EICU_CSV   = os.path.join(DATA_DIR, "eICUdata-1775370213861.csv")
TARGET     = "extubation_failure"
MV_COL     = "mv_duration_hours"
N_REPEATS  = 20
RNG        = 42
LEAK_COLS = [
    "reintubated_48h", "hours_to_reintubation", "death_within_48h_of_extubation",
    "n_invasive_episodes", "hospital_mortality", "icu_mortality", "hospital_los_days",
    "icu_los_days", "source", "mortality_28d", "mortality_90d", "first_extubation_time",
    "subject_id", "hadm_id", "stay_id", "icu_intime", "icu_outtime", "admittime",
    "dischtime", "rrt_first_time",
]

def encode_and_filter(df, feat_template):
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    for c in feat_template:
        if c not in d.columns:
            d[c] = np.nan
    return d[feat_template]

def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

def cal_slope_intercept(y, p, eps=1e-6):
    z = to_logit(p, eps).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); tot = 0.0
    for i in range(n_bins):
        hi = p <= edges[i + 1] if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            tot += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return tot

def metrics(y, p):
    sl, ic = cal_slope_intercept(y, p)
    return dict(intercept=ic, slope=sl, brier=brier_score_loss(y, p),
                ece=ece(y, p), auroc=roc_auc_score(y, p))

# ---- recalibration maps: fit on train, return an apply() fn ------------------
def map_intercept(p_tr, y_tr):
    lg = to_logit(p_tr)
    def nll(c):
        q = np.clip(expit(lg + c), 1e-9, 1 - 1e-9)
        return -np.mean(y_tr * np.log(q) + (1 - y_tr) * np.log(1 - q))
    c = minimize_scalar(nll, bounds=(-10, 10), method="bounded").x
    return lambda p: expit(to_logit(p) + c)

def map_platt(p_tr, y_tr):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(to_logit(p_tr).reshape(-1, 1), y_tr)
    return lambda p: lr.predict_proba(to_logit(p).reshape(-1, 1))[:, 1]

def map_isotonic(p_tr, y_tr):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y_tr)
    return lambda p: iso.predict(p)

def crossfit_recal(y, p, fit_map, n_repeats=N_REPEATS):
    rows = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG + rep)
        p_cal = np.zeros(len(y))
        for tr, te in skf.split(p.reshape(-1, 1), y):
            apply_fn = fit_map(p[tr], y[tr])
            p_cal[te] = apply_fn(p[te])
        rows.append(metrics(y, p_cal))
    return rows

def summ(rows, key):
    v = np.array([r[key] for r in rows])
    return f"{np.median(v):.3f} ({np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f})"

# ---- score non-SMOTE deployed model on eICU (full + clean) ------------------
bundle = pickle.load(open(PKL, "rb"))
feats, model, imp = list(bundle["features"]), bundle["model"], bundle["imputer"]
eicu = pd.read_csv(EICU_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
yE = eicu[TARGET].astype(int).values
XE = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]), feats)
clean_mask = (pd.to_numeric(eicu[MV_COL], errors="coerce") > 0).to_numpy()
XE.loc[pd.to_numeric(XE[MV_COL], errors="coerce") <= 0, MV_COL] = np.nan
p_full = model.predict_proba(imp.transform(XE.values))[:, 1]
p_clean, y_clean = p_full[clean_mask], yE[clean_mask]

def report(tag, y, p, maps):
    print(f"\n=== TABLE 6  --  {tag}  (n={len(y)}, ev={int(y.sum())}) ===")
    b = metrics(y, p)
    print(f"  {'before':16s} intercept={b['intercept']:+.3f}  slope={b['slope']:.3f}  "
          f"Brier={b['brier']:.3f}  ECE={b['ece']:.3f}  AUROC={b['auroc']:.3f}")
    for name, fn in maps:
        rows = crossfit_recal(y, p, fn)
        print(f"  {name:16s} intercept={summ(rows,'intercept')}  slope={summ(rows,'slope')}")
        print(f"  {'':16s} Brier={summ(rows,'brier')}  ECE={summ(rows,'ece')}  AUROC={summ(rows,'auroc')}")

report("eICU FULL",  yE,      p_full,  [("intercept-only", map_intercept), ("Platt (primary)", map_platt), ("isotonic", map_isotonic)])
report("eICU CLEAN", y_clean, p_clean, [("intercept-only", map_intercept), ("Platt (primary)", map_platt)])
print("\nDONE. Paste both TABLE 6 blocks back.")
