#!/usr/bin/env python3
# =============================================================================
# c1_regen_deployed_nonsmote.py   (Route B, Step 1-for-Step-2)
#
# Regenerate the DEPLOYED XGBoost WITHOUT SMOTE, on the FROZEN 28 features.
#   - keeps the exact 01 Stage-5 config (300/5/0.05 + subsample 0.8 +
#     colsample 0.8, random_state=RNG=42), only removes the SMOTE resample
#   - reuses the existing pkl's feature list + median imputer (the imputer was
#     fit on the ORIGINAL MIMIC, median strategy -> unaffected by SMOTE)
#   - writes a NEW bundle (same keys/structure) so analysis_1c.py loads it
#     identically; the only change is a natural-prevalence model
#   - freshly trained in extub's xgboost 1.7.6 -> the new pkl loads cleanly
#     (no old-pickle warning, no get_params 'use_label_encoder' breakage)
#
# This does NOT re-run feature selection (Stage 2) and does NOT touch the
# comparators / OOF csv (that is Step 3, which needs the full make_models).
#
#   conda activate extub
#   python c1_regen_deployed_nonsmote.py
# =============================================================================
import os, sys, pickle, copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

# ------------------------- CONFIG (edit if needed) ---------------------------
DATA_DIR  = __import__("os").path.expanduser("~/Documents/xgb_extubation_failure/data")
PKL_IN    = os.path.join(DATA_DIR, "xgb_extubation_failure_v2.pkl")          # SMOTE deployed model
PKL_OUT   = os.path.join(DATA_DIR, "xgb_extubation_failure_v2_nonsmote.pkl") # new, non-SMOTE
MIMIC_CSV = os.path.join(DATA_DIR, "MIMIC-IVdata-1775367119727.csv")
TARGET    = "extubation_failure"
RNG       = 42
# 01 Stage-5 deployed config, MINUS the two SMOTE lines:
DEPLOY_HP = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 eval_metric="logloss", n_jobs=-1, random_state=RNG, verbosity=0)
# -----------------------------------------------------------------------------

bundle = pickle.load(open(PKL_IN, "rb"))
feats  = list(bundle["features"])
imp    = bundle["imputer"]                       # median imputer fit on original MIMIC
print(f">>> loaded bundle: {len(feats)} features, imputer={type(imp).__name__}")

df = pd.read_csv(MIMIC_CSV)
missing = [f for f in feats if f not in df.columns]
if missing:
    sys.exit(f"[error] MIMIC csv is missing model features: {missing}")
y = df[TARGET].astype(int).values
Xraw = df[feats].apply(pd.to_numeric, errors="coerce")
X = imp.transform(Xraw.values)                   # exactly 01 Stage-5 imputation (pkl medians)
print(f">>> MIMIC: n={len(y)}, events={int(y.sum())}, prevalence={y.mean():.4f}")

# ---- refit the DEPLOYED model on full MIMIC, NO SMOTE -----------------------
model = XGBClassifier(**DEPLOY_HP)
model.fit(X, y)                                  # natural prevalence
print(">>> refit deployed XGBoost on full MIMIC (no SMOTE) -- done")

# ---- write the new bundle (same structure, model swapped) -------------------
new = copy.deepcopy(bundle)
new["model"] = model
md = new.get("metadata", {})
md["training_resampling"] = "none (natural prevalence; SMOTE removed for Route B)"
md["deployed_hp"] = DEPLOY_HP
new["metadata"] = md
with open(PKL_OUT, "wb") as f:
    pickle.dump(new, f)
print(f">>> wrote {PKL_OUT}")

# ---- sanity: 5-fold OOF with the SAME deployed config, no SMOTE -------------
# (this is a SANITY read on the calibration mechanism; the FINAL Table-2 OOF
#  in Step 3 will use 01's make_models XGBoost config, not necessarily ss/cs 0.8)
def cal_slope_intercept(yv, pv, eps=1e-6):
    z = np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z, yv)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def ece(yv, pv, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); tot = 0.0
    for i in range(n_bins):
        m = (pv >= edges[i]) & (pv < edges[i + 1] if i < n_bins - 1 else pv <= edges[i + 1])
        if m.sum():
            tot += (m.sum() / len(pv)) * abs(yv[m].mean() - pv[m].mean())
    return tot

oof = np.zeros(len(y))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=RNG).split(X, y):
    m = XGBClassifier(**DEPLOY_HP).fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
sl, ic = cal_slope_intercept(y, oof)
print("\n=== non-SMOTE deployed-config 5-fold OOF (MIMIC) -- SANITY ===")
print(f"  AUROC={roc_auc_score(y, oof):.3f}  intercept={ic:.3f}  slope={sl:.3f}  "
      f"Brier={brier_score_loss(y, oof):.3f}  ECE={ece(y, oof):.3f}")
print("  expect: intercept ~= -0.2 (NOT -0.72) -> confirms SMOTE was the intercept driver")

print("\nNext:")
print("  1) point analysis_1c.py at the new pkl:")
print(f'     sed -i \'\' \'s#xgb_extubation_failure_v2.pkl#xgb_extubation_failure_v2_nonsmote.pkl#\' \\')
print(f'       "$HOME/Desktop/data\u6316\u6398/extubation-failure-predictor/code/analysis_1c.py"  (or wherever your analysis_1c.py lives)')
print("  2) re-run analysis_1c.py and paste C1 / C2 / R0(99th) / R1 / R2 + the two C2 arms + the C1 prevalence check.")
