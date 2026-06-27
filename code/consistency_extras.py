#!/usr/bin/env python3
# =============================================================================
# consistency_extras_nonsmote.py   (Route B, Step 3 -- consistency extras)
#
# Reuses analysis_1c's own primitives + the non-SMOTE frozen model to produce the
# three remaining consistency numbers that the narrative needs:
#   (1) naive cross-database mean|SHAP| agreement (Fig 4c / start of the consistency
#       section): Spearman rho AND Pearson r of log-transformed mean|SHAP|.
#   (2) independent logistic-regression cross-database agreement: Spearman rho of
#       |standardised coefficients| + fraction of shared predictors with agreeing sign
#       (replaces the old "rho approx 0.22").
#   (3) C1 prevalence sweep 4%-23% (analysis_1c uses a fixed 0.13; this tests whether
#       the "C1 insensitive to prevalence" sentence can be kept).
#
#   conda activate extub
#   cd ~/Documents/xgb_extubation_failure/data
#   python ~/Downloads/consistency_extras_nonsmote.py
# =============================================================================
import os, sys, warnings
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

DATA_DIR = __import__("os").path.expanduser("~/Documents/xgb_extubation_failure/data")
sys.path.insert(0, DATA_DIR)
import analysis_1c as A   # MODEL_M_PATH already points at the non-SMOTE pkl

feats = A.FEATURES
print(">>> discovering data + loading non-SMOTE model")
X_M, y_M, X_E, y_E, model_path = A.discover(DATA_DIR)
model_M = A.load_model_M(model_path, X_M, y_M)

# ---- (1) naive cross-database mean|SHAP| agreement (Fig 4c) -----------------
absM = A.shap_abs_matrix(model_M, X_M[feats]).mean(axis=0)
absE = A.shap_abs_matrix(model_M, X_E[feats]).mean(axis=0)
eps = 1e-12
sp = spearmanr(absM, absE).correlation
pl = pearsonr(np.log(absM + eps), np.log(absE + eps))[0]
print("\n=== (1) NAIVE cross-database mean|SHAP| (fixed non-SMOTE model) ===")
print(f"  Spearman rho            = {sp:.3f}")
print(f"  Pearson r (log mean|SHAP|) = {pl:.3f}    <- replaces the manuscript's r = 0.86")

# ---- (2) independent LR cross-database standardized-coefficient agreement ----
def lr_std_coef(Xdf, y):
    X = Xdf[feats].values.astype(float)
    valid = ~np.all(np.isnan(X), axis=0)              # columns with >=1 non-NaN in this DB
    Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    lr = LogisticRegression(max_iter=2000, random_state=A.SEED).fit(Xs, np.asarray(y))
    return lr.coef_[0], valid
cM, vM = lr_std_coef(X_M, y_M)
cE, vE = lr_std_coef(X_E, y_E)
mask = vM & vE                                         # predictors informative in BOTH databases
rho_imp    = spearmanr(np.abs(cM[mask]), np.abs(cE[mask])).correlation
rho_signed = spearmanr(cM[mask], cE[mask]).correlation
sign_agr   = float(np.mean(np.sign(cM[mask]) == np.sign(cE[mask])))
dropped = [feats[i] for i in range(len(feats)) if not mask[i]]
print("\n=== (2) independent LR cross-database (no SMOTE) ===")
print(f"  shared informative predictors = {int(mask.sum())} of {len(feats)} (dropped all-NaN-in-one-DB: {dropped})")
print(f"  Spearman rho of |standardised coef| (importance) = {rho_imp:.3f}   <- replaces 'rho approx 0.22'")
print(f"  Spearman rho of signed coefficients              = {rho_signed:.3f}")
print(f"  fraction of shared predictors with agreeing sign = {sign_agr:.2f}")

# ---- (3) C1 prevalence sweep 4%-23% -----------------------------------------
# C1 = rho(S_M, S_M->E), both mean|SHAP| rankings composed at a common prevalence.
absM_mat = A.shap_abs_matrix(model_M, X_M[feats])
absE_mat = A.shap_abs_matrix(model_M, X_E[feats])
print("\n=== (3) C1 prevalence sweep (compose at common event rate) ===")
print("  prev    C1(Spearman)")
c1s = []
for prev in [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.19, 0.23]:
    rng = np.random.default_rng(A.SEED + 7)
    S_M  = A.compose_average(absM_mat, y_M, prev, rng, A.N_REPEATS)
    S_ME = A.compose_average(absE_mat, y_E, prev, rng, A.N_REPEATS)
    c1 = spearmanr(S_M, S_ME).correlation
    c1s.append(c1)
    print(f"  {prev:.2f}    {c1:.3f}")
print(f"  -> C1 range across 4%-23%: {min(c1s):.3f} to {max(c1s):.3f}  (spread {max(c1s)-min(c1s):.3f})")
print("\nDONE. Paste blocks (1),(2),(3).")
