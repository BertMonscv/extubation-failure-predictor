#!/usr/bin/env python3
# =============================================================================
# step3_tables_nonsmote.py   (Route B, Step 3 -- part 1: Table 2 + Table 7 + thresholds)
#
# Faithful re-run of 01_pipeline_mimic.py's Stage-3 train_cv WITHOUT SMOTE, on the
# FROZEN 28 features, reproducing 01 exactly except the three SMOTE lines are gone:
#   - StratifiedKFold(5, shuffle, RNG=42); per-fold median impute; StandardScaler
#     for {LR,SVM,MLP}; the EXACT make_models() configs.
# Produces:
#   * Table 2 MIMIC row  (XGBoost OOF: AUROC, AUPRC, Brier, slope, intercept, ECE)
#   * Table 7            (all 7 model families: OOF AUROC + 95% CI, AUPRC)
#   * Youden + Sens-80 thresholds from the XGBoost OOF, with Sens/Spec/PPV/NPV
#   * regenerated mimic_oof_predictions.csv  (y_true + 7 model columns; feeds 06/Fig 2)
#   * Table 2 eICU rows  (full + clean) by scoring the NON-SMOTE deployed pkl on eICU
#
# Does NOT do Table 6 (recalibration), Table 3 (eICU operating-threshold chars),
# Table 4 (DCA) or sex/age subgroups -- those are step3_tables_nonsmote_part2.
#
#   conda activate extub
#   python ~/Downloads/step3_tables_nonsmote.py
# =============================================================================
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")

# ------------------------- CONFIG -------------------------------------------
DATA_DIR    = "/Users/ilizyue/Documents/xgb_extubation_failure/data"
PKL_NONSMOTE= os.path.join(DATA_DIR, "xgb_extubation_failure_v2_nonsmote.pkl")
MIMIC_CSV   = os.path.join(DATA_DIR, "MIMIC-IVdata-1775367119727.csv")
EICU_CSV    = os.path.join(DATA_DIR, "eICUdata-1775370213861.csv")
OOF_OUT     = os.path.join(DATA_DIR, "mimic_oof_predictions.csv")   # overwrites; back up first if wanted
TARGET      = "extubation_failure"
MV_COL      = "mv_duration_hours"
RNG         = 42
TARGET_SENS = 0.80
B_BOOT      = 1000
LEAK_COLS = [
    "reintubated_48h", "hours_to_reintubation", "death_within_48h_of_extubation",
    "n_invasive_episodes", "hospital_mortality", "icu_mortality", "hospital_los_days",
    "icu_los_days", "source", "mortality_28d", "mortality_90d", "first_extubation_time",
    "subject_id", "hadm_id", "stay_id", "icu_intime", "icu_outtime", "admittime",
    "dischtime", "rrt_first_time",
]
SCALE_NEEDED = {"LR", "SVM", "MLP"}

def make_models():   # verbatim from 01_pipeline_mimic.py make_models()
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=RNG),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=RNG),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", use_label_encoder=False,
                                 n_jobs=-1, random_state=RNG, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   n_jobs=-1, random_state=RNG, verbosity=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RNG),
        "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                       random_seed=RNG, verbose=0),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             early_stopping=True, random_state=RNG),
    }

# ------------------------- helpers ------------------------------------------
def encode_and_filter(df, feat_template):
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    for c in feat_template:
        if c not in d.columns:
            d[c] = np.nan
    return d[feat_template]

def cal_slope_intercept(y, p, eps=1e-6):
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
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

def auroc_auprc_ci(y, p, B=B_BOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    au, ap = [], []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        au.append(roc_auc_score(y[idx], p[idx])); ap.append(average_precision_score(y[idx], p[idx]))
    f = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return (roc_auc_score(y, p), f(au), average_precision_score(y, p), f(ap))

def table2_row(tag, y, p):
    auroc, auci, auprc, apci = auroc_auprc_ci(y, p)
    sl, ic = cal_slope_intercept(y, p)
    print(f"  {tag:18s} n={len(y):5d} ev={int(y.sum()):4d}  "
          f"AUROC={auroc:.3f} ({auci[0]:.3f},{auci[1]:.3f})  "
          f"AUPRC={auprc:.3f} ({apci[0]:.3f},{apci[1]:.3f})  "
          f"Brier={brier_score_loss(y,p):.3f}  slope={sl:.3f}  intercept={ic:.3f}  ECE={ece(y,p):.3f}")

def operating(tag, y, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    flagged = yp.mean()
    print(f"  {tag:16s} thr={thr:.4f}  Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} NPV={npv:.3f} Flagged={flagged:.3f}")

# ------------------------- MIMIC: 28-feature matrix (== 01) ------------------
mimic = pd.read_csv(MIMIC_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
bundle = pickle.load(open(PKL_NONSMOTE, "rb"))
feats  = list(bundle["features"]); model_dep = bundle["model"]; imp_dep = bundle["imputer"]
yM = mimic[TARGET].astype(int).values
XM = encode_and_filter(mimic.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in mimic.columns]),
                       feat_template=feats).values
print(f">>> MIMIC n={len(yM)} ev={int(yM.sum())} prev={yM.mean():.4f} | features={len(feats)}")

# ------------------------- Stage-3 train_cv MINUS SMOTE ----------------------
print(">>> running 5-fold OOF for 7 models (no SMOTE) ...")
names = list(make_models().keys())
oof = {n: np.zeros(len(yM)) for n in names}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
for fold, (tr, va) in enumerate(skf.split(XM, yM), 1):
    imp = SimpleImputer(strategy="median").fit(XM[tr])
    Xtr_i, Xva_i = imp.transform(XM[tr]), imp.transform(XM[va])
    sc = StandardScaler().fit(Xtr_i)
    Xtr_s, Xva_s = sc.transform(Xtr_i), sc.transform(Xva_i)
    for name, mdl in make_models().items():
        if name in SCALE_NEEDED:
            mdl.fit(Xtr_s, yM[tr]);  oof[name][va] = mdl.predict_proba(Xva_s)[:, 1]
        else:
            mdl.fit(Xtr_i, yM[tr]);  oof[name][va] = mdl.predict_proba(Xva_i)[:, 1]
    print(f"    fold {fold}/5 done")

pd.DataFrame({"y_true": yM, **oof}).to_csv(OOF_OUT, index=False)
print(f">>> wrote {OOF_OUT}")

# ------------------------- Table 7 (7 models) --------------------------------
print("\n=== TABLE 7  (MIMIC pooled OOF, no SMOTE) ===")
order = sorted(names, key=lambda n: roc_auc_score(yM, oof[n]), reverse=True)
for n in order:
    auroc, auci, auprc, apci = auroc_auprc_ci(yM, oof[n])
    print(f"  {n:9s} AUROC={auroc:.3f} ({auci[0]:.3f},{auci[1]:.3f})  AUPRC={auprc:.3f} ({apci[0]:.3f},{apci[1]:.3f})")

# ------------------------- Table 2 MIMIC row (XGBoost OOF) -------------------
print("\n=== TABLE 2  (calibration + discrimination) ===")
table2_row("MIMIC (XGB OOF)", yM, oof["XGBoost"])

# ------------------------- eICU scoring with the non-SMOTE deployed model ----
eicu = pd.read_csv(EICU_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
yE = eicu[TARGET].astype(int).values
XE_raw = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]),
                           feat_template=feats)
clean_mask = (pd.to_numeric(eicu[MV_COL], errors="coerce") > 0).to_numpy()  # positive recorded mv
XE_full = XE_raw.copy()
XE_full.loc[pd.to_numeric(XE_full[MV_COL], errors="coerce") <= 0, MV_COL] = np.nan  # v2 mv rule
pE_full  = model_dep.predict_proba(imp_dep.transform(XE_full.values))[:, 1]
pE_clean = pE_full[clean_mask]
yE_clean = yE[clean_mask]
table2_row("eICU full",  yE,       pE_full)
table2_row("eICU clean", yE_clean, pE_clean)
pd.DataFrame({"y_true": yE, "prob": pE_full}).to_csv(
    os.path.join(DATA_DIR, "eicu_predictions_full_nonsmote.csv"), index=False)

# ------------------------- operating thresholds (from XGBoost MIMIC OOF) -----
def youden_thr(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p); return float(thr[np.argmax(tpr - fpr)])
def sens_thr(y, p, t=TARGET_SENS):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p); ok = tpr >= t
    if not ok.any(): return float(thr[np.argmax(tpr)])
    idx = np.where(ok)[0]; return float(thr[idx[np.argmin(fpr[idx])]])

thr_y  = youden_thr(yM, oof["XGBoost"])
thr_s  = sens_thr(yM, oof["XGBoost"])
print("\n=== OPERATING THRESHOLDS  (XGBoost, derived on MIMIC OOF) ===")
print("  -- MIMIC --")
operating("Youden",  yM, oof["XGBoost"], thr_y)
operating("Sens-80", yM, oof["XGBoost"], thr_s)
print("  -- eICU full (same thresholds) --")
operating("Youden",  yE, pE_full, thr_y)
operating("Sens-80", yE, pE_full, thr_s)
print("  -- eICU clean (same thresholds) --")
operating("Youden",  yE_clean, pE_clean, thr_y)
operating("Sens-80", yE_clean, pE_clean, thr_s)
print("\n(Thresholds: Youden=%.4f  Sens-80=%.4f)" % (thr_y, thr_s))
print("\nDONE. Paste the TABLE 2 / TABLE 7 / OPERATING blocks back.")
