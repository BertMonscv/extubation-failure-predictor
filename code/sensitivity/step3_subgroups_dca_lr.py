#!/usr/bin/env python3
# =============================================================================
# step3_subgroups_dca_lr_nonsmote.py   (Route B, Step 3 -- part 3)
#
# Non-SMOTE regeneration of the remaining numeric pieces:
#   * sex / age subgroups  (auto-detected columns, as in the rerun harness:
#     sex in ['sex','gender']; age in ['age','anchor_age']; eICU '>89' -> 90)
#       - MIMIC: XGBoost OOF AUROC by subgroup; eICU full: scored AUROC by subgroup
#   * TABLE 3 : operating-threshold characteristics at the NEW thresholds
#       (Youden + Sens-80 from XGBoost MIMIC OOF, + Default 0.5) x 3 cohorts
#       Sens/Spec/PPV/NPV/F1/NetBenefit/Flagged
#   * TABLE 4 : DCA net benefit at six standard thresholds p in
#       {0.03,0.05,0.10,0.15,0.20,0.30} x 3 cohorts  (model NB / treat-all NB)
#   * AF6     : logistic-regression comparator refit on full MIMIC, scored on eICU
#       full/clean -> AUROC + calibration slope/intercept
#
#   conda activate extub
#   python ~/Downloads/step3_subgroups_dca_lr_nonsmote.py
# =============================================================================
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss, confusion_matrix, roc_curve,
                             f1_score)
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA_DIR  = "/Users/ilizyue/Documents/xgb_extubation_failure/data"
PKL       = os.path.join(DATA_DIR, "xgb_extubation_failure_v2_nonsmote.pkl")
MIMIC_CSV = os.path.join(DATA_DIR, "MIMIC-IVdata-1775367119727.csv")
EICU_CSV  = os.path.join(DATA_DIR, "eICUdata-1775370213861.csv")
TARGET, MV_COL, RNG = "extubation_failure", "mv_duration_hours", 42
DCA_PTS = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
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
    rng = np.random.default_rng(seed); n = len(y); a = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n: a.append(roc_auc_score(y[idx], p[idx]))
    return roc_auc_score(y, p), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def cal_si(y, p, eps=1e-6):
    z = np.log(np.clip(p, eps, 1-eps)/(1-np.clip(p, eps, 1-eps))).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def net_benefit(y, p, pt):
    yp = (p >= pt).astype(int); n = len(y)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    return tp/n - fp/n * (pt/(1-pt))
def treatall_nb(y, pt):
    prev = y.mean(); return prev - (1-prev) * (pt/(1-pt))

def op_chars(y, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0.0; spec = tn/(tn+fp) if (tn+fp) else 0.0
    ppv = tp/(tp+fp) if (tp+fp) else 0.0;  npv = tn/(tn+fn) if (tn+fn) else 0.0
    return dict(Sens=sens, Spec=spec, PPV=ppv, NPV=npv, F1=f1_score(y, yp, zero_division=0),
                NB=net_benefit(y, p, thr), Flagged=yp.mean())

def detect(df, names):
    for n in names:
        if n in df.columns: return df[n], n
    return None, None

def age_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[>＞]\s*89", "90", regex=True), errors="coerce").to_numpy()

# ---- load + matrices --------------------------------------------------------
bundle = pickle.load(open(PKL, "rb"))
feats, model_dep, imp_dep = list(bundle["features"]), bundle["model"], bundle["imputer"]
mimic = pd.read_csv(MIMIC_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
eicu  = pd.read_csv(EICU_CSV).dropna(subset=[TARGET]).reset_index(drop=True)
yM = mimic[TARGET].astype(int).values
yE = eicu[TARGET].astype(int).values
XM = encode_and_filter(mimic.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in mimic.columns]), feats).values
XE = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]), feats)
clean_mask = (pd.to_numeric(eicu[MV_COL], errors="coerce") > 0).to_numpy()
XE.loc[pd.to_numeric(XE[MV_COL], errors="coerce") <= 0, MV_COL] = np.nan
pE_full = model_dep.predict_proba(imp_dep.transform(XE.values))[:, 1]
pE_clean, yE_clean = pE_full[clean_mask], yE[clean_mask]

# ---- XGBoost OOF on MIMIC (no SMOTE, RNG=42; matches part 1) -----------------
oof = np.zeros(len(yM))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=RNG).split(XM, yM):
    imp = SimpleImputer(strategy="median").fit(XM[tr])
    m = XGBClassifier(**XGB_HP).fit(imp.transform(XM[tr]), yM[tr])
    oof[va] = m.predict_proba(imp.transform(XM[va]))[:, 1]

def youden(y, p): fpr, tpr, t = roc_curve(y, p); return float(t[np.argmax(tpr-fpr)])
def sens80(y, p, tgt=0.80):
    fpr, tpr, t = roc_curve(y, p); ok = tpr >= tgt
    return float(t[np.argmax(tpr)]) if not ok.any() else float(t[np.where(ok)[0][np.argmin(fpr[np.where(ok)[0]])]])
thr_y, thr_s = youden(yM, oof), sens80(yM, oof)

# ---- (A) subgroups ----------------------------------------------------------
print("=== SUBGROUPS (sex, age) -- AUROC (95% CI), events ===")
def subg_block(tag, y, p, df):
    sexcol, sn = detect(df, ["sex", "gender"]); agecol, an = detect(df, ["age", "anchor_age"])
    print(f"  [{tag}] sex_col={sn!r} age_col={an!r}")
    if sexcol is not None:
        male = (sexcol.astype(str).str.upper().str[0] == "M").to_numpy()
        for lab, mask in (("male", male), ("female", ~male)):
            if 0 < y[mask].sum() < mask.sum():
                a, lo, hi = auroc_ci(y[mask], p[mask])
                print(f"    {lab:7s} AUROC={a:.3f} ({lo:.3f},{hi:.3f})  ev={int(y[mask].sum())}  n={int(mask.sum())}")
    if agecol is not None:
        ag = age_numeric(agecol); lt = ag < 65
        for lab, mask in (("lt65", lt & ~np.isnan(ag)), ("ge65", (~lt) & ~np.isnan(ag))):
            if 0 < y[mask].sum() < mask.sum():
                a, lo, hi = auroc_ci(y[mask], p[mask])
                print(f"    {lab:7s} AUROC={a:.3f} ({lo:.3f},{hi:.3f})  ev={int(y[mask].sum())}  n={int(mask.sum())}")
subg_block("MIMIC internal OOF", yM, oof, mimic)
subg_block("eICU full",          yE, pE_full, eicu)

# ---- (B) Table 3 ------------------------------------------------------------
print(f"\n=== TABLE 3  (thresholds: Sens-80={thr_s:.4f}, Youden={thr_y:.4f}, Default=0.5) ===")
for ctag, (y, p) in [("MIMIC-IV CV", (yM, oof)), ("eICU full", (yE, pE_full)), ("eICU clean", (yE_clean, pE_clean))]:
    for tlab, thr in [("Sens-80", thr_s), ("Youden", thr_y), ("Default", 0.5)]:
        d = op_chars(y, p, thr)
        print(f"  {ctag:12s} {tlab:8s} Sens={d['Sens']:.3f} Spec={d['Spec']:.3f} PPV={d['PPV']:.3f} "
              f"NPV={d['NPV']:.3f} F1={d['F1']:.3f} NB={d['NB']:+.3f} Flagged={100*d['Flagged']:.1f}%")

# ---- (C) Table 4 DCA --------------------------------------------------------
print(f"\n=== TABLE 4  DCA net benefit (model / treat-all) at p in {DCA_PTS} ===")
for ctag, (y, p) in [("MIMIC-IV CV", (yM, oof)), ("eICU full", (yE, pE_full)), ("eICU clean", (yE_clean, pE_clean))]:
    cells = "  ".join(f"{net_benefit(y,p,pt):+.3f}/{treatall_nb(y,pt):+.3f}" for pt in DCA_PTS)
    print(f"  {ctag:12s} prev={y.mean():.3f}  {cells}")

# ---- (D) AF6: LR comparator refit on full MIMIC, scored on eICU -------------
imp_lr = SimpleImputer(strategy="median").fit(XM)
sc_lr  = StandardScaler().fit(imp_lr.transform(XM))
lr = LogisticRegression(max_iter=2000, random_state=RNG).fit(sc_lr.transform(imp_lr.transform(XM)), yM)
def lr_score(Xdf_or_arr):
    arr = Xdf_or_arr.values if hasattr(Xdf_or_arr, "values") else Xdf_or_arr
    return lr.predict_proba(sc_lr.transform(imp_lr.transform(arr)))[:, 1]
lrE_full  = lr_score(XE)
lrE_clean = lrE_full[clean_mask]
print("\n=== AF6  LR comparator (refit full MIMIC, no SMOTE) -> eICU ===")
for tag, y, p in [("eICU full", yE, lrE_full), ("eICU clean", yE_clean, lrE_clean)]:
    a, lo, hi = auroc_ci(y, p); sl, ic = cal_si(y, p)
    print(f"  {tag:11s} AUROC={a:.3f} ({lo:.3f},{hi:.3f})  slope={sl:.3f}  intercept={ic:+.3f}  Brier={brier_score_loss(y,p):.3f}")
print("\nDONE. Paste SUBGROUPS / TABLE 3 / TABLE 4 / AF6 blocks.")
