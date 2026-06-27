#!/usr/bin/env python3
"""
prep_figure_inputs_nonsmote.py
Regenerate every input that 06_make_figures.py consumes, under the NON-SMOTE
pipeline, writing to the exact paths/columns the figure script expects:

  <outroot>/data/predictions/mimic_oof_predictions.csv     (y_true, XGBoost)
  <outroot>/data/predictions/eicu_predictions_full_1232.csv (y_true, prob)
  <outroot>/data/predictions/eicu_mv_positive_mask.npy      (1232 bool, mv>0)
  <outroot>/results/shap_mimic.npy  X_mimic.npy   (1500 x 28, deployed-model SHAP)
  <outroot>/results/shap_eicu.npy   X_eicu.npy    (1232 x 28)
  <outroot>/results/features.txt                  (28 names, model order)

NOT regenerated (data properties, SMOTE-independent — keep your existing copy):
  data/univariate_auroc_mimic_vs_eicu.csv

Sanity targets it should print: MIMIC OOF AUROC 0.872, eICU full 0.598, clean 0.815,
cross-DB mean|SHAP| Spearman ~0.89.

  conda activate extub
  cd ~/Desktop/data挖掘/extubation-failure-predictor      (or wherever you run 06)
  python prep_figure_inputs_nonsmote.py \
      --datadir ~/Documents/xgb_extubation_failure/data --outroot .
"""
import os, argparse, pickle, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

TARGET, MV_COL, RNG, N_MIMIC_SHAP = "extubation_failure", "mv_duration_hours", 42, 1500
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

def encode_and_filter(df, feats):
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    for c in feats:
        if c not in d.columns: d[c] = np.nan
    return d[feats]

def main(a):
    pred_dir = os.path.join(a.outroot, "data", "predictions")
    res_dir  = os.path.join(a.outroot, "results")
    os.makedirs(pred_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)

    bundle = pickle.load(open(os.path.join(a.datadir, "xgb_extubation_failure_v2_nonsmote.pkl"), "rb"))
    feats, model_dep, imp_dep = list(bundle["features"]), bundle["model"], bundle["imputer"]
    open(os.path.join(res_dir, "features.txt"), "w").write("\n".join(feats) + "\n")

    mimic = pd.read_csv(os.path.join(a.datadir, "MIMIC-IVdata-1775367119727.csv")).dropna(subset=[TARGET]).reset_index(drop=True)
    eicu  = pd.read_csv(os.path.join(a.datadir, "eICUdata-1775370213861.csv")).dropna(subset=[TARGET]).reset_index(drop=True)
    yM = mimic[TARGET].astype(int).values
    yE = eicu[TARGET].astype(int).values
    XM = encode_and_filter(mimic.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in mimic.columns]), feats)
    XE = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]), feats)

    # ---- MIMIC out-of-fold (XGBoost, 5-fold, non-SMOTE) -> reproduces 0.872 ----
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    oof = np.full(len(yM), np.nan); Xv = XM.values.astype(float)
    for tr, va in skf.split(Xv, yM):
        imp = SimpleImputer(strategy="median").fit(Xv[tr])
        m = XGBClassifier(**XGB_HP).fit(imp.transform(Xv[tr]), yM[tr])
        oof[va] = m.predict_proba(imp.transform(Xv[va]))[:, 1]
    pd.DataFrame({"y_true": yM, "XGBoost": oof}).to_csv(
        os.path.join(pred_dir, "mimic_oof_predictions.csv"), index=False)
    print(f"  MIMIC OOF AUROC = {roc_auc_score(yM, oof):.3f}   (target 0.872)")

    # ---- eICU full predictions (deployed model) -> reproduces 0.598 ----
    XE_dep = XE.copy(); XE_dep.loc[XE_dep[MV_COL] <= 0, MV_COL] = np.nan
    pE = model_dep.predict_proba(imp_dep.transform(XE_dep.values))[:, 1]
    pd.DataFrame({"y_true": yE, "prob": pE}).to_csv(
        os.path.join(pred_dir, "eicu_predictions_full_1232.csv"), index=False)
    mask = (XE[MV_COL] > 0).values
    np.save(os.path.join(pred_dir, "eicu_mv_positive_mask.npy"), mask)
    print(f"  eICU full AUROC = {roc_auc_score(yE, pE):.3f}   (target 0.598) | "
          f"clean AUROC = {roc_auc_score(yE[mask], pE[mask]):.3f}  (target 0.815) | clean n={int(mask.sum())}")

    # ---- SHAP arrays (deployed model; TreeExplainer) ----
    import shap
    XM_imp = imp_dep.transform(XM.values)
    rs = np.random.RandomState(RNG)
    idx = rs.choice(len(XM_imp), size=min(N_MIMIC_SHAP, len(XM_imp)), replace=False)
    XM_s = XM_imp[idx]
    XE_imp = imp_dep.transform(XE_dep.values)
    expl = shap.TreeExplainer(model_dep)
    sv_m = np.asarray(expl.shap_values(XM_s)); sv_e = np.asarray(expl.shap_values(XE_imp))
    if sv_m.ndim == 3: sv_m = sv_m[..., 1]          # guard older shap output shapes
    if sv_e.ndim == 3: sv_e = sv_e[..., 1]
    np.save(os.path.join(res_dir, "shap_mimic.npy"), sv_m)
    np.save(os.path.join(res_dir, "X_mimic.npy"),  XM_s)
    np.save(os.path.join(res_dir, "shap_eicu.npy"), sv_e)
    np.save(os.path.join(res_dir, "X_eicu.npy"),   XE_imp)
    rho, _ = spearmanr(np.abs(sv_m).mean(0), np.abs(sv_e).mean(0))
    print(f"  SHAP written: MIMIC {sv_m.shape}, eICU {sv_e.shape} | "
          f"cross-DB mean|SHAP| Spearman = {rho:.3f}  (target ~0.89)")
    print("\nDONE. Now run:  python 06_make_figures_nonsmote.py")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=os.path.expanduser("~/Documents/xgb_extubation_failure/data"))
    ap.add_argument("--outroot", default=".")
    main(ap.parse_args())
