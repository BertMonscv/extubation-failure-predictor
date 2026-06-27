"""
05_compute_shap.py
==================
Compute path-dependent TreeSHAP for the v2 XGBoost model on:
  - a stratified subsample of the MIMIC-IV training cohort (n = 1,500)
  - the full eICU external validation cohort (n = 1,232)

Outputs (results/):
  shap_mimic.npy, shap_eicu.npy     SHAP values, logit space  (n × 28)
  X_mimic.npy,    X_eicu.npy        imputed feature matrices (n × 28)
  shap_values_mimic.csv, .csv       same SHAP values as CSV (with column names)

The output `.npy` files are consumed by `06_make_figures.py`. They are
committed to the repository so that figures can be regenerated without
credentialed-data access.

Requires:
  - models/xgb_extubation_failure_v2.pkl                          (in repo)
  - data/raw/MIMIC-IVdata.csv                                     (PhysioNet)
  - data/raw/eICUdata.csv                                         (PhysioNet)

Run from repository root:
    python code/05_compute_shap.py
"""
import os
import pickle
import numpy as np
import pandas as pd

# Real libraries — no sandbox fallback needed in a normal Python environment
import shap            # >= 0.43
import xgboost as xgb  # noqa: F401 (needed for unpickling the bundle)


# ----- Paths (relative to repo root) -----
BUNDLE_PATH = "models/xgb_extubation_failure_v2.pkl"
MIMIC_RAW   = "data/raw/MIMIC-IVdata.csv"
EICU_RAW    = "data/raw/eICUdata.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main() -> None:
    # ---------- Load the v2 bundle ----------
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    imputer  = bundle["imputer"]
    features = list(bundle["features"])
    print(f"Loaded bundle: {len(features)} features, "
          f"{model.n_estimators} trees.")

    # ---------- Preprocessing helper (matches code/03_predict.py) ----------
    def prepare(df: pd.DataFrame, apply_mv_clean: bool) -> np.ndarray:
        df = df.copy()
        if apply_mv_clean:
            df.loc[df["mv_duration_hours"] <= 0, "mv_duration_hours"] = np.nan
        cat = df.select_dtypes(include=["object"]).columns.tolist()
        df_enc = pd.get_dummies(df, columns=cat, drop_first=True)
        for f in features:
            if f not in df_enc.columns:
                df_enc[f] = np.nan
        X_raw = df_enc[features].apply(pd.to_numeric, errors="coerce").values
        return imputer.transform(X_raw)

    # ---------- eICU validation cohort (full 1,232) ----------
    print("\nLoading eICU…")
    eicu_raw = pd.read_csv(EICU_RAW)
    eicu_df  = eicu_raw[eicu_raw["extubation_failure"].notna()].reset_index(drop=True)
    print(f"  eICU validation cohort: n={len(eicu_df)}, "
          f"prevalence={eicu_df['extubation_failure'].mean():.3f}")
    X_eicu = prepare(eicu_df, apply_mv_clean=True)

    # ---------- MIMIC training cohort (stratified subsample) ----------
    print("\nLoading MIMIC-IV…")
    mimic_raw = pd.read_csv(MIMIC_RAW)
    rng = np.random.RandomState(42)
    pos_idx = np.where(mimic_raw["extubation_failure"].values == 1)[0]
    neg_idx = np.where(mimic_raw["extubation_failure"].values == 0)[0]
    n_pos  = len(pos_idx)
    n_neg  = 1500 - n_pos
    neg_s  = rng.choice(neg_idx, size=n_neg, replace=False)
    keep   = np.sort(np.concatenate([pos_idx, neg_s]))
    mimic_sub = mimic_raw.iloc[keep].reset_index(drop=True)
    print(f"  MIMIC subsample: n={len(mimic_sub)} "
          f"({n_pos} positives + {n_neg} negatives, "
          f"prevalence={mimic_sub['extubation_failure'].mean():.3f})")
    X_mimic = prepare(mimic_sub, apply_mv_clean=False)

    # ---------- TreeSHAP ----------
    print("\nBuilding TreeExplainer…")
    explainer = shap.TreeExplainer(model)

    print(f"Computing SHAP on eICU (n={len(X_eicu)})…")
    sv_eicu = explainer.shap_values(X_eicu)
    if isinstance(sv_eicu, list):  # some shap/xgb versions return a 2-list for binary
        sv_eicu = sv_eicu[1] if len(sv_eicu) == 2 else sv_eicu[0]
    sv_eicu = np.asarray(sv_eicu)

    print(f"Computing SHAP on MIMIC subsample (n={len(X_mimic)})…")
    sv_mimic = explainer.shap_values(X_mimic)
    if isinstance(sv_mimic, list):
        sv_mimic = sv_mimic[1] if len(sv_mimic) == 2 else sv_mimic[0]
    sv_mimic = np.asarray(sv_mimic)

    # ---------- Save ----------
    np.save(f"{RESULTS_DIR}/shap_eicu.npy", sv_eicu)
    np.save(f"{RESULTS_DIR}/shap_mimic.npy", sv_mimic)
    np.save(f"{RESULTS_DIR}/X_eicu.npy", X_eicu)
    np.save(f"{RESULTS_DIR}/X_mimic.npy", X_mimic)
    pd.DataFrame(sv_eicu, columns=features).to_csv(
        f"{RESULTS_DIR}/shap_values_eicu.csv", index=False)
    pd.DataFrame(sv_mimic, columns=features).to_csv(
        f"{RESULTS_DIR}/shap_values_mimic.csv", index=False)
    print(f"\nSaved 6 files under {RESULTS_DIR}/")

    # ---------- Headline check ----------
    mae_e = np.abs(sv_eicu).mean(axis=0)
    mae_m = np.abs(sv_mimic).mean(axis=0)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(mae_m, mae_e)
    print(f"\nCross-database SHAP-rank Spearman ρ = {rho:.3f}")
    print("\nTop 10 features by combined mean |SHAP|:")
    order = np.argsort(-(mae_m + mae_e))
    for j in order[:10]:
        print(f"  {features[j]:<28}  MIMIC={mae_m[j]:.3f}  eICU={mae_e[j]:.3f}")


if __name__ == "__main__":
    main()
