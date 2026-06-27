"""
External sensitivity: eICU AUROC (full + clean) with inr_max DROPPED from the
frozen deployed model (xgb_extubation_failure_v2_nonsmote.pkl).

"Dropping" a feature from an already-trained 28-feature XGBoost = neutralising it
at inference: inr_max is set to missing and the bundled (training-median) imputer
fills it, so every eICU patient receives the same inr_max value. A feature that is
constant across the evaluation set contributes a uniform logit offset and therefore
cannot change the RANK ordering -> AUROC is unchanged. If inr_max is already
constant in eICU (e.g. not collected and fully imputed), the change is exactly 0.

This is the frozen-model ablation, NOT a 27-feature retrain (a retrain would
redistribute inr_max's role and the "constant in eICU" argument would not apply).

Preprocessing (encode_and_filter, the v2 mv rule, the clean mask, the bundled
imputer) is copied verbatim from prep_figure_inputs_nonsmote.py so the baseline
reproduces the manuscript's 0.598 (full) and 0.815 (clean) exactly.

Run (in the extub env):
    conda activate extub
    python drop_inr_max_eicu_nonsmote.py --datadir ~/Documents/xgb_extubation_failure/data
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET, MV_COL, DROP_COL = "extubation_failure", "mv_duration_hours", "inr_max"
LEAK_COLS = [
    "reintubated_48h", "hours_to_reintubation", "death_within_48h_of_extubation",
    "n_invasive_episodes", "hospital_mortality", "icu_mortality", "hospital_los_days",
    "icu_los_days", "source", "mortality_28d", "mortality_90d", "first_extubation_time",
    "subject_id", "hadm_id", "stay_id", "icu_intime", "icu_outtime", "admittime",
    "dischtime", "rrt_first_time",
]
B_BOOT, SEED = 1000, 42


def encode_and_filter(df, feats):
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    for c in feats:
        if c not in d.columns:
            d[c] = np.nan
    return d[feats]


def auc_ci(y, p, rng, B=B_BOOT):
    n = len(y); idx = np.arange(n); out = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, n, replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(y[s]):
            out[b] = np.nan; continue
        out[b] = roc_auc_score(y[s], p[s])
    return np.nanpercentile(out, [2.5, 97.5])


def paired_delta_ci(y, p_base, p_drop, rng, B=B_BOOT):
    """Bootstrap of (ablated - baseline) AUROC on the SAME resampled patients."""
    n = len(y); idx = np.arange(n); out = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, n, replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(y[s]):
            out[b] = np.nan; continue
        out[b] = roc_auc_score(y[s], p_drop[s]) - roc_auc_score(y[s], p_base[s])
    return np.nanmedian(out), np.nanpercentile(out, [2.5, 97.5])


def score(model, imp, X):
    return model.predict_proba(imp.transform(X.values))[:, 1]


def main(a):
    bundle = pickle.load(open(os.path.join(a.datadir, "xgb_extubation_failure_v2_nonsmote.pkl"), "rb"))
    feats, model_dep, imp_dep = list(bundle["features"]), bundle["model"], bundle["imputer"]
    assert DROP_COL in feats, f"{DROP_COL} is not among the 28 model features"
    j = feats.index(DROP_COL)
    train_median = float(imp_dep.statistics_[j])

    eicu = pd.read_csv(os.path.join(a.datadir, "eICUdata-1775370213861.csv")).dropna(subset=[TARGET]).reset_index(drop=True)
    yE = eicu[TARGET].astype(int).values
    XE = encode_and_filter(eicu.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu.columns]), feats)

    # deployment preprocessing: v2 mv rule, then bundled median imputer
    XE_dep = XE.copy()
    XE_dep.loc[XE_dep[MV_COL] <= 0, MV_COL] = np.nan
    mask = (XE[MV_COL] > 0).values  # clean subset = positive recorded ventilation duration

    # ---- baseline (reproduces 0.598 / 0.815) ----
    pE_base = score(model_dep, imp_dep, XE_dep)
    full_base = roc_auc_score(yE, pE_base)
    clean_base = roc_auc_score(yE[mask], pE_base[mask])

    # ---- diagnostic: is inr_max constant in eICU? ----
    raw = XE[DROP_COL]
    n_obs = int(raw.notna().sum())
    n_uniq_obs = int(raw.dropna().nunique())
    post = pd.Series(imp_dep.transform(XE_dep.values)[:, j])  # inr_max after the deployment pipeline
    n_uniq_post = int(post.round(9).nunique())

    # ---- ablation: inr_max -> missing -> training median (constant for all rows) ----
    XE_drop = XE_dep.copy()
    XE_drop[DROP_COL] = np.nan
    pE_drop = score(model_dep, imp_dep, XE_drop)
    full_drop = roc_auc_score(yE, pE_drop)
    clean_drop = roc_auc_score(yE[mask], pE_drop[mask])

    rng = np.random.default_rng(SEED)
    full_base_ci = auc_ci(yE, pE_base, rng)
    full_drop_ci = auc_ci(yE, pE_drop, rng)
    clean_base_ci = auc_ci(yE[mask], pE_base[mask], rng)
    clean_drop_ci = auc_ci(yE[mask], pE_drop[mask], rng)
    d_full_med, d_full_ci = paired_delta_ci(yE, pE_base, pE_drop, rng)
    d_clean_med, d_clean_ci = paired_delta_ci(yE[mask], pE_base[mask], pE_drop[mask], rng)

    print("=" * 70)
    print("BASELINE (deployed 28-feature model, non-SMOTE)")
    print(f"  eICU full  AUROC = {full_base:.3f}  (target 0.598)   n={len(yE)}, events={int(yE.sum())}")
    print(f"  eICU clean AUROC = {clean_base:.3f}  (target 0.815)   n={int(mask.sum())}, events={int(yE[mask].sum())}")
    print("-" * 70)
    print(f"DIAGNOSTIC for inr_max  (training-median fill = {train_median:.4f})")
    print(f"  observed (non-missing) in eICU: {n_obs}/{len(yE)}  | distinct observed values: {n_uniq_obs}")
    print(f"  distinct inr_max values after the deployment pipeline: {n_uniq_post}"
          + ("   -> CONSTANT across eICU" if n_uniq_post == 1 else "   -> varies"))
    print("-" * 70)
    print("DROP inr_max  (set to missing -> imputed to training median -> constant)")
    print(f"  eICU full  AUROC = {full_drop:.3f}   Δ = {full_drop - full_base:+.6f}")
    print(f"  eICU clean AUROC = {clean_drop:.3f}   Δ = {clean_drop - clean_base:+.6f}")
    print("-" * 70)
    print("Bootstrap 95% CI (1,000 resamples):")
    print(f"  full  baseline {full_base:.3f} ({full_base_ci[0]:.3f}, {full_base_ci[1]:.3f}) | "
          f"drop {full_drop:.3f} ({full_drop_ci[0]:.3f}, {full_drop_ci[1]:.3f})")
    print(f"  clean baseline {clean_base:.3f} ({clean_base_ci[0]:.3f}, {clean_base_ci[1]:.3f}) | "
          f"drop {clean_drop:.3f} ({clean_drop_ci[0]:.3f}, {clean_drop_ci[1]:.3f})")
    print(f"  paired Δ full : {d_full_med:+.4f} ({d_full_ci[0]:+.4f}, {d_full_ci[1]:+.4f})")
    print(f"  paired Δ clean: {d_clean_med:+.4f} ({d_clean_ci[0]:+.4f}, {d_clean_ci[1]:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=os.path.expanduser("~/Documents/xgb_extubation_failure/data"))
    main(ap.parse_args())
