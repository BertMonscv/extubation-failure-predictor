#!/usr/bin/env python3
"""
Strict pre-extubation feature re-windowing — internal sensitivity analysis (MIMIC-IV).

Compares the published day-1 feature window against a re-windowed
'strict pre-extubation' set, using an IDENTICAL model, fold assignment and
metric set, on the SAME cohort. Reports the AUROC delta overall and stratified
by time-to-extubation, plus calibration and per-feature availability.

This version is wired for THIS project:
  - the 28 model features are read from final_features.csv (column `feature`)
  - day1 matrix  = your existing MIMIC export (e.g. MIMIC-IVdata-*.csv); it has
    many columns, so we select the 28 by name. Rows with a missing label are
    dropped (those are the non-cohort stays).
  - pre  matrix  = output of the re-windowing SQL: stay_id + the re-windowed
    time-series features only. Whatever features the SQL did NOT recompute
    (mv_duration_hours, comorbid/treatment flags) are taken from day1 unchanged.
  - label = extubation_failure, id = stay_id, stratifier = mv_duration_hours.

Model: XGBoost if installed (the paper's model), else sklearn
HistGradientBoosting with matched hyperparameters (also native-NaN). The day1
vs pre comparison is apples-to-apples either way.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RANDOM_STATE = 42
N_SPLITS = 5
N_BOOT = 1000
HYPERPARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05)

def make_model(native_missing: bool):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=HYPERPARAMS["n_estimators"], max_depth=HYPERPARAMS["max_depth"],
            learning_rate=HYPERPARAMS["learning_rate"], objective="binary:logistic",
            tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=HYPERPARAMS["n_estimators"], max_depth=HYPERPARAMS["max_depth"],
            learning_rate=HYPERPARAMS["learning_rate"], random_state=RANDOM_STATE,
        )

def oof_predictions(X: pd.DataFrame, y: np.ndarray, folds, impute: bool) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    Xv = X.values.astype(float)
    for tr, va in folds:
        model = make_model(native_missing=not impute)
        if impute:
            imp = SimpleImputer(strategy="median").fit(Xv[tr])      # train-fold medians only
            Xtr, Xva = imp.transform(Xv[tr]), imp.transform(Xv[va])
        else:
            Xtr, Xva = Xv[tr], Xv[va]
        model.fit(Xtr, y[tr])
        oof[va] = model.predict_proba(Xva)[:, 1]
    assert not np.isnan(oof).any()
    return oof

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def calibration_slope_intercept(y, p):
    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def expected_calibration_error(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)

def auroc_with_ci(y, p, n_boot=N_BOOT, seed=RANDOM_STATE):
    auroc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = [roc_auc_score(y[bi], p[bi]) for bi in
             (np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
              for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auroc, float(lo), float(hi)

def evaluate(y, p, n_boot=N_BOOT):
    auroc, lo, hi = auroc_with_ci(y, p, n_boot=n_boot)
    slope, intercept = calibration_slope_intercept(y, p)
    return {"n": int(len(y)), "events": int(y.sum()),
            "AUROC": round(auroc, 3), "AUROC_lo": round(lo, 3), "AUROC_hi": round(hi, 3),
            "AUPRC": round(average_precision_score(y, p), 3),
            "cal_slope": round(slope, 2), "cal_intercept": round(intercept, 2),
            "Brier": round(brier_score_loss(y, p), 3),
            "ECE": round(expected_calibration_error(y, p), 3)}

TIMING_BINS = [(-np.inf, 6, "<6h"), (6, 12, "6-12h"), (12, 24, "12-24h"), (24, np.inf, ">24h")]

def timing_label(hours):
    if pd.isna(hours):
        return "NA"
    for lo, hi, lab in TIMING_BINS:
        if lo <= hours < hi:
            return lab
    return ">24h"

def availability_table(X: pd.DataFrame, strata: pd.Series) -> pd.DataFrame:
    out = {"overall": X.notna().mean()}
    for lab in [b[2] for b in TIMING_BINS]:
        m = (strata == lab).values
        out[lab] = X[m].notna().mean() if m.any() else np.nan
    return pd.DataFrame(out).round(3)

def run(day1_csv, pre_csv, features_csv, label_col, id_col, cap_hours=24.0):
    feats = pd.read_csv(features_csv)["feature"].tolist()           # the 28 model features
    d1 = pd.read_csv(day1_csv)
    pr = pd.read_csv(pre_csv)
    d1 = d1[d1[label_col].notna()].copy()                           # drop non-cohort rows
    d1 = d1.drop_duplicates(subset=id_col).set_index(id_col)
    pr = pr.drop_duplicates(subset=id_col).set_index(id_col)

    missing = [f for f in feats if f not in d1.columns]
    if missing:
        raise SystemExit(f"day1 is missing model features: {missing}")
    rewindowed = [f for f in feats if f in pr.columns]              # recomputed by SQL
    unchanged  = [f for f in feats if f not in pr.columns]          # taken from day1
    print(f"re-windowed by SQL ({len(rewindowed)}): {rewindowed}")
    print(f"unchanged from day1 ({len(unchanged)}): {unchanged}")

    y = d1[label_col].astype(int).values
    strata = pd.Series([timing_label(h) for h in d1["mv_duration_hours"].values], index=d1.index)

    X_day1 = d1[feats]
    X_pre  = d1[unchanged].join(pr[rewindowed], how="left").reindex(index=d1.index)[feats]

    # --- mv_duration treatments on top of the strict pre-extubation matrix ---
    # mv_duration_hours sits in `unchanged` (taken from day1 = total ventilation
    # time = time-to-extubation), so it leaks outcome timing. Two extra arms,
    # IDENTICAL in every other respect (same cohort, folds, model, metrics):
    #   pre_drop_mv : mv_duration_hours removed entirely  -> clean pre-decision
    #   pre_cap_mv  : mv_duration_hours -> min(value, cap_hours), the ventilation
    #                 time known at a fixed decision horizon (only changes
    #                 patients ventilated beyond the horizon).
    MVC = "mv_duration_hours"
    X_pre_drop = X_pre.drop(columns=[MVC])
    X_pre_cap = X_pre.copy()
    X_pre_cap[MVC] = np.minimum(X_pre_cap[MVC].astype(float), cap_hours)
    print(f"mv_duration arms: drop ({X_pre_drop.shape[1]} feats) | "
          f"cap at {cap_hours:g} h")

    folds = list(StratifiedKFold(N_SPLITS, shuffle=True,
                                 random_state=RANDOM_STATE).split(X_day1, y))   # SAME folds

    rows = []
    for name, X in [("day1", X_day1), ("pre_extubation", X_pre),
                    ("pre_drop_mv", X_pre_drop), ("pre_cap_mv", X_pre_cap)]:
        for impute, tag in [(True, "median_impute"), (False, "native_missing")]:
            p = oof_predictions(X, y, folds, impute=impute)
            rows.append({"window": name, "impute": tag, **evaluate(y, p)})
    overall = pd.DataFrame(rows)

    p_d1   = oof_predictions(X_day1,     y, folds, impute=False)
    p_pr   = oof_predictions(X_pre,      y, folds, impute=False)
    p_drop = oof_predictions(X_pre_drop, y, folds, impute=False)
    p_cap  = oof_predictions(X_pre_cap,  y, folds, impute=False)
    strat_rows = []
    for lab in [b[2] for b in TIMING_BINS]:
        m = (strata == lab).values
        base = {"stratum": lab, "n": int(m.sum()), "events": int(y[m].sum())}
        if m.sum() < 1 or y[m].sum() == 0 or y[m].sum() == m.sum():
            base.update(AUROC_day1=np.nan, AUROC_pre=np.nan,
                        AUROC_drop_mv=np.nan, AUROC_cap_mv=np.nan, delta_pre=np.nan)
        else:
            a1 = roc_auc_score(y[m], p_d1[m]);   a2 = roc_auc_score(y[m], p_pr[m])
            a3 = roc_auc_score(y[m], p_drop[m]); a4 = roc_auc_score(y[m], p_cap[m])
            base.update(AUROC_day1=round(a1, 3), AUROC_pre=round(a2, 3),
                        AUROC_drop_mv=round(a3, 3), AUROC_cap_mv=round(a4, 3),
                        delta_pre=round(a2 - a1, 3))
        strat_rows.append(base)
    stratified = pd.DataFrame(strat_rows)
    avail = availability_table(X_pre[rewindowed], strata)
    return overall, stratified, avail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day1", required=True, help="e.g. MIMIC-IVdata-1775367119727.csv")
    ap.add_argument("--pre", required=True, help="output of the re-windowing SQL")
    ap.add_argument("--features", default="data/final_features.csv")
    ap.add_argument("--label", default="extubation_failure")
    ap.add_argument("--id", default="stay_id")
    ap.add_argument("--cap-hours", type=float, default=24.0,
                    help="version-2 cap horizon for mv_duration_hours (default 24)")
    a = ap.parse_args()
    overall, stratified, avail = run(a.day1, a.pre, a.features, a.label, a.id,
                                     cap_hours=a.cap_hours)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== Overall (day1 vs pre-extubation) ===\n", overall.to_string(index=False))
    print("\n=== AUROC stratified by time-to-extubation (native-missing arm) ===\n",
          stratified.to_string(index=False))
    print("\n=== Re-windowed feature availability (proportion non-missing) ===\n", avail.to_string())
