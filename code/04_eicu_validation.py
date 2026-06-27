"""
Full external validation of the v2 MIMIC-trained XGBoost model on eICU.
"""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              recall_score, precision_score,
                              f1_score, brier_score_loss, confusion_matrix,
                              accuracy_score)
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/mnt/user-data/outputs/mimic_only")
from predict import predict as model_predict

MODEL = "/mnt/user-data/outputs/mimic_only/xgb_extubation_failure_v2.pkl"
EICU  = "/mnt/user-data/uploads/eicu_external_validation.csv"
OUT   = Path("/mnt/user-data/outputs/mimic_only")

bundle = pickle.load(open(MODEL, "rb"))
md = bundle["metadata"]
print(f"Model v{md['version']}  |  MIMIC internal CV AUROC={md['cv_auroc_pooled_oof']:.4f}")
print(f"Thresholds: {bundle['thresholds']}\n")

eicu_raw = pd.read_csv(EICU)
print(f"eICU raw: n={len(eicu_raw)}, positive rate={eicu_raw.extubation_failure.mean():.4f}")
print(f"  mv_duration_hours <= 0: {(eicu_raw.mv_duration_hours <= 0).sum()} rows\n")

# ---- Define two cohorts for comparison ----
cohorts = {
    "All eICU (mv<=0 auto-imputed)": eicu_raw.copy(),
    "Clean subset (mv>0)":            eicu_raw[eicu_raw.mv_duration_hours > 0].reset_index(drop=True),
}

def metrics(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Threshold": round(thr, 4),
        "AUROC":    round(roc_auc_score(y_true, y_prob), 4),
        "AUPRC":    round(average_precision_score(y_true, y_prob), 4),
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "Sensitivity": round(recall_score(y_true, pred, zero_division=0), 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) else 0, 4),
        "PPV":      round(precision_score(y_true, pred, zero_division=0), 4),
        "NPV":      round(tn / (tn + fn) if (tn + fn) else 0, 4),
        "F1":       round(f1_score(y_true, pred, zero_division=0), 4),
        "Brier":    round(brier_score_loss(y_true, y_prob), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }

def bootstrap_ci(y, p, fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2: continue
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

THRESH = {
    "Default (0.5)":                 0.5,
    "Youden (MIMIC OOF)":            bundle["thresholds"]["youden"],
    f"Sens>=0.80 (MIMIC OOF)":       bundle["thresholds"]["sens80"],
}

all_rows = []
summary_rows = []
for cohort_name, df in cohorts.items():
    probs, _, _ = model_predict(df, bundle, threshold="sens80", clean_input=True)
    y = df.extubation_failure.astype(int).values
    auc = roc_auc_score(y, probs)
    auprc = average_precision_score(y, probs)
    auc_lo, auc_hi = bootstrap_ci(y, probs, roc_auc_score)
    aup_lo, aup_hi = bootstrap_ci(y, probs, average_precision_score)

    summary_rows.append({
        "Cohort": cohort_name, "n": len(y),
        "Prevalence": round(y.mean(), 4),
        "AUROC": round(auc, 4),
        "AUROC_95CI": f"[{auc_lo:.3f}, {auc_hi:.3f}]",
        "AUPRC": round(auprc, 4),
        "AUPRC_95CI": f"[{aup_lo:.3f}, {aup_hi:.3f}]",
    })
    for tname, thr in THRESH.items():
        r = {"Cohort": cohort_name, "n": len(y), "ThresholdStrategy": tname,
             **metrics(y, probs, thr)}
        all_rows.append(r)

full = pd.DataFrame(all_rows)
summary = pd.DataFrame(summary_rows)

# Internal (MIMIC) reference for comparison table
mimic_cv_sens80 = pd.read_csv(OUT / "cv_sens80.csv")
mimic_xgb = mimic_cv_sens80[mimic_cv_sens80.Model == "XGBoost"].iloc[0]
comparison = pd.DataFrame([
    {"Setting": "MIMIC internal CV (pooled OOF)",
     "n": md["n_train"], "Prevalence": round(md["positive_rate"], 4),
     "Threshold": round(float(mimic_xgb.threshold), 4),
     "AUROC": round(float(mimic_xgb.AUROC), 4),
     "AUPRC": round(float(mimic_xgb.AUPRC), 4),
     "Sensitivity": round(float(mimic_xgb.Sensitivity), 4),
     "Specificity": round(float(mimic_xgb.Specificity), 4),
     "PPV": round(float(mimic_xgb.Precision), 4),
     "F1": round(float(mimic_xgb.F1), 4),
     "Brier": round(float(mimic_xgb.Brier), 4)},
])
# Append external Sens>=0.80 rows from full
for cohort_name in cohorts:
    r = full[(full.Cohort == cohort_name) &
             (full.ThresholdStrategy == "Sens>=0.80 (MIMIC OOF)")].iloc[0]
    comparison = pd.concat([comparison, pd.DataFrame([{
        "Setting": f"eICU external ({cohort_name})",
        "n": r.n, "Prevalence": round(cohorts[cohort_name].extubation_failure.mean(), 4),
        "Threshold": r.Threshold,
        "AUROC": r.AUROC, "AUPRC": r.AUPRC,
        "Sensitivity": r.Sensitivity, "Specificity": r.Specificity,
        "PPV": r.PPV, "F1": r.F1, "Brier": r.Brier,
    }])], ignore_index=True)

print("=" * 105)
print("SUMMARY: AUROC / AUPRC on eICU (bootstrap 95% CI, n=1000)")
print("=" * 105)
print(summary.to_string(index=False))

print("\n" + "=" * 115)
print("FULL METRICS by threshold × cohort")
print("=" * 115)
print(full.to_string(index=False))

print("\n" + "=" * 115)
print("MIMIC internal vs eICU external (XGBoost @ Sens>=0.80 threshold)")
print("=" * 115)
print(comparison.to_string(index=False))

# Save all
summary.to_csv(OUT / "eicu_report_summary.csv", index=False)
full.to_csv(OUT / "eicu_report_full_metrics.csv", index=False)
comparison.to_csv(OUT / "eicu_report_comparison.csv", index=False)
print(f"\nSaved to {OUT}")
