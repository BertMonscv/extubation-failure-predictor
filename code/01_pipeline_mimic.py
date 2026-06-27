"""
MIMIC-only training pipeline with eICU external validation.

Stages:
  1. Load MIMIC & eICU subset, restrict to 113 common columns
  2. Feature selection on MIMIC: LASSO ∩ Boruta
  3. Train 7 models on MIMIC (5-fold CV, native prevalence, no resampling) → internal metrics
  4. Threshold optimization on pooled OOF
  5. Refit best model (XGBoost) on all of MIMIC
  6. External validation on eICU using the locked threshold
  7. Save model bundle + report

=============================================================================
NON-SMOTE VERSION — this is the published-pipeline de-SMOTE of the original.
Changes vs the original (and ONLY these), to match the manuscript's
"native event prevalence, no resampling" Methods:
  - removed `from imblearn.over_sampling import SMOTE` (import)
  - Stage 3 train_cv: removed in-fold SMOTE; models now fit on Xtr_i / Xtr_s
    (and y[tr]) directly
  - Stage 5: removed SMOTE; final XGBoost fits on X_imp / mimic_y directly
  - metadata + report + docstring "resampling" wording -> "none"
Nothing else (hyperparameters, seed, folds, feature handling) is touched, so
the diff is purely the resampling change.

⚠️ TWO THINGS TO KNOW BEFORE YOU RUN / PUBLISH THIS:

(1) Stage 2 feature selection STILL uses class_weight="balanced" (LASSO C-sweep,
    final LASSO, and the Boruta RF). That is itself a form of class weighting.
    So a blanket "no resampling AND no class weighting anywhere" claim would not
    be literally true. The FINAL MODEL (make_models["XGBoost"]) and the Table-7
    comparators (e.g. LR = LogisticRegression(max_iter=2000), unweighted) use no
    weighting — so scope the Methods sentence to FINAL MODEL TRAINING, e.g.
    "the final model was trained at the native event prevalence without
    resampling or class weighting (feature selection used balanced class weights)."
    Recommended: keep Stage 2 here (auditable selection) + put `boruta` in
    environment.yml. If you would rather FREEZE the 28 features and not re-run
    selection at all (drops the boruta dependency, reproduces the paper's exact
    feature set), swap Stage 2 for a `final_features.csv` load instead.

(2) SEED / FOLD-DRAW: a fresh non-SMOTE run reproduces AUROC (0.872) and slope
    (0.76) tightly, but calibration-in-the-large (intercept) is fold-draw
    sensitive — a separate non-SMOTE run gave intercept ≈ −0.22 vs the deployed
    bundle's −0.11. Treat the ARCHIVED artifacts (bundle, mimic_oof_predictions.csv,
    SHAP matrices) as canonical for the paper; this script reproduces them up to
    seed. Run it in the `extub` env (py 3.11.15 / xgboost 1.7.6 / sklearn 1.9.0;
    boruta + lightgbm + catboost present).

Paths in CONFIG below are the original sandbox paths — set them for your machine
/ repo before running.
=============================================================================
"""
import pickle, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, brier_score_loss, confusion_matrix,
                              roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from boruta import BorutaPy

warnings.filterwarnings("ignore")

# ---- Config ----
RNG = 42
MIMIC_CSV  = "/mnt/user-data/uploads/MIMIC-IVdata-1775367119727.csv"
MERGED_CSV = "/mnt/user-data/uploads/cleaned_merged_data.csv"
OUT_DIR = Path("/mnt/user-data/outputs/mimic_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "extubation_failure"
TARGET_SENS = 0.80

LEAK_COLS = [
    "reintubated_48h", "hours_to_reintubation",
    "death_within_48h_of_extubation", "n_invasive_episodes",
    "hospital_mortality", "icu_mortality",
    "hospital_los_days", "icu_los_days", "source",
    # MIMIC-only outcome cols that aren't in eICU but must still be dropped
    "mortality_28d", "mortality_90d", "first_extubation_time",
    # ID-like
    "subject_id", "hadm_id", "stay_id",
    "icu_intime", "icu_outtime", "admittime", "dischtime",
    "rrt_first_time",
]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------------------------------------------------------------- Stage 1
def load_data():
    log("STAGE 1: Load MIMIC + eICU (restricted to common columns)")
    mimic  = pd.read_csv(MIMIC_CSV)
    merged = pd.read_csv(MERGED_CSV)
    eicu   = merged[merged["source"] == "eICU"].drop(columns=["source"]).reset_index(drop=True)

    common = sorted(set(mimic.columns) & set(eicu.columns))
    log(f"  common columns: {len(common)} (MIMIC has {mimic.shape[1]}, eICU has {eicu.shape[1]})")
    mimic = mimic[common]
    eicu  = eicu[common]

    # Drop rows with missing target
    mimic = mimic.dropna(subset=[TARGET]).reset_index(drop=True)
    eicu  = eicu.dropna(subset=[TARGET]).reset_index(drop=True)
    log(f"  MIMIC after target-drop: {mimic.shape}, pos rate={mimic[TARGET].mean():.4f}")
    log(f"  eICU  after target-drop: {eicu.shape}, pos rate={eicu[TARGET].mean():.4f}")
    return mimic, eicu

def encode_and_filter(df, feat_template=None):
    """One-hot encode and coerce to numeric.
    If feat_template (list of columns) is given, align to it (missing -> NaN)."""
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    d = pd.get_dummies(df, columns=cat, drop_first=True).apply(pd.to_numeric, errors="coerce")
    if feat_template is not None:
        for c in feat_template:
            if c not in d.columns:
                d[c] = np.nan
        d = d[feat_template]
    return d

def prepare_training_matrix(mimic):
    """Drop target + leakage, encode, drop high-missing, return X DataFrame and y."""
    y = mimic[TARGET].astype(int).values
    X = mimic.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in mimic.columns])
    X = encode_and_filter(X)
    hi = X.columns[X.isna().mean() > 0.5].tolist()
    X = X.drop(columns=hi)
    log(f"  dropped {len(hi)} high-missing features; candidate features: {X.shape[1]}")
    return X, y, hi

# ---------------------------------------------------------------- Stage 2
def select_features(X, y):
    log("STAGE 2: Feature selection (LASSO ∩ Boruta on MIMIC)")
    feat = X.columns.tolist()
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_std = StandardScaler().fit_transform(X_imp)

    Cs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    cv_auc = {}
    for C in Cs:
        aucs = []
        for tr, va in skf.split(X_std, y):
            # NOTE: class_weight="balanced" here is SELECTION-stage weighting only
            # (see header note 1); the final model and comparators are unweighted.
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                    class_weight="balanced", max_iter=1000,
                                    random_state=RNG).fit(X_std[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X_std[va])[:, 1]))
        cv_auc[C] = float(np.mean(aucs))
    bestC = max(cv_auc, key=cv_auc.get)
    log(f"  LASSO best C={bestC} (CV AUC={cv_auc[bestC]:.4f})")

    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=bestC,
                                class_weight="balanced", max_iter=2000,
                                random_state=RNG).fit(X_std, y)
    coef = lasso.coef_.ravel()
    lasso_sel = {f for f, c in zip(feat, coef) if abs(c) > 1e-6}
    log(f"  LASSO selected: {len(lasso_sel)}")

    log("  Running Boruta (~1-2 min)...")
    rf = RandomForestClassifier(n_estimators=80, max_depth=6, n_jobs=-1,
                                 class_weight="balanced", random_state=RNG)
    br = BorutaPy(estimator=rf, n_estimators="auto", max_iter=25,
                  random_state=RNG, verbose=0)
    br.fit(X_std.astype(np.float32), y)
    boruta_sel = {f for f, k in zip(feat, br.support_) if k}
    log(f"  Boruta confirmed: {len(boruta_sel)}")

    cmap = dict(zip(feat, coef))
    final = sorted(lasso_sel & boruta_sel, key=lambda f: -abs(cmap[f]))
    log(f"  FINAL (intersection): {len(final)}")
    pd.DataFrame({
        "rank": range(1, len(final) + 1),
        "feature": final,
        "lasso_coef": [cmap[f] for f in final],
    }).to_csv(OUT_DIR / "final_features.csv", index=False)
    return final

# ---------------------------------------------------------------- Stage 3
def make_models():
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=RNG),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8,
                                       n_jobs=-1, random_state=RNG),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", use_label_encoder=False,
                                  n_jobs=-1, random_state=RNG, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    n_jobs=-1, random_state=RNG, verbosity=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, random_state=RNG),
        "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                        random_seed=RNG, verbose=0),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                              early_stopping=True, random_state=RNG),
    }
SCALE_NEEDED = {"LR", "SVM", "MLP"}

def metrics_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": thr,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": spec,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
    }

def train_cv(X_df, y, feats):
    log("STAGE 3: 5-fold CV (7 models, native prevalence, no resampling) on MIMIC")
    X = X_df[feats].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    names = list(make_models().keys())
    oof = {n: np.zeros(len(y)) for n in names}

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr_i, Xva_i = imp.transform(X[tr]), imp.transform(X[va])
        sc = StandardScaler().fit(Xtr_i)
        Xtr_s, Xva_s = sc.transform(Xtr_i), sc.transform(Xva_i)
        # (de-SMOTE) no resampling: fit on the imputed/scaled training fold directly

        for name, mdl in make_models().items():
            if name in SCALE_NEEDED:
                mdl.fit(Xtr_s, y[tr])
                oof[name][va] = mdl.predict_proba(Xva_s)[:, 1]
            else:
                mdl.fit(Xtr_i, y[tr])
                oof[name][va] = mdl.predict_proba(Xva_i)[:, 1]
        log(f"  fold {fold}/5 done ({time.time()-t0:.0f}s)")

    pd.DataFrame({**{"y_true": y}, **oof}).to_csv(
        OUT_DIR / "oof_predictions.csv", index=False)
    return oof, names

# ---------------------------------------------------------------- Stage 4
def youden_thr(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])

def sens_target_thr(y_true, y_prob, target=TARGET_SENS):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = tpr >= target
    if not ok.any():
        return float(thr[np.argmax(tpr)])
    idx = np.where(ok)[0]
    return float(thr[idx[np.argmin(fpr[idx])]])

def threshold_opt(oof, names, y):
    log("STAGE 4: Threshold optimization (pooled OOF)")
    rows_def = [{"Model": n, **metrics_at(y, oof[n], 0.5)} for n in names]
    rows_you = [{"Model": n, **metrics_at(y, oof[n], youden_thr(y, oof[n]))} for n in names]
    rows_s80 = [{"Model": n, **metrics_at(y, oof[n], sens_target_thr(y, oof[n]))} for n in names]
    df_def = pd.DataFrame(rows_def).round(4); df_def.to_csv(OUT_DIR / "cv_default.csv", index=False)
    df_you = pd.DataFrame(rows_you).round(4); df_you.to_csv(OUT_DIR / "cv_youden.csv", index=False)
    df_s80 = pd.DataFrame(rows_s80).round(4); df_s80.to_csv(OUT_DIR / "cv_sens80.csv", index=False)
    return df_def, df_you, df_s80

# ---------------------------------------------------------------- Stage 5 & 6
def fit_final_and_validate(mimic_X_df, mimic_y, feats, eicu_df,
                            thr_youden, thr_sens80):
    log("STAGE 5: Refit final XGBoost on full MIMIC (native prevalence, no resampling)")
    X_train = mimic_X_df[feats].values
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_imp = imputer.transform(X_train)
    # (de-SMOTE) fit on the full imputed MIMIC matrix at native prevalence

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", use_label_encoder=False,
        n_jobs=-1, random_state=RNG, verbosity=0,
    )
    model.fit(X_imp, mimic_y)

    # ---- Stage 6 ----
    log("STAGE 6: External validation on eICU (locked thresholds)")
    y_ext = eicu_df[TARGET].astype(int).values
    X_ext_df = eicu_df.drop(columns=[TARGET] + [c for c in LEAK_COLS if c in eicu_df.columns])
    X_ext_enc = encode_and_filter(X_ext_df, feat_template=feats)
    X_ext = imputer.transform(X_ext_enc.values)
    prob_ext = model.predict_proba(X_ext)[:, 1]

    ext_rows = [
        {"Strategy": "threshold=0.5",       **metrics_at(y_ext, prob_ext, 0.5)},
        {"Strategy": "Youden (from MIMIC OOF)", **metrics_at(y_ext, prob_ext, thr_youden)},
        {"Strategy": f"Sens>={TARGET_SENS} (from MIMIC OOF)",
         **metrics_at(y_ext, prob_ext, thr_sens80)},
    ]
    ext_df = pd.DataFrame(ext_rows).round(4)
    ext_df.to_csv(OUT_DIR / "external_validation_eicu.csv", index=False)
    log(f"  eICU n={len(y_ext)}, pos rate={y_ext.mean():.4f}")
    log(f"  eICU AUROC={roc_auc_score(y_ext, prob_ext):.4f}")

    pd.DataFrame({"y_true": y_ext, "prob": prob_ext}).to_csv(
        OUT_DIR / "eicu_predictions.csv", index=False)
    return model, imputer, ext_df

# ---------------------------------------------------------------- Save + report
def save_bundle(model, imputer, feats, thr_youden, thr_sens80,
                mimic_n, pos_rate, cv_auroc):
    bundle = {
        "model": model,
        "imputer": imputer,
        "features": feats,
        "thresholds": {
            "youden": float(thr_youden),
            "sens80": float(thr_sens80),
            "default": 0.5,
        },
        "target": TARGET,
        "leak_cols": LEAK_COLS,
        "metadata": {
            "trained_on": "MIMIC-IV only (113 common-with-eICU columns)",
            "n_train": int(mimic_n),
            "positive_rate": float(pos_rate),
            "n_features": len(feats),
            "model": "XGBClassifier(n=300, depth=5, lr=0.05)",
            "resampling": "none (native event prevalence; no SMOTE/class weighting in final training)",
            "cv_auroc_pooled_oof": float(cv_auroc),
            "recommended_threshold": "sens80",
            "notes": (
                "Trained on MIMIC only. Thresholds were picked on 5-fold "
                "pooled OOF predictions from MIMIC *before* external "
                "validation on eICU (no peeking). Use the stored threshold "
                "instead of 0.5."
            ),
        },
    }
    path = OUT_DIR / "xgb_extubation_failure_mimic.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log(f"Saved model bundle: {path} ({path.stat().st_size/1024:.1f} KB)")
    return path

def write_report(n_feats, default_df, you_df, sens_df, ext_df,
                 mimic_n, mimic_pos, eicu_n, eicu_pos):
    lines = [
        "# MIMIC-trained model with eICU external validation",
        "",
        f"- **Training set**: MIMIC-IV, n={mimic_n}, positive rate={mimic_pos:.4f}",
        f"- **External set**: eICU, n={eicu_n}, positive rate={eicu_pos:.4f}",
        f"- **Features**: {n_feats} (LASSO ∩ Boruta on MIMIC, restricted to "
        f"eICU-compatible columns)",
        "- **Best model**: XGBoost",
        "",
        "## Internal validation (MIMIC 5-fold CV, pooled OOF)",
        "",
        "### Threshold = 0.5",
        "", default_df.to_markdown(index=False), "",
        "### Youden-optimal threshold",
        "", you_df.to_markdown(index=False), "",
        f"### Sens >= {TARGET_SENS} threshold",
        "", sens_df.to_markdown(index=False), "",
        "## External validation on eICU (XGBoost, locked thresholds from MIMIC)",
        "",
        ext_df.to_markdown(index=False),
        "",
        "## Notes",
        "- No resampling; models trained at the native event prevalence (final model and comparators unweighted).",
        "- Imputation/scaling fit on training folds only.",
        "- eICU thresholds come from MIMIC pooled OOF — no threshold tuning on eICU.",
        "- MIMIC and eICU have very different positive rates "
        f"({mimic_pos:.3f} vs {eicu_pos:.3f}); expect Precision and PPV to "
        "shift in external validation even if AUROC is preserved.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines))

# ---------------------------------------------------------------- Main
def main():
    t0 = time.time()
    mimic_full, eicu_full = load_data()
    mimic_X_df, mimic_y, _ = prepare_training_matrix(mimic_full)
    feats = select_features(mimic_X_df, mimic_y)

    oof, names = train_cv(mimic_X_df, mimic_y, feats)
    default_df, you_df, sens_df = threshold_opt(oof, names, mimic_y)

    best_name = max(names, key=lambda n: roc_auc_score(mimic_y, oof[n]))
    log(f"Best model by OOF AUROC: {best_name} "
        f"(AUROC={roc_auc_score(mimic_y, oof[best_name]):.4f})")
    if best_name != "XGBoost":
        log(f"  NOTE: best model is {best_name}, but we still deploy XGBoost "
            "for consistency with previous reports. Edit this code if you "
            "want to deploy a different model.")

    thr_youden = youden_thr(mimic_y, oof["XGBoost"])
    thr_sens80 = sens_target_thr(mimic_y, oof["XGBoost"])
    cv_auroc   = roc_auc_score(mimic_y, oof["XGBoost"])
    log(f"XGBoost thresholds: Youden={thr_youden:.4f}  Sens80={thr_sens80:.4f}")

    model, imputer, ext_df = fit_final_and_validate(
        mimic_X_df, mimic_y, feats, eicu_full, thr_youden, thr_sens80)
    save_bundle(model, imputer, feats, thr_youden, thr_sens80,
                len(mimic_y), mimic_y.mean(), cv_auroc)

    y_ext = eicu_full[TARGET].astype(int).values
    write_report(len(feats), default_df, you_df, sens_df, ext_df,
                 len(mimic_y), mimic_y.mean(), len(y_ext), y_ext.mean())

    log(f"DONE in {time.time()-t0:.0f}s. Artifacts in {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        log(f"  - {f.name}")

if __name__ == "__main__":
    main()
