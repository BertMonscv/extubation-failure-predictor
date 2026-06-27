# Extubation-failure prediction and cross-database SHAP-consistency

Code, trained model and reproduction materials for an XGBoost model that predicts
extubation failure (reintubation or death within 48 h of the index extubation),
developed on **MIMIC-IV** and externally validated on **eICU-CRD**. The study's
focus is methodological: how **fragile** cross-database SHAP feature-importance
rankings are, and how a single-model consistency estimate (C1) overstates the
agreement that an independently trained model (C2) actually reproduces.

> **Fill in after you create the repository and release:**
> GitHub: `https://github.com/BertMonscv/extubation-failure-predictor` · Zenodo: 10.5281/zenodo.20965771 · Live app: https://extubation-failure-predictor.streamlit.app

## What this model is (and is not)
- A research model trained at the **native event prevalence with no resampling
  and no class weighting in the final model** (this is a deliberately
  de-SMOTE'd pipeline; see "Reproducibility note" below).
- **Not** a cleared clinical device. The external full-cohort discrimination is
  weak (AUROC ≈ 0.60) and calibration drifts off-site; do not use for care.

## Headline results (non-SMOTE)
| Cohort | AUROC (95% CI) | AUPRC | Calib. slope | Calib. intercept |
| --- | --- | --- | --- | --- |
| MIMIC-IV (internal, pooled OOF) | 0.872 (0.850, 0.892) | 0.326 | 0.76 | −0.11 |
| eICU full (n=1,232) | 0.598 (0.545, 0.646) | 0.234 | 0.25 | −0.94 |
| eICU clean (n=756) | 0.815 (0.732, 0.886) | 0.175 | 0.84 | −0.43 |

Cross-database SHAP rank agreement: single-model **C1 = 0.90**; independent-model
**C2 = 0.48** (permutation p = 0.006); chance floor **R0 = 0.44**; within-database
reproducibilities 0.46 (MIMIC, deployment) and 0.81 (eICU). At 160 external events
C2 does not identify the degree of underlying conservation; the comparison is
hypothesis-generating. The model uses **28 predictors**; see `data/final_features.csv`
and Additional file 1 of the paper.

## Data access — IMPORTANT
MIMIC-IV and eICU-CRD are **credentialed PhysioNet datasets** and are **NOT
redistributed here**. This repo ships only the extraction SQL, the feature list,
the trained model, and derived non-identifiable outputs. To reproduce from raw
data you must obtain access yourself:
- MIMIC-IV: https://physionet.org/content/mimiciv/
- eICU-CRD: https://physionet.org/content/eicu-crd/
See `data/README.md`. Do not commit any raw `*.csv` extracted from PhysioNet.

## Repository layout
```
.
├── code/                 analysis pipeline (see code/README.md)
│   ├── 01_pipeline_mimic.py        train + internal CV + eICU validation (no SMOTE)
│   ├── 02_sensitivity_analysis.py  re-windowing / day-1 vs pre-extubation
│   ├── 03_predict.py 04_eicu_validation.py 05_compute_shap.py
│   ├── 06_make_figures.py prep_figure_inputs.py
│   ├── analysis_1c.py consistency_extras.py r2_ceiling_diagnostic.py
│   └── sensitivity/                step3_* arms, DCA/LR, recalibration, ablations
├── models/xgb_extubation_failure_v2.pkl   deployed non-SMOTE bundle
├── data/final_features.csv  + data/sql/    feature list + extraction SQL (no raw data)
├── results/              predictions, aggregate SHAP, performance_metrics.csv
├── figures/              publication figures (regenerated)
├── app/                  Streamlit demo
├── environment.yml       pinned "extub" environment
├── REPRODUCE.md          step-by-step regeneration in extub
├── LICENSE  CITATION.cff
```

## Quick start
```bash
conda env create -f environment.yml
conda activate extub
python -c "import numpy,scipy,pandas,sklearn,matplotlib,xgboost,shap,lightgbm,catboost; print('OK')"
```
Reproduction (predictions → SHAP → figures → metrics) is in **`REPRODUCE.md`**.

## Reproducibility note (no SMOTE)
The development pipeline was finalised **without SMOTE or class weighting in the
final model**: `01_pipeline_mimic.py` trains all models at the native event
prevalence. Feature selection (Stage 2, LASSO ∩ Boruta) uses balanced class
weights for *selection only*; the final model and all comparators are unweighted.
A fresh retrain reproduces AUROC and calibration slope tightly, but the
calibration-in-the-large (intercept) is sensitive to the cross-validation fold
draw — so the **archived artifacts** (the model bundle, `mimic_oof_predictions.csv`,
the SHAP arrays) are canonical for the published numbers, and a re-run reproduces
them up to seed.

## Citation
See `CITATION.cff`. Please cite the paper and the archived release (10.5281/zenodo.20965771).

## License
See `LICENSE`. Code is released for research reproducibility; the credentialed
source data remain under their respective PhysioNet data use agreements.
