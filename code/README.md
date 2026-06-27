# code/

Analysis pipeline. Run in the `extub` environment (see ../environment.yml).

- `01_pipeline_mimic.py` — train 7 models on MIMIC with 5-fold CV at the native
  event prevalence (no SMOTE), optimise thresholds on pooled OOF, refit the final
  XGBoost on full MIMIC, validate on eICU, save the bundle. Stage-2 feature
  selection (LASSO ∩ Boruta) uses balanced class weights for selection only.
- `02_sensitivity_analysis.py` — day-1 vs strict pre-extubation re-windowing arms.
- `03_predict.py`, `04_eicu_validation.py` — scoring / external validation.
- `05_compute_shap.py` — TreeSHAP on the deployed model.
- `prep_figure_inputs.py` — build predictions + SHAP arrays for the figures.
- `06_make_figures.py` — figures 1–8 + figure S1 + `results/performance_metrics.csv`.
- `analysis_1c.py` — cross-database consistency C1/C2/R0/R1/R2.
- `consistency_extras.py`, `r2_ceiling_diagnostic.py` — reproducibility-ceiling
  decomposition (the 0.27 vs 0.46 analysis).
- `sensitivity/` — recalibration, decision-curve / logistic comparator, subgroup,
  mv-duration handling, inr_max ablation, deployed-model regeneration.

See ../REPRODUCE.md for the run order.
