# results/

Derived, non-identifiable outputs (regenerated; see ../REPRODUCE.md):
- `performance_metrics.csv` — discrimination/calibration for all cohorts.
- `shap_mimic_meanabs.csv`, `shap_eicu_meanabs.csv` — aggregate mean|SHAP| per
  feature (the per-patient SHAP arrays are intentionally not shared).
- `conda_list_extub.txt` — exact environment lock (optional).

Patient-level arrays (`X_*.npy`, per-patient `shap_*.npy`) are blocked by
`.gitignore`: they are derived directly from credentialed data.
