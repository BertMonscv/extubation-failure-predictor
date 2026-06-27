# models/

`xgb_extubation_failure_v2.pkl` — the deployed **non-SMOTE** bundle (a pickled
dict with keys `model`, `imputer`, `features`, `thresholds`, `metadata`). Trained
on full MIMIC-IV at the native event prevalence; XGBoost(n_estimators=300,
max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8). This file
is canonical for the published numbers; see the reproducibility note in the root
README. It was serialised under an earlier XGBoost and loads under 1.7.6 with a
benign compatibility warning.
