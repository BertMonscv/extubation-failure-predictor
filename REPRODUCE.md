# REPRODUCE.md — regenerate results and figures in `extub`

All commands run from the **repo root**, in the `extub` environment, against your
local PhysioNet extracts. Paths assume the data lives in
`~/Documents/xgb_extubation_failure/data/` (MIMIC/eICU CSVs + the non-SMOTE
bundle); pass `--datadir` to point elsewhere.

> zsh note: copy commands **one line at a time**, with no inline `#` comments
> (interactive zsh treats `#` as a literal arg).

## 0. Environment
```bash
conda activate extub
python -c "import numpy,scipy,pandas,sklearn,matplotlib,xgboost,shap,lightgbm,catboost; print('OK')"
```

## 1. Predictions + SHAP matrices + univariate AUROC (from the non-SMOTE bundle)
`prep_figure_inputs.py` loads `xgb_extubation_failure_v2_nonsmote.pkl` and the raw
MIMIC/eICU CSVs and writes the per-figure inputs into this repo:
```bash
cd ~/Desktop/data挖掘/efp-public
python code/prep_figure_inputs.py --datadir ~/Documents/xgb_extubation_failure/data --outroot .
```
Writes:
- `data/predictions/mimic_oof_predictions.csv`  (y_true, XGBoost)
- `data/predictions/eicu_predictions_full_1232.csv`  (y_true, prob)
- `data/predictions/eicu_mv_positive_mask.npy`
- `data/univariate_auroc_mimic_vs_eicu.csv`
- `results/shap_mimic.npy  results/X_mimic.npy`  (1500 × 28)
- `results/shap_eicu.npy   results/X_eicu.npy`   (1232 × 28)

`X_*.npy` and the per-patient `shap_*.npy` are **local intermediates only** —
`.gitignore` blocks them (they are patient-level arrays; do not commit). See step 4
for the shareable aggregate.

## 2. Cross-database consistency arrays (for Fig 6/7)
`analysis_1c.py` computes C1/C2/R0/R1/R2 and writes the arrays Fig 6/7 read.
**Verify first** that its `MODEL_M_PATH` points at `models/xgb_extubation_failure_v2.pkl`
(the non-SMOTE bundle).
```bash
python code/analysis_1c.py
```

## 3. Figures + performance_metrics.csv (run from repo root; relative paths)
```bash
python code/06_make_figures.py
```
Writes `figures/fig1..fig8(.png/.pdf)`, `figures/figS1...`, and
`results/performance_metrics.csv`.

## 4. DUA-safe aggregates to commit (instead of per-patient arrays)
Derive the eICU clean subset and the aggregate mean|SHAP| tables, then let
`.gitignore` keep the per-patient arrays out:
```bash
python - <<'PY'
import numpy as np, pandas as pd, json
feats = pd.read_csv("data/final_features.csv")["feature"].tolist()
for tag in ["mimic","eicu"]:
    sv = np.load(f"results/shap_{tag}.npy")
    pd.DataFrame({"feature": feats,
                  "mean_abs_shap": np.abs(sv).mean(0)}
                 ).sort_values("mean_abs_shap", ascending=False
                 ).to_csv(f"results/shap_{tag}_meanabs.csv", index=False)
full = pd.read_csv("data/predictions/eicu_predictions_full_1232.csv")
mask = np.load("data/predictions/eicu_mv_positive_mask.npy")
full[mask].to_csv("data/predictions/eicu_predictions_clean_756.csv", index=False)
print("wrote aggregate SHAP + clean subset")
PY
```

## 5. Sanity-check the headline numbers before committing
```bash
python - <<'PY'
import pandas as pd
from sklearn.metrics import roc_auc_score
m = pd.read_csv("data/predictions/mimic_oof_predictions.csv")
print("MIMIC OOF AUROC =", round(roc_auc_score(m.y_true, m.XGBoost), 3), "(expect 0.872)")
e = pd.read_csv("data/predictions/eicu_predictions_full_1232.csv")
print("eICU full AUROC =", round(roc_auc_score(e.y_true, e.prob), 3), "(expect 0.598)")
PY
```
If MIMIC OOF ≠ 0.872, you are in `base`, not `extub` — re-activate and rerun.

## 6. Archive the exact environment (optional but recommended)
```bash
conda list -n extub > results/conda_list_extub.txt
```

## What ends up committed in results/ and figures/
- `results/`: `performance_metrics.csv`, `shap_mimic_meanabs.csv`,
  `shap_eicu_meanabs.csv`, `conda_list_extub.txt`
- `data/predictions/`: `mimic_oof_predictions.csv`,
  `eicu_predictions_full_1232.csv`, `eicu_predictions_clean_756.csv`,
  `eicu_mv_positive_mask.npy`
- `data/univariate_auroc_mimic_vs_eicu.csv`
- `figures/`: the figure PNG/PDFs
- **not committed:** `X_*.npy`, per-patient `shap_*.npy` (blocked by `.gitignore`)
