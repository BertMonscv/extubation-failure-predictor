# data/

This directory does **not** contain raw patient data. MIMIC-IV and eICU-CRD are
credentialed PhysioNet datasets and may not be redistributed.

- `final_features.csv` — the 28 selected predictors (field names; see Additional
  file 1 of the paper for definitions).
- `sql/01_extract_pre_extubation_features.sql` — PostgreSQL extraction of the
  cohort and features from MIMIC-IV / eICU-CRD.
- `sql/reconstruct_surgery_codes.sql` — cardiac-surgery code reconstruction.
- `predictions/` — model outputs regenerated locally (probabilities + outcome
  labels only, no identifiers). See ../REPRODUCE.md.

## To obtain the source data
1. Complete CITI training and request access on PhysioNet:
   - MIMIC-IV  https://physionet.org/content/mimiciv/
   - eICU-CRD  https://physionet.org/content/eicu-crd/
2. Build the cohort with `sql/`, export to CSV, and point the scripts'
   `--datadir` at that folder. Never commit the resulting CSVs (`.gitignore`
   blocks them).
