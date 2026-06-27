-- ============================================================================
-- mimic_cardiac_surgery_codes / reconstruct_surgery_codes.sql
--
-- Recovers the ICD-9-CM / ICD-10-PCS procedure codes that define the
-- cardiac-surgery sub-types (CABG / valve / aortic) of the MIMIC-IV training
-- cohort, by joining the cohort's per-admission surgery flags
-- (has_cabg / has_valve / has_aortic, taken from the model's feature export)
-- back to mimiciv_hosp.procedures_icd.
--
-- A procedure code with frac_<category> = 1.000 in the output is present ONLY in
-- patients flagged for that category, i.e. it is a defining code for that
-- category. The output of STEP 3 is surgery_code_category_map.csv in this folder.
--
-- Engine: PostgreSQL (mimic-code build).  >>> Adjust schema name if your
-- procedures table is not under `mimiciv_hosp`.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- STEP 1 (one-time, in Python): build cohort_flags.csv from the model export.
--
--   import pandas as pd, os
--   d = pd.read_csv(os.path.expanduser(
--         '~/Documents/xgb_extubation_failure/data/MIMIC-IVdata-1775367119727.csv'))
--   (d[['subject_id','hadm_id','has_cabg','has_valve','has_aortic']]
--      .drop_duplicates('hadm_id')
--      .to_csv('cohort_flags.csv', index=False))
-- ----------------------------------------------------------------------------

-- STEP 2: load cohort_flags.csv into a table.
--   (DBeaver: create the table below, then right-click it > Import Data > CSV.)
DROP TABLE IF EXISTS cohort_flags;
CREATE TABLE cohort_flags (
    subject_id bigint,
    hadm_id    bigint PRIMARY KEY,
    has_cabg   int,
    has_valve  int,
    has_aortic int
);
-- <-- import cohort_flags.csv into cohort_flags here -->

-- STEP 3: recover the defining procedure codes per category.
SELECT p.icd_version,
       p.icd_code,
       COUNT(DISTINCT c.hadm_id)            AS n_stays,
       ROUND(AVG(c.has_cabg::numeric),  3)  AS frac_cabg,
       ROUND(AVG(c.has_valve::numeric), 3)  AS frac_valve,
       ROUND(AVG(c.has_aortic::numeric),3)  AS frac_aortic
FROM cohort_flags c
JOIN mimiciv_hosp.procedures_icd p
  ON p.hadm_id = c.hadm_id
WHERE (p.icd_version = 9  AND LEFT(p.icd_code, 2) IN ('35','36','37','38','39'))
   OR (p.icd_version = 10 AND LEFT(p.icd_code, 2) = '02')   -- heart / great-vessel procedure chapters
GROUP BY p.icd_version, p.icd_code
HAVING COUNT(DISTINCT c.hadm_id) >= 5
ORDER BY n_stays DESC;
-- Export the result grid as surgery_code_category_map.csv.
