-- ============================================================================
-- PostgreSQL / DBeaver — re-window the 24 time-series features to CLOSE AT
-- EXTUBATION, using the mimic-code DERIVED tables and the first_extubation_time
-- you ALREADY have (no need to re-derive the cohort or tau).
--
-- Output: one row per stay_id with the 24 re-windowed features. Export as
-- pre.csv, then feed to 02_sensitivity_analysis.py (--pre pre.csv). The 4
-- unchanged features (mv_duration_hours, received_rrt, cerebrovascular_disease,
-- congestive_heart_failure) are taken from your day1 file by the Python script,
-- so they are intentionally NOT recomputed here.
--
-- ----------------------------------------------------------------------------
-- PREREQUISITES (one-time, in DBeaver):
--   1. From your MIMIC export, write a 5-column cohort_tau.csv (cohort rows
--      only). In Python:
--        import pandas as pd, os
--        d = pd.read_csv(os.path.expanduser(
--              '~/Documents/xgb_extubation_failure/data/MIMIC-IVdata-1775367119727.csv'))
--        c = d[d['extubation_failure'].notna()][
--              ['stay_id','subject_id','hadm_id','icu_intime','first_extubation_time']]
--        c.to_csv('cohort_tau.csv', index=False)
--   2. In DBeaver: create the table below and import cohort_tau.csv into it
--      (right-click schema > Import Data > CSV). Make sure the two time columns
--      import as timestamp, not text.
--        CREATE TABLE cohort_tau (
--          stay_id               bigint PRIMARY KEY,
--          subject_id            bigint,
--          hadm_id               bigint,
--          icu_intime            timestamp,
--          first_extubation_time timestamp
--        );
--
-- TWO THINGS TO VERIFY / ADJUST (flagged inline as >>> ADJUST):
--   A. Schema name. This assumes the mimic-code derived concepts live in
--      `mimiciv_derived`. Check DBeaver's schema tree; if yours is named
--      differently (e.g. `derived`, `public`), find-replace mimiciv_derived.
--   B. The WINDOW. The whole point is to keep your ORIGINAL feature window but
--      truncate it at extubation. This template assumes the original window was
--      "first 24 h from ICU admission" -> new upper bound = LEAST(intime+24h, tau),
--      lower bound = intime. If your original extraction used a different window
--      (e.g. anchored at first_mv_start, or a different length), edit the single
--      `win` CTE below. (Your original DBeaver query — check SQL Editor > SQL
--      History — pins this exactly.)
-- ============================================================================

WITH win AS (                                   -- >>> ADJUST window here (only here)
    SELECT
        stay_id, subject_id, hadm_id,
        icu_intime                                   AS win_start,
        LEAST(icu_intime + INTERVAL '24 hour',
              first_extubation_time)                 AS win_end
    FROM cohort_tau
    WHERE first_extubation_time IS NOT NULL
),

-- blood gas (subject/hadm-level, has charttime) ------------------------------
bg AS (
    SELECT w.stay_id,
        MIN(b.po2)     AS po2_min,
        MAX(b.po2)     AS po2_max,
        MAX(b.pco2)    AS pco2_max,
        MIN(b.ph)      AS ph_min,
        MAX(b.lactate) AS lactate_max
    FROM win w
    JOIN mimiciv_derived.bg b
      ON b.subject_id = w.subject_id
     AND b.hadm_id    = w.hadm_id
     AND b.charttime >= w.win_start
     AND b.charttime <  w.win_end
    GROUP BY w.stay_id
),

-- chemistry ------------------------------------------------------------------
chem AS (
    SELECT w.stay_id,
        MIN(x.aniongap)    AS aniongap_min,
        MAX(x.aniongap)    AS aniongap_max,
        MIN(x.bicarbonate) AS bicarbonate_min,
        MAX(x.bun)         AS bun_max,
        MIN(x.creatinine)  AS creatinine_min,
        MAX(x.sodium)      AS sodium_max,
        MIN(x.potassium)   AS potassium_min
    FROM win w
    JOIN mimiciv_derived.chemistry x
      ON x.subject_id = w.subject_id
     AND x.hadm_id    = w.hadm_id
     AND x.charttime >= w.win_start
     AND x.charttime <  w.win_end
    GROUP BY w.stay_id
),

-- complete blood count (note: column is `platelet`) --------------------------
cbc AS (
    SELECT w.stay_id,
        MIN(x.hemoglobin) AS hemoglobin_min,
        MAX(x.platelet)   AS platelets_max
    FROM win w
    JOIN mimiciv_derived.complete_blood_count x
      ON x.subject_id = w.subject_id
     AND x.hadm_id    = w.hadm_id
     AND x.charttime >= w.win_start
     AND x.charttime <  w.win_end
    GROUP BY w.stay_id
),

-- coagulation ----------------------------------------------------------------
coag AS (
    SELECT w.stay_id,
        MAX(x.pt)         AS pt_max,
        MIN(x.ptt)        AS ptt_min,
        MAX(x.ptt)        AS ptt_max,
        MAX(x.inr)        AS inr_max,
        MIN(x.fibrinogen) AS fibrinogen_min
    FROM win w
    JOIN mimiciv_derived.coagulation x
      ON x.subject_id = w.subject_id
     AND x.hadm_id    = w.hadm_id
     AND x.charttime >= w.win_start
     AND x.charttime <  w.win_end
    GROUP BY w.stay_id
),

-- vital signs (ICU-level, stay_id) -------------------------------------------
vital AS (
    SELECT w.stay_id,
        AVG(v.sbp)       AS sbp_mean,
        MAX(v.sbp)       AS sbp_max,
        AVG(v.spo2)      AS spo2_mean,
        AVG(v.resp_rate) AS resp_rate_mean
    FROM win w
    JOIN mimiciv_derived.vitalsign v
      ON v.stay_id   = w.stay_id
     AND v.charttime >= w.win_start
     AND v.charttime <  w.win_end
    GROUP BY w.stay_id
),

-- last FiO2 before win_end (ICU-level, stay_id) ------------------------------
fio2 AS (
    SELECT DISTINCT ON (w.stay_id) w.stay_id, s.fio2 AS last_fio2
    FROM win w
    JOIN mimiciv_derived.ventilator_setting s
      ON s.stay_id   = w.stay_id
     AND s.charttime >= w.win_start
     AND s.charttime <  w.win_end
     AND s.fio2 IS NOT NULL
    ORDER BY w.stay_id, s.charttime DESC
)

-- assemble the 24 re-windowed features ---------------------------------------
SELECT w.stay_id,
       bg.po2_min, bg.po2_max, bg.pco2_max, bg.ph_min, bg.lactate_max,
       chem.aniongap_min, chem.aniongap_max, chem.bicarbonate_min, chem.bun_max,
       chem.creatinine_min, chem.sodium_max, chem.potassium_min,
       cbc.hemoglobin_min, cbc.platelets_max,
       coag.pt_max, coag.ptt_min, coag.ptt_max, coag.inr_max, coag.fibrinogen_min,
       vital.sbp_mean, vital.sbp_max, vital.spo2_mean, vital.resp_rate_mean,
       fio2.last_fio2
FROM win w
LEFT JOIN bg    ON bg.stay_id    = w.stay_id
LEFT JOIN chem  ON chem.stay_id  = w.stay_id
LEFT JOIN cbc   ON cbc.stay_id   = w.stay_id
LEFT JOIN coag  ON coag.stay_id  = w.stay_id
LEFT JOIN vital ON vital.stay_id = w.stay_id
LEFT JOIN fio2  ON fio2.stay_id  = w.stay_id;
-- LEFT JOINs keep stays with zero pre-extubation measurements (NULLs) — do NOT
-- drop them; the Python script imputes / uses native-missing. Export as pre.csv.
