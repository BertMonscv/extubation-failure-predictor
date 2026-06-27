#!/usr/bin/env python3
# =============================================================================
# r2_ceiling_diagnostic.py   (Route B, Step ②: "查再定")
#
# Question: C2 (~0.47) sits at/above the published R2 ceiling (~0.29). Is that
# because R2 is an UNFAIRLY-LOW ceiling -- it is computed with the SMALL eICU
# model (HP_E) on data DOWNSAMPLED to 160 events, whereas C2's MIMIC side uses
# the BIG model (HP_M) trained on all 342 MIMIC events?
#
# This reuses analysis_1c's exact machinery (same SEED, same non-SMOTE model,
# same crossfit/compose) to compute several reproducibility ceilings that vary
# ONE thing at a time, and shows where C2 falls relative to a FAIR ceiling:
#
#   1. R2 published      : within-MIMIC, HP_E (small), downsample->160 ev   [anchor ~0.29]
#   2. R2 HP_M @160ev     : within-MIMIC, HP_M (big),   downsample->160 ev   [model effect]
#   3. R2 HP_M @native    : within-MIMIC, HP_M (big),   NO downsample        [+ size effect]
#   4. within-eICU        : within-eICU,  HP_E (small), NO downsample (~80ev)[the eICU-side ceiling]
#
# Run FROM the data dir so analysis_1c auto-discovers the CSVs:
#   conda activate extub
#   cd ~/Documents/xgb_extubation_failure/data
#   python ~/Downloads/r2_ceiling_diagnostic.py
# =============================================================================
import os, sys
import numpy as np
from scipy.stats import spearmanr

DATA_DIR  = "/Users/ilizyue/Documents/xgb_extubation_failure/data"
N_SPLITS  = 40          # reduced from 200 for speed; medians are stable enough to compare LEVELS
sys.path.insert(0, DATA_DIR)
import analysis_1c as A  # reuses MODEL_M_PATH (now the non-SMOTE pkl), HP_M, HP_E_PRIMARY, etc.

PREV   = A.PREV_PRIMARY
NREP   = A.N_REPEATS
CVS    = A.CV_SPLITS
SEED   = A.SEED

# ---- generalized reproducibility ceiling (== A.r2_independent_within, but with
#      a choice of hp, optional downsample, and any database) -------------------
def repro_ceiling(Xdf, y, hp, n_events, n_splits, tag):
    rng = np.random.default_rng(SEED + 99)
    y = np.asarray(y)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    cors = []; size0 = None
    for s in range(n_splits):
        rng.shuffle(pos); rng.shuffle(neg)
        posA, posB = pos[: len(pos)//2], pos[len(pos)//2:]
        negA, negB = neg[: len(neg)//2], neg[len(neg)//2:]
        vecs = []; sz = []
        for ph, nh in ((posA, negA), (posB, negB)):
            if n_events is not None:
                sel = A.downsample_to_events(ph, nh, n_events, PREV, rng)
            else:
                sel = np.concatenate([ph, nh]); rng.shuffle(sel)
            cf = A.crossfit_shap_abs_matrix(Xdf.iloc[sel], y[sel], hp, SEED + s + 1, CVS)
            vecs.append(A.compose_average(cf, y[sel], PREV, rng, NREP))
            sz.append((len(sel), int(y[sel].sum())))
        cors.append(spearmanr(vecs[0], vecs[1]).correlation)
        if size0 is None: size0 = sz
        if (s + 1) % 10 == 0:
            print("   [%s] %d/%d splits  running median=%.3f" %
                  (tag, s + 1, n_splits, float(np.median(cors))), flush=True)
    return np.array(cors), size0

def summ(c):
    return "%.3f  (2.5–97.5: %.3f, %.3f)" % (float(np.median(c)),
            float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5)))

# ---- load data + the (non-SMOTE) frozen model, exactly as run_pipeline does --
print(">>> discovering data + loading model (mirrors analysis_1c.run_pipeline)")
X_M, y_M, X_E, y_E, model_path = A.discover(DATA_DIR)
model_M = A.load_model_M(model_path, X_M, y_M)
print(">>> MIMIC n=%d ev=%d | eICU n=%d ev=%d" %
      (len(y_M), int(np.sum(y_M)), len(y_E), int(np.sum(y_E))))

# ---- observed C2 + R0 floor (same model, same SEED) -------------------------
S_ME, S_E, c2m, absME, absE, _ = A.c2_pipeline(
    X_M, y_M, X_E, y_E, A.FEATURES, A.HP_M, A.HP_E_PRIMARY, PREV, SEED, model_M=model_M)
c2_obs = float(c2m["spearman"])
_, r0_null, r0_p = A.permutation_null(S_ME, S_E, np.random.default_rng(SEED + 2), A.N_PERM)
floor = float(np.percentile(r0_null, 99))
print(">>> C2 (observed) = %.3f   |   R0 99th floor = %.3f   (p=%.4f)\n" % (c2_obs, floor, r0_p))

# ---- the four ceilings ------------------------------------------------------
print(">>> computing ceilings (N_SPLITS=%d, CV_SPLITS=%d, N_REPEATS=%d)" % (N_SPLITS, CVS, NREP))
c1, s1 = repro_ceiling(X_M, y_M, A.HP_E_PRIMARY, 160,  N_SPLITS, "R2 pub  (MIMIC, HP_E, 160ev)")
c2, s2 = repro_ceiling(X_M, y_M, A.HP_M,         160,  N_SPLITS, "R2 HP_M (MIMIC, HP_M, 160ev)")
c3, s3 = repro_ceiling(X_M, y_M, A.HP_M,         None, N_SPLITS, "R2 HP_M (MIMIC, HP_M, native)")
c4, s4 = repro_ceiling(X_E, y_E, A.HP_E_PRIMARY, None, N_SPLITS, "within-eICU (eICU, HP_E, native)")

print("\n" + "=" * 78)
print("REPRODUCIBILITY-CEILING DIAGNOSTIC  (Spearman rho between two independent models)")
print("=" * 78)
print("%-34s %-22s %s" % ("ceiling", "per-model train (n/ev)", "median rho (95%)"))
print("-" * 78)
print("%-34s %-22s %s" % ("1. R2 published (HP_E, 160ev)",  "%d / %d" % s1[0], summ(c1)))
print("%-34s %-22s %s" % ("2. R2 HP_M @160ev",              "%d / %d" % s2[0], summ(c2)))
print("%-34s %-22s %s" % ("3. R2 HP_M @native (no downsamp)","%d / %d" % s3[0], summ(c3)))
print("%-34s %-22s %s" % ("4. within-eICU (HP_E, ~80ev)",   "%d / %d" % s4[0], summ(c4)))
print("-" * 78)
print("%-34s %-22s %.3f" % ("C2 OBSERVED (cross-database)",
      "MIMIC 342ev / eICU ~160ev", c2_obs))
print("%-34s %-22s %.3f" % ("R0 chance floor (99th pct)", "-", floor))
print("=" * 78)

med = lambda c: float(np.median(c))
print("\nDecomposition of why the published ceiling is low:")
print("  model effect  (HP_M vs HP_E @160ev) : %.3f -> %.3f  (x%.2f)"
      % (med(c1), med(c2), med(c2) / max(med(c1), 1e-6)))
print("  size  effect  (native vs 160ev, HP_M): %.3f -> %.3f  (x%.2f)"
      % (med(c2), med(c3), med(c3) / max(med(c2), 1e-6)))
print("\nWhere C2=%.3f sits:" % c2_obs)
print("  vs published R2 (0.29-ish) ........ %s" % ("ABOVE" if c2_obs > med(c1) else "below"))
print("  vs FAIR within-MIMIC ceiling (#3) . %s" % ("ABOVE" if c2_obs > med(c3) else "BELOW"))
print("  vs within-eICU ceiling (#4) ....... %s" % ("ABOVE" if c2_obs > med(c4) else "below"))
print("\nReading: if #3 (fair MIMIC ceiling) >> published #1, the 0.29 ceiling was")
print("depressed by HP_E + 160-downsampling. Then whether C2 is BELOW #3 (-> partial /")
print("limited by eICU power) or ABOUT #3 (-> agreement as high as reproducibility allows)")
print("is the number that decides reframe-as-inconclusive vs reframe-as-partial-conservation.")
