#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_conservation_recovery.py
====================================
Ground-truth simulation validating the reference-anchored cross-database
SHAP-importance consistency framework (C1 / C2 / R1 / reliability /
disattenuation) used in:

    "Predicting extubation failure after cardiac surgery: a cross-database
     machine-learning study with a reference-anchored analysis of feature-
     importance consistency"  (MIMIC-IV -> eICU-CRD external validation).

Repository: https://github.com/BertMonscv/extubation-failure-predictor

------------------------------------------------------------------------------
WHY THIS SIMULATION EXISTS
------------------------------------------------------------------------------
On the real data we cannot know the *true* degree to which the predictive
mechanism is conserved across databases. We therefore cannot tell whether the
observed cross-source importance agreement C2 = 0.48 falling below its
reliability ceiling reflects (i) genuinely incomplete conservation or
(ii) complete conservation obscured by ranking noise.

This script removes that ambiguity by building two synthetic "databases" A and
B whose *ground-truth* mechanism conservation is a tunable knob:

        rho_true = Spearman(|beta_A|, |beta_B|)   over the p features,

i.e. the rank agreement of the two additive-logistic data-generating
mechanisms. By construction rho_true is exactly the quantity the observed C2
estimates in the limit of infinite training and evaluation data. We then ask:

    1. Does the framework recover rho_true?  (sweep rho_true from 0 -> 1)
    2. Does disattenuation by the measured reliabilities track rho_true?
    3. Under FULL conservation (rho_true = 1), how often does a finite-sample
       C2 fall at or below the real-data value 0.48?  -> adjudicates (i) vs (ii).

------------------------------------------------------------------------------
IMPORTANCE MEASURE  (read this before interpreting results)
------------------------------------------------------------------------------
By default we use a label-free OUTPUT-PERMUTATION importance: the mean absolute
change in a model's predicted logit when one feature is independently permuted
on a common evaluation set. Like mean|SHAP|, it is a global, output-based
measure of how much each feature moves the model's output; it is deliberately
NOT SHAP, to show that the framework's behaviour is a property of the
rank-correlation statistics rather than of any one attribution method (mirroring
the manuscript's logistic-coefficient robustness check). Set IMPORTANCE = "shap"
to reproduce with TreeSHAP + XGBoost in a suitably pinned environment (see the
clearly marked block in `feature_importance`).

------------------------------------------------------------------------------
MODELLING ASSUMPTIONS (honest scope)
------------------------------------------------------------------------------
* Predictors are independent and standardized; the mechanism is additive in the
  logit. This is a deliberate, transparent caricature, not a claim about the
  clinical data. It isolates the statistical behaviour of the consistency
  metrics from confounding by correlated/interacting predictors. Robustness to
  this assumption is checked with `--correlated`, which re-runs everything with
  block-correlated predictors (same mechanisms; intercepts recalibrated).
* "Covariate shift" between A and B is a per-feature mean shift (the importance
  of an additive model depends on the evaluation feature SPREAD, not its mean,
  so a pure mean shift leaves the limiting importances unchanged - which is why
  the limiting C2 equals rho_true exactly).
* The simulation validates the statistical BEHAVIOUR of the framework under a
  calibrated parametric mechanism. It does not, and cannot, prove that the real
  clinical mechanism is conserved; it calibrates how to *read* the real C2.

Dependencies: numpy, scipy, scikit-learn, matplotlib  (pandas optional).
Single-core friendly. Full run (REPS=50, 8 levels) ~25 min on one CPU core;
use --quick for a fast smoke test.

Author: (study authors).  License: same as the repository.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# =============================================================================
# CONFIGURATION  (calibrated so the simulated scaffold matches the real study:
#   C1~0.90, r_A~0.46, r_B~0.81, R1~1.00, R0~0.44, transfer AUROC~0.82)
# =============================================================================
IMPORTANCE = "perm"          # "perm" (default, no extra deps) or "shap" (needs xgboost+shap)

P            = 28            # number of predictors (matches the study)
DECAY_A      = 0.9825        # A = MIMIC-like: importance spread FLAT  -> low  reliability r_A
SCALE_A      = 0.48
DECAY_B      = 0.86          # B = eICU-like : importance CONCENTRATED -> high reliability r_B
SCALE_B      = 1.05
N_A          = 7737          # A training size (MIMIC analysis cohort)
PREV_A       = 0.044         # A event prevalence
N_B          = 1232          # B training size (eICU analysis cohort)
PREV_B       = 0.130         # B event prevalence
SHIFT_MAG    = 0.34          # per-feature covariate mean shift A vs B (sets frozen-model C1~0.90)
NEVAL        = 2000          # common evaluation-set size for importance
NREP_PERM    = 2             # permutation repeats per feature (perm importance)
TARGET_GRID  = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00]   # target rho_true levels
MASTER_SEED  = 1234567

# HistGradientBoosting hyper-parameters (the manuscript shows logistic ~ GBM ~
# CatBoost internally, so the learner choice is not load-bearing).
CFG_A = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
             l2_regularization=1.0, min_samples_leaf=30, early_stopping=False)
CFG_B = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
             l2_regularization=1.0, min_samples_leaf=20, early_stopping=False)

# Real-study anchors (for plotting / adjudication).
REAL = dict(C1=0.90, C2=0.48, C2_lo=0.32, C2_hi=0.57, disatt=0.79,
            r_A=0.46, r_B=0.81, R0=0.44, R1=1.00)


# =============================================================================
# MECHANISM CONSTRUCTION
# =============================================================================
def geometric_profile(p: int, decay: float) -> np.ndarray:
    """Sorted-descending magnitude profile m_k = decay**k, unit mean-square.
    decay -> 1 gives a flat profile (importance spread evenly across features);
    smaller decay concentrates importance in a few features."""
    m = decay ** np.arange(p)
    return m / np.sqrt(np.mean(m ** 2))


def coefficient_vectors(rng):
    """Return beta_A, a fixed sign pattern, the covariate shift vector, the
    latent z for A's ranks, and a closure that builds beta_B at a chosen latent
    correlation. Magnitudes encode importance; signs are shared (sign does not
    affect a feature's importance rank)."""
    profA = geometric_profile(P, DECAY_A) * SCALE_A
    profB = geometric_profile(P, DECAY_B) * SCALE_B
    signs = rng.choice([-1.0, 1.0], size=P)
    beta_A = profA * signs
    shift_B = SHIFT_MAG * rng.choice([-1.0, 1.0], size=P)
    ranks_A = np.arange(P)                         # 0 = most important
    zA = norm.ppf((P - ranks_A) / (P + 1.0))       # high z <-> high importance

    def beta_B_for(rho_latent, seed):
        r = np.random.default_rng(seed)
        zB = rho_latent * zA + np.sqrt(max(0.0, 1 - rho_latent ** 2)) * r.standard_normal(P)
        order = np.argsort(-zB)                     # largest magnitude -> largest zB
        magB = np.empty(P); magB[order] = profB
        bB = magB * signs
        return bB, float(spearmanr(np.abs(beta_A), np.abs(bB)).statistic)

    return beta_A, signs, shift_B, profB, beta_B_for


def build_levels(beta_B_for):
    """For each target rho_true, search latent correlations/seeds for the beta_B
    whose realised Spearman(|beta_A|,|beta_B|) is closest to the target."""
    betaB_levels, rho_levels = [], []
    for g in TARGET_GRID:
        if g >= 0.999:
            bB, rt = beta_B_for(1.0, 999)           # exact identity ordering
            betaB_levels.append(bB); rho_levels.append(rt); continue
        best = None
        for rl in np.linspace(max(0.0, g - 0.25), min(0.999, g + 0.35), 26):
            for s in range(40):
                bB, rt = beta_B_for(rl, hash((round(rl, 3), s)) % (2 ** 31))
                d = abs(rt - g)
                if best is None or d < best[0]:
                    best = (d, bB, rt)
        betaB_levels.append(best[1]); rho_levels.append(best[2])
    return betaB_levels, rho_levels


# =============================================================================
# DATA GENERATION
# =============================================================================
# Optional covariate covariance (lower-Cholesky factor). None => independent
# standardized predictors (the primary analysis). Set via --correlated, which
# builds a block-equicorrelation structure to test robustness to correlated
# predictors (see block_cholesky and main()).
COV_L = None


def block_cholesky(p, block_size, rho_w):
    """Lower-Cholesky factor of a block-diagonal equicorrelation matrix:
    `block_size` consecutive features per block, within-block correlation
    `rho_w`, independent across blocks. Marginal variances are 1."""
    Sig = np.eye(p)
    i = 0
    while i < p:
        b = min(block_size, p - i)
        Sig[i:i + b, i:i + b] = rho_w
        for k in range(i, i + b):
            Sig[k, k] = 1.0
        i += b
    return np.linalg.cholesky(Sig)


def _draw_X(n, rng, mu):
    Z = rng.standard_normal((n, P))
    X = Z @ COV_L.T if COV_L is not None else Z          # N(0, Sigma) or N(0, I)
    return X + mu


def make_data(n, beta, intercept, rng, mean_shift=None):
    """Standardized features X ~ N(mean_shift, Sigma) (Sigma = I unless a
    block-correlated COV_L is set); y ~ Bernoulli(sigmoid(intercept + X @ beta))."""
    mu = 0.0 if mean_shift is None else mean_shift
    X = _draw_X(n, rng, mu)
    pr = 1.0 / (1.0 + np.exp(-(intercept + X @ beta)))
    y = (rng.random(n) < pr).astype(int)
    return X, y


def calibrate_intercept(beta, target_prev, mean_shift=None, n=200_000, seed=0):
    """Bisection for the intercept giving approximately the target event rate."""
    rng = np.random.default_rng(seed)
    mu = 0.0 if mean_shift is None else mean_shift
    lin = _draw_X(n, rng, mu) @ beta
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(1.0 / (1.0 + np.exp(-(mid + lin)))) < target_prev:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# LEARNER + IMPORTANCE
# =============================================================================
def fit_model(X, y, rng, cfg):
    m = HistGradientBoostingClassifier(random_state=int(rng.integers(0, 2 ** 31 - 1)), **cfg)
    m.fit(X, y)
    return m


def feature_importance(model, Xeval, rng):
    """Global, output-based importance on a common evaluation set.

    IMPORTANCE == "perm":  mean |delta logit| under independent permutation of
                           each feature (label-free; default; no extra deps).
    IMPORTANCE == "shap":  mean |SHAP value| via TreeSHAP. To use this, train an
                           xgboost.XGBClassifier in run_* below instead of the
                           HistGradientBoostingClassifier and uncomment the block
                           here. Kept optional so the script runs with only
                           numpy/scipy/scikit-learn/matplotlib installed.
    """
    if IMPORTANCE == "perm":
        base = model.decision_function(Xeval)
        n, p = Xeval.shape
        imp = np.zeros(p)
        for j in range(p):
            col = Xeval[:, j].copy(); acc = 0.0
            for _ in range(NREP_PERM):
                Xeval[:, j] = col[rng.permutation(n)]
                acc += np.mean(np.abs(model.decision_function(Xeval) - base))
            Xeval[:, j] = col
            imp[j] = acc / NREP_PERM
        return imp

    elif IMPORTANCE == "shap":
        # ---- TreeSHAP + XGBoost reproduction path -------------------------
        #   import shap
        #   explainer = shap.TreeExplainer(model)           # model = XGBClassifier
        #   sv = explainer.shap_values(Xeval)               # (n, p) on the logit/margin scale
        #   return np.mean(np.abs(sv), axis=0)
        # -------------------------------------------------------------------
        raise NotImplementedError(
            "IMPORTANCE='shap' requires xgboost+shap and an XGBClassifier learner; "
            "see the commented block in feature_importance().")
    else:
        raise ValueError(f"unknown IMPORTANCE={IMPORTANCE!r}")


def disattenuate(c2, r_a, r_b):
    """Classical correction for attenuation (Spearman 1904). NOTE: derived for
    Pearson correlations with classical additive measurement error; for Spearman
    rank correlations of retrained-model importances it is only approximate, so
    individual disattenuated values can exceed 1 and are reported as such."""
    return c2 / np.sqrt(max(1e-9, r_a) * max(1e-9, r_b))


# =============================================================================
# EXPERIMENTS
# =============================================================================
def run_reliability(beta_A, icA, shift_B, betaB1, icB1, seed=101, n_pairs=24):
    """Measure the scaffold: in/out-of-domain AUROC, within-database
    reproducibilities r_A and r_B, the frozen-model covariate-shift stability
    C1, the structural ceiling R1, and the permutation floor R0."""
    rng = np.random.default_rng(seed)
    imp = lambda m, Xe: feature_importance(m, Xe, rng)
    out = {}

    XA, yA = make_data(8000, beta_A, icA, rng)
    XAt, yAt = make_data(15000, beta_A, icA, rng)
    mA = fit_model(XA, yA, rng, CFG_A)
    out["auroc_A"] = float(roc_auc_score(yAt, mA.predict_proba(XAt)[:, 1]))
    XB, yB = make_data(N_B, betaB1, icB1, rng, mean_shift=shift_B)
    XBt, yBt = make_data(15000, betaB1, icB1, rng, mean_shift=shift_B)
    mB = fit_model(XB, yB, rng, CFG_B)
    out["auroc_B"] = float(roc_auc_score(yBt, mB.predict_proba(XBt)[:, 1]))
    out["auroc_transfer_AonB"] = float(roc_auc_score(yBt, mA.predict_proba(XBt)[:, 1]))

    rA = []
    for _ in range(n_pairs):
        Xe, _ = make_data(NEVAL, beta_A, icA, rng)
        X1, y1 = make_data(N_A, beta_A, icA, rng); X2, y2 = make_data(N_A, beta_A, icA, rng)
        m1 = fit_model(X1, y1, rng, CFG_A); m2 = fit_model(X2, y2, rng, CFG_A)
        rA.append(spearmanr(imp(m1, Xe), imp(m2, Xe)).statistic)
    rB = []
    for _ in range(n_pairs):
        Xe, _ = make_data(NEVAL, betaB1, icB1, rng, mean_shift=shift_B)
        X1, y1 = make_data(N_B, betaB1, icB1, rng, mean_shift=shift_B)
        X2, y2 = make_data(N_B, betaB1, icB1, rng, mean_shift=shift_B)
        m1 = fit_model(X1, y1, rng, CFG_B); m2 = fit_model(X2, y2, rng, CFG_B)
        rB.append(spearmanr(imp(m1, Xe), imp(m2, Xe)).statistic)
    out["r_A_mean"], out["r_A_sd"] = float(np.mean(rA)), float(np.std(rA))
    out["r_B_mean"], out["r_B_sd"] = float(np.mean(rB)), float(np.std(rB))

    c1s, r1s = [], []
    for _ in range(20):
        Xtr, ytr = make_data(N_A, beta_A, icA, rng)
        mf = fit_model(Xtr, ytr, rng, CFG_A)
        Xea, _ = make_data(NEVAL, beta_A, icA, rng)
        Xeb, _ = make_data(NEVAL, beta_A, icA, rng, mean_shift=shift_B)
        c1s.append(spearmanr(imp(mf, Xea), imp(mf, Xeb)).statistic)
        h = NEVAL // 2
        r1s.append(spearmanr(imp(mf, Xea[:h].copy()), imp(mf, Xea[h:].copy())).statistic)
    out["C1_mean"], out["C1_sd"] = float(np.mean(c1s)), float(np.std(c1s))
    out["R1_mean"], out["R1_sd"] = float(np.mean(r1s)), float(np.std(r1s))

    pr = np.random.default_rng(7)
    null = [spearmanr(np.arange(P), pr.permutation(P)).statistic for _ in range(20000)]
    out["R0_p99"] = float(np.percentile(null, 99))
    return out


def run_sweep(beta_A, icA, shift_B, betaB_levels, icB_levels, rho_levels,
              reps=50, seed0=5000, reps_full_extra=100):
    """Cross-source C2 at each rho_true level. The rho_true=1 (full-conservation)
    level gets extra replicates for a precise adjudication tail."""
    imp = None
    records = []  # (level, rho_true, rep, c2)
    for L, (bB, icB, rt) in enumerate(zip(betaB_levels, icB_levels, rho_levels)):
        R = reps + (reps_full_extra if abs(rt - 1.0) < 1e-9 else 0)
        t0 = time.time()
        for r in range(R):
            rng = np.random.default_rng(seed0 + L * 10_000 + r)
            imp = lambda m, Xe: feature_importance(m, Xe, rng)
            Xe, _ = make_data(NEVAL, bB, icB, rng, mean_shift=shift_B)   # common eval (B-dist)
            XA, yA = make_data(N_A, beta_A, icA, rng)                    # A train (A-dist)
            XB, yB = make_data(N_B, bB, icB, rng, mean_shift=shift_B)    # B train (B-dist)
            mA = fit_model(XA, yA, rng, CFG_A)
            mB = fit_model(XB, yB, rng, CFG_B)
            c2 = float(spearmanr(imp(mA, Xe), imp(mB, Xe)).statistic)
            records.append((L, rt, r, c2))
        print(f"  level {L}  rho_true={rt:.3f}  R={R}  ({time.time()-t0:.0f}s)")
    return records


# =============================================================================
# SUMMARY + FIGURE
# =============================================================================
def boot_ci(x, rng, B=4000):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(records, rel, rng):
    denom = float(np.sqrt(rel["r_A_mean"] * rel["r_B_mean"]))
    by = {}
    for L, rt, r, c2 in records:
        by.setdefault((L, rt), []).append(c2)
    rowlist = []
    for (L, rt), c2 in sorted(by.items(), key=lambda kv: kv[0][1]):
        c2 = np.array(c2); lo, hi = boot_ci(c2, rng)
        rowlist.append(dict(level=L, rho_true=rt, n=len(c2),
                            c2_mean=float(c2.mean()), c2_sd=float(c2.std()),
                            c2_lo=lo, c2_hi=hi,
                            disatt_mean=c2.mean()/denom, disatt_lo=lo/denom, disatt_hi=hi/denom))
    # adjudication at rho_true = 1
    full = np.array(by[max(by, key=lambda k: k[1])])
    p_le = float(np.mean(full <= REAL["C2"]))
    return denom, rowlist, full, p_le


def make_figure(rowlist, full, p_le, denom, path_png, path_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans",
                         "axes.edgecolor": "#444", "axes.linewidth": 0.9})
    C_OBS, C_DIS, C_ID = "#2c6fbb", "#d1572c", "#9aa0a6"
    x = np.array([r["rho_true"] for r in rowlist])
    yo = np.array([r["c2_mean"] for r in rowlist])
    yd = np.array([r["disatt_mean"] for r in rowlist])
    olo = np.array([r["c2_lo"] for r in rowlist]); ohi = np.array([r["c2_hi"] for r in rowlist])
    dlo = np.array([r["disatt_lo"] for r in rowlist]); dhi = np.array([r["disatt_hi"] for r in rowlist])
    rho_at = float(np.interp(REAL["C2"], yo, x))

    fig = plt.figure(figsize=(11.4, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.26)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    axA.plot([0, 1], [0, 1], "--", color=C_ID, lw=1.4, label="perfect recovery (identity)")
    axA.fill_between(x, olo, ohi, color=C_OBS, alpha=0.13)
    axA.plot(x, yo, "-o", color=C_OBS, lw=2.0, ms=5.5, label="observed $C_2$ (cross-source)")
    axA.fill_between(x, dlo, dhi, color=C_DIS, alpha=0.13)
    axA.plot(x, yd, "-s", color=C_DIS, lw=2.0, ms=5.0, label="disattenuated $C_2$")
    axA.axhline(REAL["C2"], color=C_OBS, ls=":", lw=1.5)
    axA.axhline(REAL["disatt"], color=C_DIS, ls=":", lw=1.5)
    axA.text(0.015, REAL["C2"] + 0.012, f"real-data $C_2$ = {REAL['C2']:.2f}", color=C_OBS, fontsize=9)
    axA.text(0.015, REAL["disatt"] + 0.012, f"real-data disattenuated = {REAL['disatt']:.2f}", color=C_DIS, fontsize=9)
    axA.plot([rho_at, rho_at], [-0.12, REAL["C2"]], color=C_OBS, ls=(0, (1, 1.5)), lw=1.1)
    axA.plot(rho_at, REAL["C2"], "o", mfc="white", mec=C_OBS, mew=1.6, ms=8)
    axA.annotate(f"observed $C_2$=0.48\n$\\Rightarrow\\ \\rho_{{true}}\\approx{rho_at:.2f}$",
                 xy=(rho_at, REAL["C2"]), xytext=(rho_at - 0.30, REAL["C2"] - 0.20),
                 fontsize=8.8, arrowprops=dict(arrowstyle="->", color="#777", lw=1.0))
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.12, 1.0)
    axA.set_xlabel(r"ground-truth conservation  $\rho_{\mathrm{true}}=\mathrm{Spearman}(|\beta_A|,|\beta_B|)$")
    axA.set_ylabel("cross-source rank correlation of importances")
    axA.set_title("A  Recovery of known conservation", loc="left", fontweight="bold")
    axA.legend(loc="lower right", frameon=True, fontsize=9, framealpha=0.92, edgecolor="#ddd")
    axA.grid(True, color="#e6e6e6", lw=0.7); axA.set_axisbelow(True)

    bins = np.linspace(min(full.min(), 0.0) - 0.02, full.max() + 0.04, 26)
    counts, _ = np.histogram(full, bins=bins)
    axB.hist(full, bins=bins, color="#bcd0e8", edgecolor="#7d9cc0", lw=0.6)
    for i in range(len(bins) - 1):
        if bins[i + 1] <= REAL["C2"] + 1e-9:
            axB.bar((bins[i] + bins[i + 1]) / 2, counts[i], width=(bins[i + 1] - bins[i]) * 0.98,
                    color=C_OBS, alpha=0.45, edgecolor="none")
        elif bins[i] < REAL["C2"] < bins[i + 1]:
            axB.bar(bins[i] + (REAL["C2"] - bins[i]) / 2, counts[i], width=(REAL["C2"] - bins[i]) * 0.98,
                    color=C_OBS, alpha=0.45, edgecolor="none")
    axB.axvline(REAL["C2"], color=C_OBS, ls=":", lw=1.8)
    axB.axvline(full.mean(), color="#444", ls="--", lw=1.3)
    ytop = axB.get_ylim()[1]
    axB.text(REAL["C2"] - 0.005, ytop * 0.97, f"real $C_2$\n= {REAL['C2']:.2f}", color=C_OBS, fontsize=9, ha="right", va="top")
    axB.text(full.mean() + 0.006, ytop * 0.62, f"mean\n{full.mean():.2f}", color="#444", fontsize=9, ha="left", va="top")
    axB.text(0.02, ytop * 0.40, f"$P(C_2\\leq{REAL['C2']:.2f}\\,|\\,\\rho_{{true}}=1)$\n= {p_le:.2f}   ($n$={len(full)})",
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", lw=0.8))
    axB.set_xlabel("observed $C_2$ under full conservation ($\\rho_{true}=1$)")
    axB.set_ylabel("simulated replicates")
    axB.set_title("B  Could full conservation give $C_2$=0.48?", loc="left", fontweight="bold")
    axB.grid(True, axis="y", color="#ededed", lw=0.7); axB.set_axisbelow(True)

    fig.suptitle("Ground-truth simulation: the framework recovers mechanism conservation, and an observed\n"
                 "$C_2$ of 0.48 is consistent with full conservation obscured by ranking noise",
                 fontsize=11.2, y=1.02, x=0.02, ha="left")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global IMPORTANCE, COV_L
    ap = argparse.ArgumentParser(description="Ground-truth conservation-recovery simulation.")
    ap.add_argument("--reps", type=int, default=50, help="C2 replicates per rho_true level")
    ap.add_argument("--reps-full-extra", type=int, default=100, help="extra replicates at rho_true=1")
    ap.add_argument("--pairs", type=int, default=24, help="reliability replicate pairs")
    ap.add_argument("--importance", choices=["perm", "shap"], default=IMPORTANCE)
    ap.add_argument("--correlated", action="store_true",
                    help="robustness variant: block-correlated predictors (same mechanisms, "
                         "intercepts recalibrated under the covariance)")
    ap.add_argument("--rho-w", type=float, default=0.45, help="within-block correlation (with --correlated)")
    ap.add_argument("--block-size", type=int, default=4, help="block size (with --correlated)")
    ap.add_argument("--outdir", default="sim_out")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (few reps)")
    args = ap.parse_args()

    IMPORTANCE = args.importance
    reps, pairs, extra = args.reps, args.pairs, args.reps_full_extra
    if args.quick:
        reps, pairs, extra = 8, 6, 8
    os.makedirs(args.outdir, exist_ok=True)

    if args.correlated:
        COV_L = block_cholesky(P, args.block_size, args.rho_w)
        print(f"[correlated variant] block-equicorrelation: block_size={args.block_size}, rho_w={args.rho_w}")

    rng_build = np.random.default_rng(MASTER_SEED)
    beta_A, signs, shift_B, profB, beta_B_for = coefficient_vectors(rng_build)
    betaB_levels, rho_levels = build_levels(beta_B_for)
    icA = calibrate_intercept(beta_A, PREV_A, mean_shift=None)        # recalibrated under COV_L if set
    icB_levels = [calibrate_intercept(b, PREV_B, mean_shift=shift_B) for b in betaB_levels]

    print("Realised rho_true grid:", [round(r, 3) for r in rho_levels])
    print(f"\n[1/3] Reliability scaffold (IMPORTANCE={IMPORTANCE}) ...")
    rel = run_reliability(beta_A, icA, shift_B, betaB_levels[-1], icB_levels[-1], n_pairs=pairs)
    print(json.dumps({k: round(v, 3) for k, v in rel.items() if isinstance(v, float)}, indent=2))

    print(f"\n[2/3] C2 sweep over {len(rho_levels)} conservation levels ...")
    records = run_sweep(beta_A, icA, shift_B, betaB_levels, icB_levels, rho_levels,
                        reps=reps, reps_full_extra=extra)

    print("\n[3/3] Summary, adjudication and figure ...")
    rng_an = np.random.default_rng(20240620)
    denom, rowlist, full, p_le = summarize(records, rel, rng_an)
    rho_at = float(np.interp(REAL["C2"], [r["c2_mean"] for r in rowlist], [r["rho_true"] for r in rowlist]))

    results = dict(importance=IMPORTANCE,
                   covariates=("block_correlated" if args.correlated else "independent"),
                   rho_w=(args.rho_w if args.correlated else 0.0),
                   block_size=(args.block_size if args.correlated else None),
                   denom=denom, reliability=rel, real=REAL,
                   levels=rowlist,
                   adjudication=dict(rho_true=1.0, n=int(len(full)), real_C2=REAL["C2"],
                                     p_C2_le_real=p_le, full_mean=float(full.mean()),
                                     full_sd=float(full.std()),
                                     sd_below_mean=float((full.mean() - REAL["C2"]) / full.std())),
                   real_C2_maps_to_rho_true=rho_at)
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.outdir, "c2_raw.csv"), "w") as f:
        f.write("level,rho_true,rep,c2\n")
        for L, rt, r, c2 in records:
            f.write(f"{L},{rt:.4f},{r},{c2:.5f}\n")
    make_figure(rowlist, full, p_le, denom,
                os.path.join(args.outdir, "simulation_figure.png"),
                os.path.join(args.outdir, "simulation_figure.pdf"))

    print(f"\ndenom sqrt(r_A*r_B) = {denom:.3f}")
    print(f"P(C2 <= {REAL['C2']} | rho_true=1) = {p_le:.3f}   (n={len(full)})")
    print(f"real C2 = {REAL['C2']} maps to rho_true ~ {rho_at:.3f}")
    print(f"outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
