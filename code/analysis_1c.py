#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_1c.py  -- pre-configured, ready-to-run version of the 1C analysis
==========================================================================
Cross-database SHAP consistency test (MIMIC-IV training, eICU-CRD external).
The 28 FEATURES are hardcoded (from final_features.csv). The script AUTO-DISCOVERS
your cohort CSVs, the outcome column, and a saved model in DATA_DIR, then APPLIES
the pre-registered decision rule and prints a verdict. You should not need to edit
anything except possibly DATA_DIR.

WHAT IT NEEDS TO FIND in DATA_DIR (auto-discovered):
  * TWO patient-level CSVs, each = 28 FEATURES columns + a binary outcome (0/1):
      - MIMIC-IV (development, ~7,737 rows, ~4.4% events)
      - eICU     (external,    ~1,232 rows, ~13% events)
    Recognised by filename ('mimic'/'eicu') or by row count (more rows = MIMIC).
    A single combined CSV with a 'source/cohort/database' column is also handled.
  * OPTIONAL saved frozen MIMIC model (*.json/*.ubj/*.pkl/*.joblib); else retrain w/ HP_M.
  * final_features.csv is ignored.

RUN ORDER:
  pip install numpy pandas scipy scikit-learn xgboost shap matplotlib
  python analysis_1c.py --dry-run     # verify discovery, then STOP
  python analysis_1c.py               # real run -> ./results_1c
  python analysis_1c.py --selftest    # plumbing test (no xgboost/shap/data)
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG  (you should only ever need to touch DATA_DIR)
# =============================================================================
DATA_DIR = "/Users/ilizyue/Documents/xgb_extubation_failure/data"

MIMIC_CSV = None
EICU_CSV = None
# The published 0.86 came from the v2 pipeline, so the frozen model is the v2 pickle.
MODEL_M_PATH = "models/xgb_extubation_failure_v2.pkl"
OUTCOME_COL = "extubation_failure"   # composite: reintubated_48h OR death_within_48h_of_extubation

FEATURES = [
    "mv_duration_hours", "po2_min", "received_rrt", "cerebrovascular_disease",
    "sbp_mean", "po2_max", "lactate_max", "aniongap_min",
    "congestive_heart_failure", "last_fio2", "sodium_max", "sbp_max",
    "ptt_max", "spo2_mean", "platelets_max", "bun_max", "creatinine_min",
    "bicarbonate_min", "ptt_min", "potassium_min", "ph_min", "hemoglobin_min",
    "fibrinogen_min", "resp_rate_mean", "aniongap_max", "pco2_max", "pt_max",
    "inr_max",
]
MV_DURATION_COL = "mv_duration_hours"

# Set True ONLY if you use the v2 model AND your CSVs still contain raw (non-positive)
# mv_duration values; this applies the v2 rule (mv_duration <= 0 -> NaN) before the
# imputer fills them. Leave False if your CSVs are already preprocessed as you want.
APPLY_V2_MV_RULE = True

PREV_PRIMARY = 0.13
PREV_SENSITIVITY = 0.044

SEED = 20240611
HP_M = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, random_state=SEED)
HP_E_PRIMARY = dict(n_estimators=120, max_depth=3, learning_rate=0.05,
                    min_child_weight=5, reg_lambda=2.0, subsample=0.9, colsample_bytree=0.9,
                    objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", n_jobs=4, random_state=SEED)
HP_E_SAMEHP = dict(HP_M)

B_BOOT, N_PERM, N_SPLITS = 1000, 10000, 200
N_REPEATS, N_REPEATS_BOOT, CV_SPLITS = 20, 5, 5
RBO_P, TOPK = 0.9, 5

# ---- PRE-REGISTERED DECISION RULE (commit this file to git BEFORE the real run) ----
FRAC_THRESHOLD = 0.60
INTERP_RULE_NOTE = (
    "C2 (independent eICU model vs frozen MIMIC model, same eICU eval set) is the headline. "
    "Decision fixed before seeing results: SUPPORTS conservation iff C2 95%CI lower bound > R0_99 "
    "AND (C2 reaches >=60% of floor->ceiling OR C2 95%CI overlaps R2); NOT SUPPORTED iff C2 95%CI "
    "upper bound <= R0_99; otherwise PARTIAL/UNCERTAIN. C1 vs R1 is interpretive: if C1 falls "
    "within/near R1, state plainly that the frozen-model cross-database stability is largely structural."
)


def _xgb():
    import xgboost as xgb
    return xgb

def _shap():
    import shap
    return shap


# ----- SHAP helpers -----
def _to_pos_class(sv):
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv

def _names_of(o):
    try:
        fn = list(getattr(o, "feature_names_in_", []) or [])
        return fn or None
    except Exception:
        return None

def _est_order(est):
    fn = _names_of(est)
    if fn:
        return fn
    try:
        b = est.get_booster() if hasattr(est, "get_booster") else est
        fn = getattr(b, "feature_names", None)
        if fn:
            return list(fn)
    except Exception:
        pass
    return None

def _is_estimator(v):
    if hasattr(v, "get_booster"):
        return True
    try:
        return isinstance(v, _xgb().Booster)
    except Exception:
        return False

def _is_preprocessor(v):
    return hasattr(v, "transform") and hasattr(v, "statistics_")   # SimpleImputer-like

def _unwrap(obj):
    """Return (estimator, preprocessor_or_None, feature_order_or_None).
    Handles a bare model, a sklearn Pipeline, or a dict bundle {model, imputer, features,...}."""
    # sklearn Pipeline
    if hasattr(obj, "steps") and hasattr(obj, "named_steps"):
        est = obj.steps[-1][1]
        pre = obj[:-1] if len(obj.steps) > 1 else None
        order = _names_of(obj) or _est_order(est)
        return est, pre, order
    # dict bundle
    if isinstance(obj, dict):
        est = next((v for v in obj.values() if _is_estimator(v)), None)
        pre = next((v for v in obj.values() if _is_preprocessor(v)), None)
        order = None
        for k in ("features", "feature_names", "columns", "feature_cols"):
            if isinstance(obj.get(k), (list, tuple)):
                order = list(obj[k]); break
        order = order or _est_order(est) or _names_of(pre)
        return est, pre, order
    # bare estimator
    return obj, None, _est_order(obj)

def shap_abs_matrix(model, X_df, feats=None):
    """Per-row |SHAP| matrix, columns aligned to `feats` (default: X_df column order).
    Unwraps a bundled imputer+estimator, applies the imputer (so the SHAP matches the
    published median-imputed model), feeds X in the estimator's own feature order, then
    reindexes the output to `feats` (handles a model whose order differs from FEATURES)."""
    est, pre, order = _unwrap(model)
    feats = list(feats) if feats is not None else list(X_df.columns)
    order = order or feats
    miss = [f for f in order if f not in X_df.columns]
    if miss:
        raise SystemExit("Model expects features missing from data: %s" % miss[:8])
    Xo = X_df[order]
    Xt = np.asarray(pre.transform(Xo)) if pre is not None else np.asarray(Xo)
    M = np.abs(_to_pos_class(_shap().TreeExplainer(est).shap_values(Xt)))
    if M.shape[1] != len(order):
        return M                                        # shape mismatch: best effort
    pos = {f: i for i, f in enumerate(order)}
    return M[:, [pos[f] for f in feats]]

def train_xgb(X_df, y, hp):
    """Independent model = median imputation + XGBClassifier, mirroring the published
    pipeline (keep_empty_features so all-NaN columns like eICU inr_max are retained)."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", _xgb().XGBClassifier(**hp)),
    ])
    pipe.fit(X_df, np.asarray(y))
    return pipe

def crossfit_shap_abs_matrix(X_df, y, hp, seed, n_splits=CV_SPLITS):
    y = np.asarray(y)
    out = np.zeros((len(y), X_df.shape[1]), dtype=float)
    minc = int(min((y == 0).sum(), (y == 1).sum()))
    nsp = max(2, min(n_splits, minc))
    skf = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=seed)
    for tr, te in skf.split(X_df, y):
        clf = train_xgb(X_df.iloc[tr], y[tr], hp)
        out[te] = shap_abs_matrix(clf, X_df.iloc[te])
    return out


# ----- composition / sampling -----
def composed_indices(y, target_prev, rng):
    y = np.asarray(y)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    need_neg = int(round(len(pos) * (1 - target_prev) / target_prev))
    if need_neg <= len(neg):
        sel_pos, sel_neg = pos, rng.choice(neg, max(need_neg, 1), replace=False)
    else:
        need_pos = int(round(len(neg) * target_prev / (1 - target_prev)))
        sel_pos = rng.choice(pos, max(min(need_pos, len(pos)), 1), replace=False)
        sel_neg = neg
    idx = np.concatenate([sel_pos, sel_neg]); rng.shuffle(idx)
    return idx

def compose_average(abs_matrix, y, target_prev, rng, n_repeats):
    vecs = [abs_matrix[composed_indices(y, target_prev, rng)].mean(axis=0) for _ in range(n_repeats)]
    return np.mean(np.vstack(vecs), axis=0)

def downsample_to_events(idx_pos, idx_neg, n_events, target_prev, rng):
    n_pos = min(n_events, len(idx_pos))
    sel_pos = rng.choice(idx_pos, max(n_pos, 1), replace=False)
    n_neg = min(int(round(len(sel_pos) * (1 - target_prev) / target_prev)), len(idx_neg))
    sel_neg = rng.choice(idx_neg, max(n_neg, 1), replace=False)
    return np.concatenate([sel_pos, sel_neg])


# ----- ranking metrics -----
def rank_desc(v):
    v = np.asarray(v); order = np.argsort(-v)
    r = np.empty(len(v), dtype=float); r[order] = np.arange(1, len(v) + 1)
    return r

def rbo(rank_a, rank_b, p=RBO_P):
    k = max(len(rank_a), len(rank_b)); setA, setB = set(), set(); s = 0.0
    for d in range(1, k + 1):
        if d - 1 < len(rank_a): setA.add(rank_a[d - 1])
        if d - 1 < len(rank_b): setB.add(rank_b[d - 1])
        s += (p ** (d - 1)) * (len(setA & setB) / d)
    return (1 - p) * s + (p ** k) * (len(setA & setB) / k)

def rank_metrics(a, b, names, rbo_p=RBO_P, topk=TOPK):
    a = np.asarray(a, float); b = np.asarray(b, float); eps = 1e-12
    ra = [names[i] for i in np.argsort(-a)]; rb = [names[i] for i in np.argsort(-b)]
    jacc = len(set(ra[:topk]) & set(rb[:topk])) / len(set(ra[:topk]) | set(rb[:topk]))
    return dict(spearman=float(spearmanr(a, b).correlation),
                pearson_log=float(pearsonr(np.log(a + eps), np.log(b + eps))[0]),
                rbo=float(rbo(ra, rb, rbo_p)), top5_jaccard=float(jacc))


# ----- reference levels + bootstrap -----
def permutation_null(a, b, rng, n_perm=N_PERM):
    a = np.asarray(a, float); b = np.asarray(b, float).copy()
    obs = spearmanr(a, b).correlation; null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(b); null[i] = spearmanr(a, b).correlation
    return float(obs), null, float((np.sum(null >= obs) + 1) / (n_perm + 1))

def r1_frozen_within(absM, y_M, prev, rng, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    y_M = np.asarray(y_M); idx = np.arange(absM.shape[0]); out = []
    for _ in range(n_splits):
        rng.shuffle(idx); h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        s1 = compose_average(absM[h1], y_M[h1], prev, rng, n_repeats)
        s2 = compose_average(absM[h2], y_M[h2], prev, rng, n_repeats)
        out.append(spearmanr(s1, s2).correlation)
    return np.array(out)

def r2_independent_within(X_M, y_M, hp, prev, n_events, seed,
                          n_splits=N_SPLITS, n_repeats=N_REPEATS, cv_splits=CV_SPLITS):
    rng = np.random.default_rng(seed); y = np.asarray(y_M)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; out = []
    for s in range(n_splits):
        rng.shuffle(pos); rng.shuffle(neg)
        posA, posB = pos[: len(pos) // 2], pos[len(pos) // 2:]
        negA, negB = neg[: len(neg) // 2], neg[len(neg) // 2:]
        vecs = []
        for ph, nh in ((posA, negA), (posB, negB)):
            sel = downsample_to_events(ph, nh, n_events, prev, rng)
            cf = crossfit_shap_abs_matrix(X_M.iloc[sel], y[sel], hp, seed + s + 1, cv_splits)
            vecs.append(compose_average(cf, y[sel], prev, rng, n_repeats))
        out.append(spearmanr(vecs[0], vecs[1]).correlation)
    return np.array(out)

def bootstrap_c1(absM, y_M, absME, y_E, prev, seed, B=B_BOOT, n_repeats=N_REPEATS_BOOT):
    y_M = np.asarray(y_M); y_E = np.asarray(y_E); boot = []
    for b in range(B):
        rng = np.random.default_rng(seed + 50000 + b)
        im = rng.integers(0, len(y_M), len(y_M)); ie = rng.integers(0, len(y_E), len(y_E))
        s_m = compose_average(absM[im], y_M[im], prev, rng, n_repeats)
        s_me = compose_average(absME[ie], y_E[ie], prev, rng, n_repeats)
        boot.append(spearmanr(s_m, s_me).correlation)
    return np.array(boot)

def bootstrap_c2(model_M, X_E, y_E, hp_E, names, prev, seed,
                 B=B_BOOT, n_repeats=N_REPEATS_BOOT, cv_splits=CV_SPLITS):
    # NOTE: cross-fitting inside a bootstrap can mildly leak via duplicated rows
    # (second-order). For a stricter estimate use an out-of-bag/.632 scheme.
    y_E = np.asarray(y_E); boot = []; cnt = np.zeros(len(names)); used = 0
    for b in range(B):
        rng = np.random.default_rng(seed + 1 + b)
        idx = rng.integers(0, len(y_E), len(y_E))
        Xb = X_E.iloc[idx].reset_index(drop=True); yb = y_E[idx]
        if (yb == 1).sum() < cv_splits or (yb == 0).sum() < cv_splits:
            continue
        me = shap_abs_matrix(model_M, Xb)
        ee = crossfit_shap_abs_matrix(Xb, yb, hp_E, seed + 1 + b, cv_splits)
        s_me = compose_average(me, yb, prev, rng, n_repeats)
        s_e = compose_average(ee, yb, prev, rng, n_repeats)
        boot.append(spearmanr(s_me, s_e).correlation)
        for i in np.argsort(-s_e)[:TOPK]:
            cnt[i] += 1
        used += 1
    return np.array(boot), cnt / max(used, 1)

def c2_pipeline(X_M, y_M, X_E, y_E, feats, hp_M, hp_E, prev, seed, model_M=None):
    Xm, Xe = X_M[feats], X_E[feats]
    if model_M is None:
        model_M = train_xgb(Xm, y_M, hp_M)
    rng = np.random.default_rng(seed + 7)
    absME = shap_abs_matrix(model_M, Xe)
    absE = crossfit_shap_abs_matrix(Xe, y_E, hp_E, seed + 7)
    S_ME = compose_average(absME, y_E, prev, rng, N_REPEATS)
    S_E = compose_average(absE, y_E, prev, rng, N_REPEATS)
    return S_ME, S_E, rank_metrics(S_ME, S_E, feats), absME, absE, model_M


# ----- pre-registered decision -----
def decide(c2, c2_ci, r0_null, R2, frac_threshold=FRAC_THRESHOLD):
    floor = float(np.percentile(r0_null, 99))
    ceiling = float(np.median(R2))
    r2_lo = float(np.percentile(R2, 2.5))
    lo, hi = float(c2_ci[0]), float(c2_ci[1])
    frac = (c2 - floor) / max(ceiling - floor, 1e-9)
    if hi <= floor:
        v = "NOT SUPPORTED"
        why = ("C2 95%% CI [%.3f, %.3f] does not clear the chance floor (R0 99th=%.3f): the "
               "cross-database consistency is largely a frozen-model artefact." % (lo, hi, floor))
    elif lo > floor and (frac >= frac_threshold or hi >= r2_lo):
        v = "SUPPORTS CONSERVATION"
        why = ("C2=%.3f, CI [%.3f, %.3f] entirely above chance (floor=%.3f) and reaches %.0f%% of "
               "the way to the within-system ceiling (median R2=%.3f; frac=%.2f)."
               % (c2, lo, hi, floor, frac_threshold * 100, ceiling, frac))
    else:
        v = "PARTIAL / UNCERTAIN"
        why = ("C2=%.3f is above chance (floor=%.3f) but below the within-system reproducibility "
               "ceiling (median R2=%.3f; frac=%.2f); most plausibly limited by the eICU event count."
               % (c2, floor, ceiling, frac))
    return v, why, dict(floor=floor, ceiling=ceiling, frac=float(frac), c2=float(c2), c2_ci=[lo, hi])


# ----- figures -----
def slopegraph(ax, lv, rv, names, ll, rl, title, rho_text, label_top=8):
    n = len(names); a, b = rank_desc(lv), rank_desc(rv)
    for i in range(n):
        ax.plot([0, 1], [a[i], b[i]], "-", color="0.85", lw=0.7, zorder=1)
    for i in np.argsort(-np.asarray(lv))[:label_top]:
        ax.plot([0, 1], [a[i], b[i]], "-", lw=1.6, zorder=2)
        ax.text(-0.04, a[i], names[i], ha="right", va="center", fontsize=6.5)
        ax.text(1.04, b[i], names[i], ha="left", va="center", fontsize=6.5)
    ax.scatter([0] * n, a, s=9, color="0.3", zorder=3); ax.scatter([1] * n, b, s=9, color="0.3", zorder=3)
    ax.set_ylim(n + 0.6, 0.4); ax.set_xlim(-0.65, 1.65)
    ax.set_xticks([0, 1]); ax.set_xticklabels([ll, rl], fontsize=8); ax.set_yticks([])
    ax.set_title(title + "\n" + rho_text, fontsize=9)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)

def figure_A(S_M, S_ME, S_E, names, c1, c2, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    slopegraph(ax[0], S_M, S_ME, names, "MIMIC", "eICU", "(a) C1: frozen model, two databases",
               "Spearman rho = %.2f" % c1["spearman"])
    slopegraph(ax[1], S_ME, S_E, names, "MIMIC-trained", "eICU-trained",
               "(b) C2: same eICU eval, two training sources", "Spearman rho = %.2f" % c2["spearman"])
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def _interval(ax, y, lo, hi, mid, marker="o", color="tab:blue"):
    ax.plot([lo, hi], [y, y], "-", lw=2.5, color=color, solid_capstyle="round")
    ax.plot([mid], [y], marker, color=color, ms=7, zorder=4)

def figure_B(r0_null, R1, R2, c1, c1ci, c2, c2ci, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = ["R0 null (permuted)", "R1 frozen / within-MIMIC", "R2 independent / within (~N_E)",
              "C1 frozen | shift", "C2 independent | same eval"]
    p1, p99 = np.percentile(r0_null, [1, 99])
    _interval(ax, 4, p1, p99, np.median(r0_null), "s", "0.55"); ax.axvline(p99, color="0.55", lw=0.8, ls="--")
    _interval(ax, 3, *np.percentile(R1, [2.5, 97.5]), np.median(R1), color="tab:green")
    _interval(ax, 2, *np.percentile(R2, [2.5, 97.5]), np.median(R2), color="tab:olive")
    _interval(ax, 1, c1ci[0], c1ci[1], c1, "D", "tab:blue")
    _interval(ax, 0, c2ci[0], c2ci[1], c2, "D", "tab:red")
    ax.set_yticks([4, 3, 2, 1, 0]); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.25, 1.0); ax.set_xlabel("Spearman rho between mean|SHAP| rankings")
    ax.axvline(0.0, color="0.9", lw=1)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def figure_C(freq, names, path, topn=12):
    order = np.argsort(-freq)[:topn]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([names[i] for i in order][::-1], freq[order][::-1], color="tab:red", alpha=0.8)
    ax.set_xlim(0, 1); ax.set_xlabel("Frequency in eICU-model top-%d across bootstraps" % TOPK)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ----- summary table -----
def _ci(rho, ci): return "%.2f (%.2f, %.2f)" % (rho, ci[0], ci[1])

def build_summary(rows, csv_path, md_path):
    pd.DataFrame(rows, columns=["comparison", "spearman_ci", "rbo", "top5_jaccard", "note"]).to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("| Comparison | Spearman rho (95% CI) | RBO | Top-5 Jaccard | Note |\n|---|---|---|---|---|\n")
        for r in rows:
            rb = "%.2f" % r[2] if isinstance(r[2], float) else r[2]
            jj = "%.2f" % r[3] if isinstance(r[3], float) else r[3]
            f.write("| %s | %s | %s | %s | %s |\n" % (r[0], r[1], rb, jj, r[4]))


# ----- data discovery -----
def _read_head(p, n=5):
    return pd.read_csv(p, nrows=n)

def _looks_like_feature_list(p):
    try:
        cols = set(c.lower() for c in _read_head(p, 1).columns)
    except Exception:
        return False
    return "feature" in cols and len(cols) <= 4

def detect_outcome(df):
    if OUTCOME_COL:
        if OUTCOME_COL not in df.columns:
            raise SystemExit("OUTCOME_COL '%s' not in data columns." % OUTCOME_COL)
        return OUTCOME_COL
    cands = []
    for c in df.columns:
        if c in FEATURES:
            continue
        vals = set(pd.unique(df[c].dropna()))
        if vals and vals <= {0, 1}:
            cands.append(c)
        elif vals and vals <= {True, False}:
            cands.append(c)
    if len(cands) == 1:
        return cands[0]
    pat = re.compile(r"(outcome|label|fail|target|event|reintub|death|^y$)", re.I)
    named = [c for c in cands if pat.search(c)]
    if len(named) == 1:
        return named[0]
    raise SystemExit("Outcome column ambiguous. Binary non-feature candidates: %s.\n"
                     "Set OUTCOME_COL near the top of the script." % cands)

def _validate_features(df, tag):
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit("[%s] is missing %d FEATURES columns: %s" % (tag, len(missing), missing))

def discover(data_dir):
    if MIMIC_CSV and EICU_CSV:
        dM, dE = pd.read_csv(MIMIC_CSV), pd.read_csv(EICU_CSV)
    else:
        if not os.path.isdir(data_dir):
            raise SystemExit("DATA_DIR does not exist: %s\nEdit DATA_DIR or pass --datadir." % data_dir)
        csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not csvs:
            raise SystemExit("No CSV files found in %s" % data_dir)
        patient = []
        for p in csvs:
            if _looks_like_feature_list(p):
                continue
            cols = set(_read_head(p).columns)
            if sum(f in cols for f in FEATURES) >= 20:
                patient.append(p)
        if len(patient) >= 2:
            mimic = next((p for p in patient if "mimic" in os.path.basename(p).lower()), None)
            eicu = next((p for p in patient if "eicu" in os.path.basename(p).lower()), None)
            if not (mimic and eicu):
                if len(patient) == 2:
                    counts = sorted(((sum(1 for _ in open(p)) - 1, p) for p in patient), reverse=True)
                    mimic, eicu = counts[0][1], counts[1][1]
                else:
                    raise SystemExit("Found %d candidate patient CSVs but cannot tell MIMIC from eICU.\n"
                                     "Rename so one filename contains 'mimic' and one 'eicu', or set "
                                     "MIMIC_CSV / EICU_CSV.\nCandidates: %s"
                                     % (len(patient), [os.path.basename(p) for p in patient]))
            dM, dE = pd.read_csv(mimic), pd.read_csv(eicu)
            print("   MIMIC <- %s\n   eICU  <- %s" % (os.path.basename(mimic), os.path.basename(eicu)))
        elif len(patient) == 1:
            df = pd.read_csv(patient[0]); src = None
            for c in df.columns:
                if c in FEATURES:
                    continue
                vals = set(str(v).lower() for v in pd.unique(df[c].dropna()))
                if len(vals) == 2 and (vals & {"mimic", "mimic-iv", "mimiciv"}) and (vals & {"eicu", "eicu-crd", "eicucrd"}):
                    src = c; break
            if src is None:
                raise SystemExit("Only one patient CSV (%s) and no MIMIC/eICU split column.\n"
                                 "Provide separate files or set MIMIC_CSV / EICU_CSV." % os.path.basename(patient[0]))
            low = df[src].astype(str).str.lower()
            dM = df[low.str.contains("mimic")].copy(); dE = df[low.str.contains("eicu")].copy()
            print("   split single file %s by column '%s'" % (os.path.basename(patient[0]), src))
        else:
            raise SystemExit("No patient-level CSVs found in %s (need the 28 FEATURES + an outcome column)." % data_dir)

    _validate_features(dM, "MIMIC"); _validate_features(dE, "eICU")
    oc_M, oc_E = detect_outcome(dM), detect_outcome(dE)

    def _xy(df, oc, tag):
        n0 = len(df)
        df = df[df[oc].notna()].copy()
        if n0 - len(df):
            print("   [%s] dropped %d rows with missing outcome ('%s')" % (tag, n0 - len(df), oc))
        return df[FEATURES].copy(), df[oc].astype(int).to_numpy()

    X_M, y_M = _xy(dM, oc_M, "MIMIC")
    X_E, y_E = _xy(dE, oc_E, "eICU")
    print("   outcome column: MIMIC='%s'  eICU='%s'" % (oc_M, oc_E))

    if APPLY_V2_MV_RULE and MV_DURATION_COL in FEATURES:
        for tag, X in (("MIMIC", X_M), ("eICU", X_E)):
            bad = X[MV_DURATION_COL] <= 0
            X.loc[bad, MV_DURATION_COL] = np.nan
            print("   v2 rule: set %d non-positive %s -> NaN in %s" % (int(bad.sum()), MV_DURATION_COL, tag))

    model_path = MODEL_M_PATH
    if model_path is None and not (MIMIC_CSV and EICU_CSV):
        models = []
        for ext in ("*.json", "*.ubj", "*.model", "*.bin", "*.pkl", "*.joblib"):
            models += sorted(glob.glob(os.path.join(data_dir, ext)))
        if models:
            mimic_named = [p for p in models if "mimic" in os.path.basename(p).lower()]
            model_path = (mimic_named or models)[0]
            if len(models) > 1:
                print("   model files found: %s" % [os.path.basename(p) for p in models])
                print("   -> using: %s  (override with --model to pick another, e.g. the v2 model)"
                      % os.path.basename(model_path))
    return X_M, y_M, X_E, y_E, model_path

def load_model_M(model_path, X_M, y_M):
    if model_path:
        ext = os.path.splitext(model_path)[1].lower()
        loaded, err = None, None
        if ext in (".pkl", ".pickle", ".joblib", ".sav"):
            try:
                import joblib
                loaded = joblib.load(model_path)
            except Exception as e1:
                try:
                    import pickle
                    with open(model_path, "rb") as f:
                        loaded = pickle.load(f)
                except Exception as e2:
                    err = "%s / %s" % (e1, e2)
        else:                                        # native xgboost format
            try:
                m = _xgb().XGBClassifier(); m.load_model(model_path); loaded = m
            except Exception as e:
                err = str(e)
        if loaded is not None:
            est, pre, order = _unwrap(loaded)
            if est is None:
                print(">> %s loaded but no xgboost model inside (%s); will retrain."
                      % (os.path.basename(model_path), type(loaded).__name__))
            else:
                if order is not None and set(order) != set(FEATURES):
                    missing = sorted(set(FEATURES) - set(order))
                    extra = sorted(set(order) - set(FEATURES))
                    raise SystemExit(
                        "Loaded model's features do NOT match FEATURES.\n"
                        "  in FEATURES but not in model: %s\n"
                        "  in model but not in FEATURES: %s\n"
                        "The saved model was trained on a different feature set; "
                        "align FEATURES with the model (or check you loaded the right .pkl)." % (missing, extra))
                print(">> loaded frozen MIMIC model: %s (%s features%s)"
                      % (os.path.basename(model_path),
                         len(order) if order else "?",
                         ", median-imputer bundled" if pre is not None else ", NO imputer found"))
                return loaded                         # return whole bundle; shap_abs_matrix unwraps + imputes
        else:
            print(">> could not load %s (%s); will retrain." % (os.path.basename(model_path), err))
    print(">> no usable saved model -> retraining MIMIC model with HP_M (reproduces published model)")
    return train_xgb(X_M[FEATURES], y_M, HP_M)


# ----- modes -----
def run_dry(datadir):
    print(">> DRY RUN: discovery only (no modelling)\n   scanning: %s" % datadir)
    X_M, y_M, X_E, y_E, model_path = discover(datadir)
    for tag, X, y in (("MIMIC", X_M, y_M), ("eICU", X_E, y_E)):
        nan_cols = [c for c in FEATURES if X[c].isna().all()]
        zero_cols = [c for c in FEATURES if (X[c].fillna(0) == 0).all()]
        print("   [%s] rows=%d events=%d prevalence=%.3f  all-NaN=%s  all-zero=%s"
              % (tag, len(y), int(y.sum()), y.mean(), nan_cols or "none", zero_cols or "none"))
        if MV_DURATION_COL in X.columns:
            mv = X[MV_DURATION_COL]
            print("        %s: non-positive=%d, NaN=%d  (v2 rule %s)"
                  % (MV_DURATION_COL, int((mv <= 0).sum()), int(mv.isna().sum()),
                     "applied" if APPLY_V2_MV_RULE else "OFF"))
    if not APPLY_V2_MV_RULE:
        print("   note: if eICU shows many non-positive mv_duration above, the v2 rule has NOT")
        print("         been applied yet -> set APPLY_V2_MV_RULE=True to reproduce the v2 pipeline.")
    print("   model: %s" % (os.path.basename(model_path) if model_path else "NONE (will retrain)"))
    print(">> Discovery OK. Re-run without --dry-run to execute the analysis.")

def run_pipeline(X_M, y_M, X_E, y_E, model_path, outdir, prev=PREV_PRIMARY):
    os.makedirs(outdir, exist_ok=True); feats = FEATURES
    print(">> samples: MIMIC n=%d ev=%d | eICU n=%d ev=%d | features=%d"
          % (len(y_M), int(y_M.sum()), len(y_E), int(y_E.sum()), len(feats)))
    print(">> PRE-REGISTERED RULE:\n   " + INTERP_RULE_NOTE)
    model_M = load_model_M(model_path, X_M, y_M)
    rng = np.random.default_rng(SEED)

    absM = shap_abs_matrix(model_M, X_M[feats])
    S_ME, S_E, c2, absME, absE, _ = c2_pipeline(X_M, y_M, X_E, y_E, feats, HP_M, HP_E_PRIMARY, prev, SEED, model_M)
    S_M = compose_average(absM, y_M, prev, rng, N_REPEATS)
    c1 = rank_metrics(S_M, S_ME, feats)
    print(">> C1=%.3f  C2=%.3f" % (c1["spearman"], c2["spearman"]))

    r0_obs, r0_null, r0_p = permutation_null(S_ME, S_E, np.random.default_rng(SEED + 2), N_PERM)
    R1 = r1_frozen_within(absM, y_M, prev, np.random.default_rng(SEED + 3), N_SPLITS)
    n_events = int(y_E.sum())
    R2 = r2_independent_within(X_M, y_M, HP_E_PRIMARY, prev, n_events, SEED + 4, N_SPLITS)
    print(">> R0 p=%.4f | R1 med=%.3f | R2 med=%.3f" % (r0_p, np.median(R1), np.median(R2)))

    c1b = bootstrap_c1(absM, y_M, absME, y_E, prev, SEED, B_BOOT)
    c2b, top5 = bootstrap_c2(model_M, X_E[feats], y_E, HP_E_PRIMARY, feats, prev, SEED, B_BOOT)
    c1ci = tuple(np.percentile(c1b, [2.5, 97.5])); c2ci = tuple(np.percentile(c2b, [2.5, 97.5]))

    _, _, c2_same, _, _, _ = c2_pipeline(X_M, y_M, X_E, y_E, feats, HP_M, HP_E_SAMEHP, prev, SEED + 11)
    feats_nomv = [f for f in feats if f != MV_DURATION_COL]
    _, _, c2_nomv, _, _, _ = c2_pipeline(X_M, y_M, X_E, y_E, feats_nomv, HP_M, HP_E_PRIMARY, prev, SEED + 12)

    figure_A(S_M, S_ME, S_E, feats, c1, c2, os.path.join(outdir, "figureA_slopegraphs.png"))
    figure_B(r0_null, R1, R2, c1["spearman"], c1ci, c2["spearman"], c2ci, os.path.join(outdir, "figureB_reference_levels.png"))
    figure_C(top5, feats, os.path.join(outdir, "figureC_eicu_top5_stability.png"))

    rows = [
        ("C1 = rho(S_M, S_M->E)", _ci(c1["spearman"], c1ci), c1["rbo"], c1["top5_jaccard"], "frozen model | covariate shift"),
        ("C2 = rho(S_M->E, S_E) [PRIMARY]", _ci(c2["spearman"], c2ci), c2["rbo"], c2["top5_jaccard"], "independent eICU model | same eval"),
        ("C2 (same-hp eICU)", "%.2f" % c2_same["spearman"], c2_same["rbo"], c2_same["top5_jaccard"], "sensitivity (expected overfit)"),
        ("C2 (drop mv_duration)", "%.2f" % c2_nomv["spearman"], c2_nomv["rbo"], c2_nomv["top5_jaccard"], "sensitivity"),
        ("R0 permutation (99th pct)", "%.2f" % np.percentile(r0_null, 99), "-", "-", "floor; empirical p=%.4f" % r0_p),
        ("R1 frozen / within-MIMIC", "%.2f (%.2f, %.2f)" % (np.median(R1), *np.percentile(R1, [2.5, 97.5])), "-", "-", "ceiling for C1"),
        ("R2 independent / within (~%d ev)" % n_events, "%.2f (%.2f, %.2f)" % (np.median(R2), *np.percentile(R2, [2.5, 97.5])), "-", "-", "ceiling for C2"),
    ]
    build_summary(rows, os.path.join(outdir, "summary_rho_table.csv"), os.path.join(outdir, "summary_rho_table.md"))

    verdict, why, det = decide(c2["spearman"], c2ci, r0_null, R2)
    with open(os.path.join(outdir, "VERDICT.txt"), "w") as f:
        f.write("PRE-REGISTERED RULE:\n%s\n\nVERDICT: %s\nWHY: %s\nDETAIL: %s\n" % (INTERP_RULE_NOTE, verdict, why, det))
    np.savez(os.path.join(outdir, "arrays.npz"), S_M=S_M, S_ME=S_ME, S_E=S_E,
             r0_null=r0_null, R1=R1, R2=R2, c1_boot=c1b, c2_boot=c2b, top5=top5, features=np.array(feats))

    print("\n" + "=" * 70 + "\nVERDICT: %s\n%s\n" % (verdict, why) + "=" * 70)
    print(">> wrote figures, summary_rho_table.{csv,md}, VERDICT.txt, arrays.npz to %s" % outdir)


def make_synthetic_tabular(seed=SEED):
    rng = np.random.default_rng(seed); p = len(FEATURES)
    def gen(n, prev):
        X = rng.normal(size=(n, p)); w = np.linspace(1.5, 0.0, p)
        logit = X @ w - np.log((1 - prev) / prev) - 0.5 * X[:, 0] ** 2
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return pd.DataFrame(X, columns=FEATURES), y
    Xm, ym = gen(2500, 0.044); Xe, ye = gen(1232, 0.13)
    return Xm, ym, Xe, ye

def run_selftest(outdir):
    print(">> SELFTEST: synthetic SHAP matrices (no xgboost/shap)")
    rng = np.random.default_rng(SEED); names = FEATURES; p = len(names)
    base = np.exp(np.linspace(0.5, -2.0, p))
    mk = lambda n, prof, s: np.abs(prof[None, :] * np.exp(rng.normal(scale=s, size=(n, p))))
    yM = (rng.random(2000) < 0.044).astype(int); yE = (rng.random(1232) < 0.13).astype(int)
    absM = mk(len(yM), base, 0.3); absME = mk(len(yE), base, 0.4)
    prof2 = base.copy(); prof2[2], prof2[5] = prof2[5], prof2[2]; absE = mk(len(yE), prof2, 0.6)
    S_M = compose_average(absM, yM, PREV_PRIMARY, rng, 8)
    S_ME = compose_average(absME, yE, PREV_PRIMARY, rng, 8)
    S_E = compose_average(absE, yE, PREV_PRIMARY, rng, 8)
    c1 = rank_metrics(S_M, S_ME, names); c2 = rank_metrics(S_ME, S_E, names)
    r0_obs, r0_null, r0_p = permutation_null(S_ME, S_E, rng, 1000)
    R1 = r1_frozen_within(absM, yM, PREV_PRIMARY, rng, 20, 5)
    c1b = bootstrap_c1(absM, yM, absME, yE, PREV_PRIMARY, SEED, 60, 4)
    c2b = np.clip(rng.normal(c2["spearman"], 0.05, 200), -1, 1)
    R2 = np.clip(rng.normal(0.8, 0.05, 50), -1, 1)
    top5 = np.zeros(p)
    for _ in range(200):
        for i in np.argsort(-(absE[rng.integers(0, len(yE), len(yE))].mean(0)))[:TOPK]:
            top5[i] += 1
    top5 /= 200
    os.makedirs(outdir, exist_ok=True)
    c1ci = tuple(np.percentile(c1b, [2.5, 97.5])); c2ci = tuple(np.percentile(c2b, [2.5, 97.5]))
    figure_A(S_M, S_ME, S_E, names, c1, c2, os.path.join(outdir, "figureA_slopegraphs.png"))
    figure_B(r0_null, R1, R2, c1["spearman"], c1ci, c2["spearman"], c2ci, os.path.join(outdir, "figureB_reference_levels.png"))
    figure_C(top5, names, os.path.join(outdir, "figureC_eicu_top5_stability.png"))
    v, why, _ = decide(c2["spearman"], c2ci, r0_null, R2)
    print("   C1=%.3f C2=%.3f R0_p=%.4f -> VERDICT(selftest, synthetic): %s" % (c1["spearman"], c2["spearman"], r0_p, v))
    print(">> SELFTEST OK; figures in %s" % outdir)


def main():
    ap = argparse.ArgumentParser(description="1C cross-database SHAP consistency (pre-configured)")
    ap.add_argument("--datadir", default=DATA_DIR)
    ap.add_argument("--model", default=MODEL_M_PATH)
    ap.add_argument("--outdir", default="./results_1c")
    ap.add_argument("--prev", type=float, default=PREV_PRIMARY)
    ap.add_argument("--dry-run", action="store_true", help="discovery only; verify files/columns then stop")
    ap.add_argument("--selftest", action="store_true", help="plumbing test (no xgboost/shap, no data)")
    ap.add_argument("--smoke", action="store_true", help="end-to-end on synthetic data (needs xgboost+shap)")
    ap.add_argument("--quick", action="store_true", help="real data but reduced iterations (fast end-to-end check)")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args.outdir); return
    if args.dry_run:
        run_dry(args.datadir); return
    if args.smoke:
        global B_BOOT, N_PERM, N_SPLITS, N_REPEATS
        B_BOOT, N_PERM, N_SPLITS, N_REPEATS = 50, 1000, 20, 5
        Xm, ym, Xe, ye = make_synthetic_tabular()
        run_pipeline(Xm, ym, Xe, ye, None, args.outdir, args.prev); return
    if args.quick:
        globals()["B_BOOT"], globals()["N_PERM"], globals()["N_SPLITS"] = 50, 1000, 20
        print(">> QUICK MODE: reduced iterations (B_BOOT=50, N_PERM=1000, N_SPLITS=20) — for a fast check, not the final result")
        X_M, y_M, X_E, y_E, model_path = discover(args.datadir)
        run_pipeline(X_M, y_M, X_E, y_E, args.model or model_path, args.outdir, args.prev); return

    X_M, y_M, X_E, y_E, model_path = discover(args.datadir)
    run_pipeline(X_M, y_M, X_E, y_E, args.model or model_path, args.outdir, args.prev)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
