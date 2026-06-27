"""
Academic-style figures for the extubation-failure XGBoost paper.

Style guide applied:
  - Typography: Liberation Sans (Arial-compatible) at 8.5–9.5 pt
  - Title weight: regular (not bold), titlepad small
  - Restrained palette: deep navy / brick red / warm gold
  - Lines: 1.5 lw for primary curves, 0.7 for axes/spines
  - CI bands: alpha 0.10 (lighter for less visual weight)
  - Grids: very light dotted gray, only where useful
  - CONSORT: white-filled boxes, thin gray border, colored accent stripes
  - Beeswarm colormap: restrained RdBu_r (no yellow midtone)
  - Legends: inline / no frame, minimal padding

Outputs PNG @ 300 dpi + PDF (vector, fonttype 42).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                              average_precision_score, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr

# ============================================================================
# Theme — academic/journal style
# ============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8.5,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.2,
    "savefig.dpi": 300,
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "grid.color": "#dcdcdc",
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "grid.alpha": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stixsans",
})

# Cohort palette — desaturated, journal-style
C_MIMIC = "#1F4E79"      # deep navy
C_EICU  = "#A0322F"      # restrained brick red
C_CLEAN = "#C28A2C"      # muted ochre / warm gold

# Reference / utility
C_PERFECT = "#888888"
C_GRID    = "#e8e8e8"
C_TEXT    = "#222222"
C_MUTED   = "#666666"

# Operating-threshold colors (subtler, derive from cohort tones)
C_THR_SENS80 = "#5E548E"  # muted slate purple
C_THR_YOUDEN = "#2D6A4F"  # forest green

# Beeswarm divergent: smooth blue → gray → red, no yellow midtone
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_blue_red",
    ["#2A6F97", "#74A2BE", "#BDBDBD", "#C28A8A", "#9D2A30"],
    N=256,
)


# ============================================================================
# Data loading
# ============================================================================
mimic_oof = pd.read_csv("data/predictions/mimic_oof_predictions.csv")
y_mimic = mimic_oof.y_true.values.astype(int)
p_mimic = mimic_oof.XGBoost.values

eicu_full = pd.read_csv("data/predictions/eicu_predictions_full_1232.csv")
y_eicu = eicu_full.y_true.values.astype(int)
p_eicu = eicu_full.prob.values

# Load the pre-computed clean-subset mask (mv_duration_hours > 0).
# This 1,232-element boolean array is derived from raw eICU but contains
# no PHI, so it is committed to the repository — figures reproduce exactly
# without PhysioNet credentialed access.
mv_pos = np.load("data/predictions/eicu_mv_positive_mask.npy")
y_eicu_clean = y_eicu[mv_pos]
p_eicu_clean = p_eicu[mv_pos]

T_YOUDEN  = 0.0208
T_SENS80  = 0.0154
T_DEFAULT = 0.5

COHORTS = [
    dict(name="MIMIC-IV (internal CV)",    short="MIMIC-IV",
         y=y_mimic, p=p_mimic, color=C_MIMIC, marker="o"),
    dict(name="eICU full (external)",      short="eICU full",
         y=y_eicu, p=p_eicu, color=C_EICU, marker="s"),
    dict(name="eICU clean (mv>0)",         short="eICU clean",
         y=y_eicu_clean, p=p_eicu_clean, color=C_CLEAN, marker="^"),
]

# Authoritative AUROC / AUPRC 95% CIs from Table 2 (main text; 1,000 stratified
# bootstrap). Fig 2 panel labels read these so the figure and Table 2 are exactly
# consistent. The ROC/PR shaded bands remain this script's own pointwise-curve
# bootstrap (a different quantity from the scalar-area CI; Table 2 has no bands).
TABLE2_CI = {
    "MIMIC-IV":   dict(auroc=(0.850, 0.892), auprc=(0.278, 0.382)),
    "eICU full":  dict(auroc=(0.545, 0.646), auprc=(0.180, 0.298)),
    "eICU clean": dict(auroc=(0.732, 0.886), auprc=(0.108, 0.305)),
}

CACHE = "results"
shap_e = np.load(f"{CACHE}/shap_eicu.npy")
shap_m = np.load(f"{CACHE}/shap_mimic.npy")
X_e = np.load(f"{CACHE}/X_eicu.npy")
X_m = np.load(f"{CACHE}/X_mimic.npy")
features = open(f"{CACHE}/features.txt").read().strip().split("\n")
n_feat = len(features)
mae_e = np.abs(shap_e).mean(axis=0)
mae_m = np.abs(shap_m).mean(axis=0)
combined_imp = (mae_m + mae_e) / 2

uni = pd.read_csv("data/univariate_auroc_mimic_vs_eicu.csv")
uni_flag = dict(zip(uni.feature, uni.flag.fillna("")))


def disp(f):
    return f.replace("_", " ")
feat_disp = [disp(f) for f in features]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# arrays.npz written by analysis_1c.py (non-SMOTE; default ./results_1c).
# Holds S_M, S_ME, S_E, top5, c1_boot, c2_boot, r0_null, R1, R2, features.
ARRAYS_1C = "results_1c/arrays.npz"


# ============================================================================
# Statistical helpers (bootstrap)
# ============================================================================
def stratified_boot_idx(y, n_boot, seed=42):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
            for _ in range(n_boot)]


def boot_auroc_ci(y, p, n_boot=1000, seed=42):
    aucs = np.array([roc_auc_score(y[i], p[i])
                      for i in stratified_boot_idx(y, n_boot, seed)])
    return np.percentile(aucs, [2.5, 97.5])


def boot_auprc_ci(y, p, n_boot=1000, seed=42):
    aps = np.array([average_precision_score(y[i], p[i])
                     for i in stratified_boot_idx(y, n_boot, seed)])
    return np.percentile(aps, [2.5, 97.5])


def boot_band_roc(y, p, n_boot=500, seed=42):
    g = np.linspace(0, 1, 101)
    tprs = np.empty((n_boot, len(g)))
    for i, idx in enumerate(stratified_boot_idx(y, n_boot, seed)):
        f, t, _ = roc_curve(y[idx], p[idx])
        tprs[i] = np.interp(g, f, t); tprs[i, 0] = 0
    return g, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)


def boot_band_pr(y, p, n_boot=500, seed=42):
    g = np.linspace(0, 1, 101)
    pres = np.empty((n_boot, len(g)))
    for i, idx in enumerate(stratified_boot_idx(y, n_boot, seed)):
        pre, rec, _ = precision_recall_curve(y[idx], p[idx])
        order = np.argsort(rec)
        pres[i] = np.interp(g, rec[order], pre[order])
    return g, np.percentile(pres, 2.5, axis=0), np.percentile(pres, 97.5, axis=0)


def kernel_calibration_curve(p, y, p_grid, bw):
    out = np.empty_like(p_grid, dtype=float)
    for i, pg in enumerate(p_grid):
        w = np.exp(-0.5 * ((p - pg) / bw) ** 2); ws = w.sum()
        out[i] = (w * y).sum() / ws if ws > 0 else np.nan
    return out


def boot_calibration_band(y, p, p_grid, bw, n_boot=200, seed=42):
    curves = np.empty((n_boot, len(p_grid)))
    for i, idx in enumerate(stratified_boot_idx(y, n_boot, seed)):
        curves[i] = kernel_calibration_curve(p[idx], y[idx], p_grid, bw)
    return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def calibration_metrics(y, p):
    brier = brier_score_loss(y, p)
    eps = 1e-9
    pp = np.clip(p, eps, 1 - eps)
    logit_p = np.log(pp / (1 - pp))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    lr.fit(logit_p.reshape(-1, 1), y)
    slope = lr.coef_[0, 0]; intercept = lr.intercept_[0]
    bins = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = sum((bi == b).sum() / len(y) * abs(p[bi == b].mean() - y[bi == b].mean())
               for b in range(10) if (bi == b).any())
    return brier, intercept, slope, ece


def net_benefit(y, p, t):
    if t >= 1: return 0.0
    pred = p >= t; n = len(y)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


def boot_dca_band(y, p, thr, n_boot=500, seed=42):
    nbs = np.empty((n_boot, len(thr)))
    for i, idx in enumerate(stratified_boot_idx(y, n_boot, seed)):
        nbs[i] = [net_benefit(y[idx], p[idx], t) for t in thr]
    return np.percentile(nbs, 2.5, axis=0), np.percentile(nbs, 97.5, axis=0)


def panel_label(ax, label, x=-0.16, y=1.05, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold",
             fontsize=fontsize, va="bottom", ha="left", color=C_TEXT)


def light_grid(ax, axis="both"):
    ax.grid(axis=axis, lw=0.4, color="#dcdcdc", alpha=1.0, zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    print(f"  -> {OUT}/{name}.png + .pdf")
    plt.close(fig)


# ============================================================================
# FIG 1 — CONSORT  (clean white-box style with thin colored accent stripes)
# ============================================================================
def make_fig1_consort():
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, accent=None, fontsize=8.8, bold=False,
            highlight=False):
        """Clean white box with thin gray border. accent = color stripe on
        the left side; highlight = subtle gray fill for emphasis."""
        face = "#f5f5f5" if highlight else "white"
        ax.add_patch(Rectangle(
            (x - w/2, y - h/2), w, h,
            facecolor=face, edgecolor="#444444",
            linewidth=0.7, zorder=2))
        if accent:
            # 4-unit-wide colored bar on left edge
            ax.add_patch(Rectangle(
                (x - w/2, y - h/2), 0.7, h,
                facecolor=accent, edgecolor="none", zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                 fontweight="bold" if bold else "normal", zorder=4,
                 color=C_TEXT)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=10, color="#333",
            linewidth=0.7, zorder=1))

    def excl_box(x, y, w, h, text):
        ax.add_patch(Rectangle(
            (x - w/2, y - h/2), w, h,
            facecolor="#fafafa", edgecolor="#999",
            linewidth=0.5, zorder=2, linestyle="--"))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.8,
                 color=C_MUTED, fontstyle="italic", zorder=4)

    # Column headers — colored to match cohort
    ax.text(20, 96, "MIMIC-IV (training)", ha="center", fontsize=11,
             fontweight="bold", color=C_MIMIC)
    ax.text(70, 96, "eICU-CRD (external validation)", ha="center",
             fontsize=11, fontweight="bold", color=C_EICU)

    # MIMIC chain
    box(20, 88, 32, 5.5, "ICU stays screened\nN = 10,202",
        accent=C_MIMIC)
    arrow(20, 84.8, 20, 81)
    excl_box(48, 82, 22, 4.5, "Excluded n = 2,465\nno extubation event")
    arrow(36.8, 82, 30, 82)

    box(20, 76, 32, 5.5,
        "MIMIC training cohort\nN = 7,737  ·  342 events (4.4%)",
        accent=C_MIMIC, bold=True)
    arrow(20, 73, 20, 69.5)

    box(20, 64, 32, 6,
        "5-fold stratified cross-validation\n"
        "XGBoost (n_est=300, depth=5, η=0.05)\n"
        "28 features  ·  no resampling",
        accent=C_MIMIC)
    arrow(20, 60.8, 20, 57.5)

    box(20, 51.5, 32, 7,
        "Pooled out-of-fold predictions\n"
        "AUROC = 0.872 [0.850, 0.892]\n"
        "AUPRC = 0.326  ·  Brier = 0.036",
        accent=C_MIMIC, highlight=True, bold=True)

    arrow(20, 47.7, 20, 43.5)
    box(20, 38.5, 32, 5.5,
        "Final model refit on all 7,737\nv2 bundle deployed",
        accent=C_MIMIC)

    # eICU chain
    box(70, 88, 32, 5.5, "Patient records screened\nN = 16,866",
        accent=C_EICU)
    arrow(70, 84.8, 70, 81)
    excl_box(94, 82, 12, 5, "Excluded\nn = 15,634\nno label")
    arrow(87.6, 82, 80, 82)

    box(70, 76, 32, 5.5,
        "eICU validation cohort (full)\nN = 1,232  ·  160 events (13.0%)",
        accent=C_EICU, bold=True)
    arrow(70, 73, 70, 69.5)

    box(70, 64, 32, 6,
        "v2 preprocessing applied to all 1,232\n"
        "mv_duration_hours ≤ 0  →  NaN  →\n"
        "median imputation (9.0 h)",
        accent=C_EICU)

    arrow(70, 60.8, 60, 56.5)
    arrow(70, 60.8, 80, 56.5)

    box(60, 51.5, 22, 7,
        "Clean subset (mv > 0)\nN = 756\n32 events (4.2%)",
        accent=C_CLEAN, bold=True)
    box(80, 51.5, 18, 7,
        "Imputed subset\n(mv ≤ 0 / NaN)\nN = 476",
        accent="#999")

    arrow(60, 48, 60, 43.5)
    arrow(80, 48, 80, 43.5)

    box(60, 38.5, 22, 6.5,
        "AUROC = 0.815\n[0.732, 0.886]\nAUPRC = 0.175",
        accent=C_CLEAN, highlight=True, fontsize=8.5)
    box(80, 38.5, 18, 6.5,
        "Pooled into\nfull cohort below",
        accent="#999", fontsize=8.4)

    arrow(60, 35, 70, 30)
    arrow(80, 35, 70, 30)

    box(70, 26, 32, 7,
        "Full external cohort\nN = 1,232\n"
        "AUROC = 0.598 [0.545, 0.646] · Brier = 0.117",
        accent=C_EICU, highlight=True, bold=True)

    # Footer note — narrow strip, less heavy
    ax.text(50, 8, "All three cohorts evaluated for ROC, calibration, "
                    "decision-curve and SHAP analyses",
             ha="center", va="center", fontsize=8.5, color=C_MUTED,
             fontstyle="italic")
    ax.text(50, 4.5,
             "Operating thresholds: Sens-80 p = 0.0154  ·  Youden p = 0.0208  "
             "·  default p = 0.5",
             ha="center", va="center", fontsize=8.5, color=C_MUTED)

    save_fig(fig, "fig1_consort")


# ============================================================================
# FIG 2 — Discrimination & calibration
# ============================================================================
def make_fig2_discrim_calib():
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25,
                           left=0.07, right=0.97, top=0.95, bottom=0.07)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr  = fig.add_subplot(gs[0, 1])
    ax_cal = fig.add_subplot(gs[1, 0])
    ax_hst = fig.add_subplot(gs[1, 1])

    # ---- (a) ROC ----
    print("\nROC + AUC bootstrap CIs:")
    for c in COHORTS:
        fpr, tpr, _ = roc_curve(c["y"], c["p"])
        auc = roc_auc_score(c["y"], c["p"])
        boot_lo, boot_hi = boot_auroc_ci(c["y"], c["p"], n_boot=1000)
        lo, hi = TABLE2_CI[c["short"]]["auroc"]
        print(f"  {c['short']}: AUROC={auc:.3f}  label [{lo:.3f}, {hi:.3f}] (Table 2)"
              f"  | this-script bootstrap [{boot_lo:.3f}, {boot_hi:.3f}]")
        fpr_g, tpr_lo, tpr_hi = boot_band_roc(c["y"], c["p"], n_boot=500)
        ax_roc.fill_between(fpr_g, tpr_lo, tpr_hi, color=c["color"],
                             alpha=0.10, linewidth=0)
        ax_roc.plot(fpr, tpr, color=c["color"], lw=1.5,
                     label=f"{c['short']}  AUROC = {auc:.3f} "
                           f"[{lo:.3f}, {hi:.3f}]")
    ax_roc.plot([0, 1], [0, 1], ls=":", color=C_PERFECT, lw=0.7)
    ax_roc.set_xlim(-0.005, 1.005); ax_roc.set_ylim(-0.005, 1.01)
    ax_roc.set_xlabel("False positive rate (1 − specificity)")
    ax_roc.set_ylabel("True positive rate (sensitivity)")
    ax_roc.set_title("ROC curves with 95% CI")
    ax_roc.legend(loc="lower right")
    ax_roc.set_aspect("equal")
    light_grid(ax_roc)
    panel_label(ax_roc, "a", x=-0.13)

    # ---- (b) PR ----
    print("\nPR + AUPRC bootstrap CIs:")
    for c in COHORTS:
        pre, rec, _ = precision_recall_curve(c["y"], c["p"])
        ap = average_precision_score(c["y"], c["p"])
        boot_lo, boot_hi = boot_auprc_ci(c["y"], c["p"], n_boot=1000)
        lo, hi = TABLE2_CI[c["short"]]["auprc"]
        print(f"  {c['short']}: AUPRC={ap:.3f}  label [{lo:.3f}, {hi:.3f}] (Table 2)"
              f"  | this-script bootstrap [{boot_lo:.3f}, {boot_hi:.3f}]")
        rg, pre_lo, pre_hi = boot_band_pr(c["y"], c["p"], n_boot=500)
        ax_pr.fill_between(rg, pre_lo, pre_hi, color=c["color"],
                            alpha=0.10, linewidth=0)
        ax_pr.plot(rec, pre, color=c["color"], lw=1.5,
                    label=f"{c['short']}  AUPRC = {ap:.3f} "
                          f"[{lo:.3f}, {hi:.3f}]")
        ax_pr.axhline(c["y"].mean(), ls=":", lw=0.6, color=c["color"],
                       alpha=0.6)
    ax_pr.set_xlim(-0.005, 1.005); ax_pr.set_ylim(-0.005, 1.01)
    ax_pr.set_xlabel("Recall (sensitivity)")
    ax_pr.set_ylabel("Precision (positive predictive value)")
    ax_pr.set_title("Precision–recall curves with 95% CI")
    ax_pr.legend(loc="upper right")
    ax_pr.set_aspect("equal")
    light_grid(ax_pr)
    panel_label(ax_pr, "b", x=-0.13)

    # ---- (c) Calibration ----
    p_grid = np.linspace(0.005, 0.40, 80)
    print("\nCalibration metrics:")
    for c in COHORTS:
        b, ic, sl, ece = calibration_metrics(c["y"], c["p"])
        print(f"  {c['short']}: Brier={b:.3f}, intercept={ic:+.2f}, "
              f"slope={sl:.2f}, ECE={ece:.3f}")
        bw = 0.04
        smoothed = kernel_calibration_curve(c["p"], c["y"], p_grid, bw)
        lo, hi = boot_calibration_band(c["y"], c["p"], p_grid, bw,
                                         n_boot=200)
        eff_n = np.array([np.exp(-0.5 * ((c["p"] - pg) / bw) ** 2).sum()
                           for pg in p_grid])
        mask = eff_n >= 0.02 * eff_n.max()
        smoothed_m = np.where(mask, smoothed, np.nan)
        lo_m = np.where(mask, lo, np.nan)
        hi_m = np.where(mask, hi, np.nan)
        ax_cal.fill_between(p_grid, lo_m, hi_m, color=c["color"],
                             alpha=0.10, linewidth=0)
        ax_cal.plot(p_grid, smoothed_m, color=c["color"], lw=1.5,
                     label=f"{c['short']}  Brier = {b:.3f}, slope = {sl:.2f}")
    ax_cal.plot([0, 0.40], [0, 0.40], ls=":", color=C_PERFECT, lw=0.7,
                 label="Perfect calibration")
    ax_cal.set_xlim(0, 0.40); ax_cal.set_ylim(0, 0.40)
    ax_cal.set_xlabel("Predicted probability")
    ax_cal.set_ylabel("Observed event rate")
    ax_cal.set_title("Calibration  (Gaussian-kernel smoothed)")
    ax_cal.legend(loc="upper left")
    ax_cal.set_aspect("equal")
    light_grid(ax_cal)
    panel_label(ax_cal, "c", x=-0.13)

    # ---- (d) Predicted-prob distribution ----
    bins = np.linspace(0, 0.40, 50)
    for c in COHORTS:
        ax_hst.hist(c["p"], bins=bins, color=c["color"], alpha=0.45,
                     label=c["short"], edgecolor="none")
    for tval, tname, tcol, ypos in [
        (T_SENS80, f"Sens-80  p = {T_SENS80:.3f}", C_THR_SENS80, 0.93),
        (T_YOUDEN, f"Youden  p = {T_YOUDEN:.3f}",  C_THR_YOUDEN, 0.83),
    ]:
        ax_hst.axvline(tval, color=tcol, lw=0.8, ls="--")
        ax_hst.text(tval + 0.005, ypos, tname, color=tcol, fontsize=7.8,
                     va="top", ha="left",
                     transform=ax_hst.get_xaxis_transform(),
                     bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.92, pad=1.2))
    ax_hst.set_yscale("log")
    ax_hst.set_xlim(0, 0.40)
    ax_hst.set_xlabel("Predicted probability")
    ax_hst.set_ylabel("count (log scale)")
    ax_hst.set_title("Distribution of predicted probabilities")
    ax_hst.legend(loc="upper right")
    light_grid(ax_hst, axis="y")
    panel_label(ax_hst, "d", x=-0.13)

    save_fig(fig, "fig2_discrimination_calibration")


# ============================================================================
# FIG 3 — Decision-curve analysis
# ============================================================================
def make_fig3_dca():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4),
                              gridspec_kw=dict(wspace=0.30,
                                                left=0.06, right=0.97,
                                                top=0.83, bottom=0.16))
    thr_grid = np.linspace(0.005, 0.40, 80)

    print("\nDCA bootstrap (500) per cohort:")
    for ax, c, plabel in zip(axes, COHORTS, "abc"):
        prev = c["y"].mean()
        nb_model = np.array([net_benefit(c["y"], c["p"], t) for t in thr_grid])
        nb_all = np.array([prev - (1 - prev) * t / (1 - t) for t in thr_grid])
        nb_lo, nb_hi = boot_dca_band(c["y"], c["p"], thr_grid, n_boot=500)

        # Reference curves first
        ax.plot(thr_grid, nb_all, color=C_MUTED, lw=0.9, ls="--",
                 label="Treat all")
        ax.axhline(0, color=C_TEXT, lw=0.5, ls=":", label="Treat none")

        # Operating-threshold gridlines (subtle, vertical)
        ax.axvline(T_SENS80, color=C_THR_SENS80, lw=0.7, ls="-", alpha=0.55)
        ax.axvline(T_YOUDEN, color=C_THR_YOUDEN, lw=0.7, ls="-", alpha=0.55)

        # Model with CI band
        ax.fill_between(thr_grid, nb_lo, nb_hi, color=c["color"],
                         alpha=0.13, linewidth=0)
        ax.plot(thr_grid, nb_model, color=c["color"], lw=1.7,
                 label="Model")

        ax.set_xlim(0, 0.40)
        y_lo = -0.025
        y_hi = max(nb_model.max(), prev) * 1.18
        ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel("Threshold probability ($p_t$)")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"{c['name']}\nprevalence = {prev:.3f}")
        light_grid(ax)
        panel_label(ax, plabel, x=-0.20)

    # Single legend along top
    legend_handles = [
        Line2D([], [], color=C_MIMIC, lw=2.0, label="Model (95% CI shaded)"),
        Line2D([], [], color=C_MUTED, lw=0.9, ls="--", label="Treat all"),
        Line2D([], [], color=C_TEXT,  lw=0.6, ls=":",  label="Treat none"),
        Line2D([], [], color=C_THR_SENS80, lw=1.2,
                label=f"Sens-80 (p = {T_SENS80:.3f})"),
        Line2D([], [], color=C_THR_YOUDEN, lw=1.2,
                label=f"Youden  (p = {T_YOUDEN:.3f})"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
                ncol=5, bbox_to_anchor=(0.5, 0.98))
    save_fig(fig, "fig3_dca")


# ============================================================================
# FIG 4 — SHAP global importance & cross-database consistency
# ============================================================================
def make_fig4_shap_global():
    fig = plt.figure(figsize=(13.5, 12.5))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.32,
                           left=0.075, right=0.95,
                           top=0.96, bottom=0.05,
                           height_ratios=[1.15, 1.0])
    ax_bm = fig.add_subplot(gs[0, 0])
    ax_be = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_sl = fig.add_subplot(gs[1, 1])

    TOP = 15
    order = np.argsort(combined_imp)[::-1][:TOP][::-1]

    rng = np.random.RandomState(0)
    for ax, X, shap, title, plabel in [
        (ax_bm, X_m, shap_m, "MIMIC-IV (n = 1,500)", "a"),
        (ax_be, X_e, shap_e, "eICU (n = 1,232)", "b"),
    ]:
        for yi, j in enumerate(order):
            sj = shap[:, j]; xj = X[:, j]
            lo, hi = np.percentile(xj, [5, 95])
            cj = np.clip((xj - lo) / (hi - lo), 0, 1) if hi > lo \
                  else np.zeros_like(xj)
            jit = (rng.rand(len(sj)) - 0.5) * 0.65
            ax.scatter(sj, np.full_like(sj, yi) + jit, c=cj,
                       cmap=SHAP_CMAP, s=4.5, alpha=0.6,
                       edgecolors="none", vmin=0, vmax=1, rasterized=True)
        ax.axvline(0, color=C_MUTED, lw=0.5, ls="--", alpha=0.7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_disp[j] for j in order])
        ax.set_xlabel("SHAP value (logit space)")
        ax.set_title(title)
        light_grid(ax, axis="x")
        panel_label(ax, plabel, x=-0.32)

    # Shared, smaller colorbar (right of panel b)
    cbar_ax = fig.add_axes([0.965, 0.62, 0.011, 0.28])
    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbar_ax, ticks=[0, 0.5, 1])
    cb.set_ticklabels(["low", "med", "high"])
    cb.set_label("Feature value", rotation=270, labelpad=15, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.5, length=2, labelsize=7.5)

    # ---- (c) cross-DB scatter ----
    rho_imp, _ = spearmanr(mae_m, mae_e)
    rho_log, _ = pearsonr(np.log(mae_m + 1e-6), np.log(mae_e + 1e-6))
    flipped = {f for f in features if uni_flag.get(f, "") == "FLIPPED"}

    # Refined marker style: filled dots, white edge, smaller
    for j in range(n_feat):
        col = C_EICU if features[j] in flipped else C_MIMIC
        ax_sc.scatter(mae_m[j], mae_e[j], s=42, color=col,
                       edgecolors="white", linewidths=0.7, alpha=0.9,
                       zorder=3)
    lims = [min(mae_m.min(), mae_e.min()) * 0.7,
             max(mae_m.max(), mae_e.max()) * 1.4]
    ax_sc.plot(lims, lims, ls=":", color=C_PERFECT, lw=0.7, zorder=1)
    ax_sc.set_xscale("log"); ax_sc.set_yscale("log")
    ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)

    # Collision-avoiding label placement
    top_idx = set(np.argsort(combined_imp)[::-1][:8].tolist())
    flipped_idx = {j for j in range(n_feat) if features[j] in flipped}
    label_idx = sorted(top_idx | flipped_idx,
                        key=lambda j: -combined_imp[j])

    fig.canvas.draw()
    candidate_offsets = [
        (8, 4), (8, -10), (-12, 8), (-12, -12),
        (16, 14), (16, -18), (-22, 14), (-22, -20),
        (8, 22), (8, -26), (28, 4), (-36, 4),
    ]
    placed_bboxes = []
    renderer = fig.canvas.get_renderer()
    for j in label_idx:
        x, y = mae_m[j], mae_e[j]
        for dx, dy in candidate_offsets:
            t = ax_sc.annotate(feat_disp[j], (x, y),
                                xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=7.8, color=C_TEXT, zorder=5,
                                ha="left" if dx >= 0 else "right",
                                va="bottom" if dy >= 0 else "top")
            bbox = t.get_window_extent(renderer=renderer).expanded(1.05, 1.10)
            if not any(bbox.overlaps(b) for b in placed_bboxes):
                placed_bboxes.append(bbox)
                break
            t.remove()
        else:
            t = ax_sc.annotate(feat_disp[j], (x, y),
                                xytext=candidate_offsets[0],
                                textcoords="offset points",
                                fontsize=7.8, color=C_TEXT, zorder=5)
            placed_bboxes.append(t.get_window_extent(renderer=renderer))

    ax_sc.set_xlabel("mean | SHAP |  in MIMIC-IV  (log scale)")
    ax_sc.set_ylabel("mean | SHAP |  in eICU  (log scale)")
    ax_sc.set_title("Cross-database importance agreement\n"
                     f"Spearman ρ = {rho_imp:.2f}  ·  "
                     f"Pearson(log) r = {rho_log:.2f}")
    ax_sc.legend(handles=[
        Patch(facecolor=C_MIMIC, label="Univariate AUROC same direction"),
        Patch(facecolor=C_EICU,  label="Univariate AUROC FLIPPED"),
    ], loc="lower right")
    light_grid(ax_sc, axis="both")
    panel_label(ax_sc, "c", x=-0.16)

    # ---- (d) slopegraph ----
    rank_m = (-mae_m).argsort().argsort()
    rank_e = (-mae_e).argsort().argsort()
    rank_rho, _ = spearmanr(rank_m, rank_e)
    y_m_rank = -rank_m; y_e_rank = -rank_e
    xL, xR = 0, 1

    for j in range(n_feat):
        is_f = features[j] in flipped
        col = C_EICU if is_f else C_MIMIC
        ax_sl.plot([xL, xR], [y_m_rank[j], y_e_rank[j]],
                    color=col, lw=0.9 if is_f else 0.6,
                    alpha=0.85 if is_f else 0.30,
                    zorder=4 if is_f else 2)
        ax_sl.scatter([xL, xR], [y_m_rank[j], y_e_rank[j]],
                       color=col, s=14, edgecolors="white", linewidths=0.4,
                       zorder=5)
        text_color = C_EICU if is_f else C_TEXT
        weight = "bold" if is_f else "normal"
        ax_sl.annotate(feat_disp[j], (xL - 0.025, y_m_rank[j]),
                        ha="right", va="center", fontsize=7.5,
                        color=text_color, fontweight=weight)
        ax_sl.annotate(feat_disp[j], (xR + 0.025, y_e_rank[j]),
                        ha="left", va="center", fontsize=7.5,
                        color=text_color, fontweight=weight)

    ax_sl.set_xticks([xL, xR])
    ax_sl.set_xticklabels(["MIMIC-IV rank", "eICU rank"], fontsize=9)
    ax_sl.set_xlim(-0.45, 1.45)
    ax_sl.set_yticks([])
    ax_sl.spines["left"].set_visible(False)
    ax_sl.spines["bottom"].set_visible(False)
    ax_sl.tick_params(axis="x", length=0)
    ax_sl.set_title(f"Per-feature rank movement\n"
                     f"Spearman of ranks = {rank_rho:.2f}")
    panel_label(ax_sl, "d", x=-0.06)

    save_fig(fig, "fig4_shap_global_consistency")


# ============================================================================
# FIG 5 — SHAP dependence plots
# ============================================================================
def make_fig5_shap_dependence():
    top6 = np.argsort(combined_imp)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                              gridspec_kw=dict(wspace=0.28, hspace=0.45,
                                                left=0.06, right=0.98,
                                                top=0.91, bottom=0.07))

    for ax, j, plabel in zip(axes.ravel(), top6, "abcdef"):
        f = features[j]
        is_binary = (np.unique(X_m[:, j]).size <= 3 and
                     set(np.unique(X_m[:, j])).issubset({0.0, 1.0}))

        if is_binary:
            for cohort_label, X, shap, color in [
                ("MIMIC-IV", X_m, shap_m, C_MIMIC),
                ("eICU",     X_e, shap_e, C_EICU),
            ]:
                for v in [0.0, 1.0]:
                    mask = X[:, j] == v
                    if not mask.any(): continue
                    rng = np.random.RandomState(int(v) +
                            (0 if cohort_label == "MIMIC-IV" else 100))
                    jit = (rng.rand(mask.sum()) - 0.5) * 0.28
                    x_pos = (v + jit
                              + (-0.16 if cohort_label == "MIMIC-IV"
                                 else 0.16))
                    ax.scatter(x_pos, shap[mask, j], s=3.5, alpha=0.30,
                               color=color, edgecolors="none",
                               rasterized=True)
                    ax.hlines(np.median(shap[mask, j]),
                               v + (-0.30 if cohort_label == "MIMIC-IV" else 0.0),
                               v + (0.0 if cohort_label == "MIMIC-IV" else 0.30),
                               color=color, lw=1.6, alpha=0.95)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["no (0)", "yes (1)"])
            ax.set_xlim(-0.55, 1.55)
        else:
            xall = np.concatenate([X_m[:, j], X_e[:, j]])
            lo, hi = np.percentile(xall, [1, 99])
            x_grid = np.linspace(lo, hi, 80)
            for cohort_label, X, shap, color in [
                ("MIMIC-IV", X_m, shap_m, C_MIMIC),
                ("eICU",     X_e, shap_e, C_EICU),
            ]:
                ax.scatter(X[:, j], shap[:, j], s=4, alpha=0.25,
                           color=color, edgecolors="none",
                           rasterized=True)
                bw = (hi - lo) * 0.05
                pdp = np.array([
                    (np.exp(-0.5 * ((X[:, j] - xv) / bw) ** 2) * shap[:, j]).sum() /
                    np.exp(-0.5 * ((X[:, j] - xv) / bw) ** 2).sum()
                    for xv in x_grid
                ])
                ax.plot(x_grid, pdp, color=color, lw=1.8, alpha=0.95)
            ax.set_xlim(lo, hi)
        ax.axhline(0, color=C_MUTED, lw=0.5, ls="--", alpha=0.7)
        ax.set_xlabel(disp(f))
        ax.set_ylabel("SHAP value (logit)")
        flag = uni_flag.get(f, "")
        if flag == "FLIPPED":
            ax.set_title(f"{disp(f)}\n[univariate FLIPPED]", fontsize=9.5)
        else:
            ax.set_title(disp(f))
        light_grid(ax)
        panel_label(ax, plabel, x=-0.18, fontsize=11)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=C_MIMIC,
                markersize=6, alpha=0.7, label="MIMIC-IV  (n = 1,500)"),
        Line2D([], [], marker="o", linestyle="", color=C_EICU,
                markersize=6, alpha=0.7, label="eICU  (n = 1,232)"),
        Line2D([], [], color=C_MUTED, lw=2,
                label="kernel-smoothed partial dependence"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 0.99))
    save_fig(fig, "fig5_shap_dependence")


# ============================================================================
# Figure 6 — reference-anchored cross-database SHAP comparison (non-SMOTE reframe)
# Summary correlations + CIs are fixed analysis results (locked), so this panel
# is drawn from values rather than re-derived here.
# ============================================================================
def make_fig6_reference_anchored():
    C_FROZEN, C_INDEP, C_ART = C_MIMIC, "#C2741C", "#9a9a9a"
    # label, y, point, lo, hi, group, marker
    rows = [
        ("R1  frozen model, within-MIMIC halves\n(structural ceiling for C1)",          9.0, 1.00, 0.99, 1.00, "frozen", "D"),
        ("C1  frozen model, MIMIC vs eICU\n(cross-database, covariate shift)",           8.0, 0.90, 0.87, 0.91, "frozen", "o"),
        ("within-eICU reproducibility\n(two independent eICU models)",                   6.3, 0.81, 0.66, 0.87, "indep",  "s"),
        ("within-MIMIC reproducibility\n(deployment config)",                            5.3, 0.46, 0.20, 0.61, "indep",  "D"),
        ("C2  independent eICU vs MIMIC model\n(same evaluation data)  [PRIMARY]",        4.3, 0.48, 0.32, 0.57, "indep",  "o"),
        ("C2 sens.:  eICU model, MIMIC hyperparameters",                                 2.7, 0.47, None, None, "sens",   "o"),
        ("C2 sens.:  mv_duration removed",                                               1.7, 0.33, None, None, "sens",   "o"),
        ("within-MIMIC @160 ev, small regularised model\n(configuration artefact — not the reproducibility ceiling)", 0.2, 0.27, 0.06, 0.55, "art", "D"),
    ]
    sty = {
        "frozen": dict(color=C_FROZEN, mfc=C_FROZEN, ms=8.5),
        "indep":  dict(color=C_INDEP,  mfc=C_INDEP,  ms=8.5),
        "sens":   dict(color=C_INDEP,  mfc="white",  ms=8.0),
        "art":    dict(color=C_ART,    mfc="white",  ms=7.5),
    }
    fig, ax = plt.subplots(figsize=(9.6, 6.3))
    ax.axvspan(-0.32, 0.44, color="#eef0f2", zorder=0)
    ax.axvline(0.0, ls=(0, (4, 3)), color="#999", lw=0.8, zorder=1)
    ax.axvline(0.44, ls=":", color="#b0b0b0", lw=0.8, zorder=1)
    ax.axvline(0.61, ls=(0, (5, 2)), color=C_INDEP, lw=1.1, alpha=0.75, zorder=1)
    ax.text(0.61, 7.15, "attenuation ceiling for C2\n√(0.46 × 0.81) ≈ 0.61", ha="center",
            va="center", fontsize=7.0, color=C_INDEP, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_INDEP, lw=0.6), zorder=6)
    for lab, y, pt, lo, hi, grp, mk in rows:
        s = sty[grp]
        if lo is not None:
            ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="none",
                        ecolor=s["color"], elinewidth=1.4, capsize=3.0, zorder=3)
        ax.plot(pt, y, marker=mk, ms=s["ms"], mfc=s["mfc"], mec=s["color"],
                mew=1.5, linestyle="none", zorder=4)
        ax.text(pt, y + 0.30, f"{pt:.2f}", ha="center", va="bottom", fontsize=7.6,
                color=s["color"], fontweight="bold", zorder=5)
    ax.text(0.57 + 0.03, 4.3, "permutation p = 0.006", va="center", ha="left",
            fontsize=7.6, style="italic", color=C_INDEP)
    ax.text(0.06, 9.55, "chance region  (ρ ≤ permutation-null 99th pct = 0.44)",
            ha="left", va="center", fontsize=7.8, color=C_MUTED, style="italic")
    ax.set_yticks([r[1] for r in rows]); ax.set_yticklabels([r[0] for r in rows], fontsize=8.0)
    ax.set_ylim(-0.5, 10.0); ax.set_xlim(-0.32, 1.04)
    ax.set_xlabel("Spearman rank correlation of mean |SHAP| feature importance  (ρ)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", color="w", mfc=C_FROZEN, mec=C_FROZEN, ms=8.5,
               label="Frozen-model comparison / its structural ceiling"),
        Line2D([0], [0], marker="o", color="w", mfc=C_INDEP, mec=C_INDEP, ms=8.5,
               label="Independent-model comparison / its reproducibility ceiling"),
        Line2D([0], [0], marker="o", color="w", mfc="white", mec=C_INDEP, mew=1.5, ms=8.0,
               label="C2 sensitivity analysis (point estimate)"),
        Line2D([0], [0], marker="D", color="w", mfc="white", mec=C_ART, mew=1.4, ms=7.5,
               label="Pre-specified conservative variant (down-sampled; artefactual)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=7.8, handletextpad=0.4, columnspacing=1.4)
    save_fig(fig, "fig6_reference_anchored")


# ============================================================================
# Figure 7 — C2 slopegraph (same eICU evaluation, two training sources)
# Reads S_ME / S_E from analysis_1c.py's arrays.npz (non-SMOTE).
# ============================================================================
def _load_1c():
    try:
        return np.load(ARRAYS_1C, allow_pickle=True)
    except FileNotFoundError:
        print(f"  [skip] {ARRAYS_1C} not found — run analysis_1c.py first.")
        return None


def make_fig7_c2_slopegraph():
    z = _load_1c()
    if z is None:
        return
    S_ME = z["S_ME"].astype(float); S_E = z["S_E"].astype(float)
    names = [disp(f) for f in z["features"].tolist()]
    n = len(names)

    def rank_desc(v):
        r = np.empty(n, int); r[np.argsort(-v)] = np.arange(1, n + 1); return r

    a, b = rank_desc(S_ME), rank_desc(S_E)
    rho = spearmanr(S_ME, S_E).correlation

    fig, ax = plt.subplots(figsize=(7.0, 8.4))
    for i in range(n):
        ax.plot([0, 1], [a[i], b[i]], "-", color="0.86", lw=0.7, zorder=1)
    cmap = plt.get_cmap("tab10")
    top = np.argsort(-S_ME)[:8]
    for k, i in enumerate(top):
        c = cmap(k % 10)
        ax.plot([0, 1], [a[i], b[i]], "-", lw=1.9, color=c, zorder=3)
        ax.scatter([0, 1], [a[i], b[i]], s=18, color=c, zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.text(-0.05, a[i], names[i], ha="right", va="center", fontsize=7.6,
                color=c, fontweight="bold")
        ax.text(1.05, b[i], names[i], ha="left", va="center", fontsize=7.6,
                color=c, fontweight="bold")
    ax.scatter([0] * n, a, s=11, color="0.4", zorder=4)
    ax.scatter([1] * n, b, s=11, color="0.4", zorder=4)
    ax.set_ylim(n + 0.6, 0.4); ax.set_xlim(-0.78, 1.78)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["MIMIC-trained", "eICU-trained"], fontsize=9.5)
    ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("C2: same eICU evaluation, two training sources\n"
                 f"Spearman ρ = {rho:.2f}")
    save_fig(fig, "fig7_c2_slopegraph")


# ============================================================================
# Figure 8 — eICU-model top-5 feature stability across bootstrap resamples
# ============================================================================
def make_fig8_eicu_top5_stability():
    z = _load_1c()
    if z is None:
        return
    freq = z["top5"].astype(float); names = [disp(f) for f in z["features"].tolist()]
    order = np.argsort(-freq)[:12][::-1]
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    ax.barh(range(len(order)), freq[order], color=C_EICU, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order])
    for k, i in enumerate(order):
        ax.text(freq[i] + 0.015, k, f"{freq[i]:.2f}", va="center",
                fontsize=7.5, color=C_TEXT)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("Frequency in eICU-model top-5 across bootstrap resamples")
    light_grid(ax, axis="x")
    save_fig(fig, "fig8_eicu_top5_stability")


# ============================================================================
# Figure S1 — SHAP dependence for two additional features (sodium_max, last_fio2)
# Same kernel-smoothed PDP style as Fig 5; reads the non-SMOTE SHAP arrays.
# ============================================================================
def make_figS1_dependence():
    want = ["sodium_max", "last_fio2"]
    idxs = [features.index(f) for f in want if f in features]
    if not idxs:
        print("  [skip] FigS1 features not found in features.txt."); return
    fig, axes = plt.subplots(1, len(idxs), figsize=(10, 4.3),
                             gridspec_kw=dict(wspace=0.24, left=0.08, right=0.97,
                                              top=0.86, bottom=0.15))
    axes = np.ravel(axes)
    for ax, j, plabel in zip(axes, idxs, "abcdef"):
        f = features[j]
        xall = np.concatenate([X_m[:, j], X_e[:, j]])
        lo, hi = np.percentile(xall, [1, 99]); x_grid = np.linspace(lo, hi, 80)
        for X, shap, color in [(X_m, shap_m, C_MIMIC), (X_e, shap_e, C_EICU)]:
            ax.scatter(X[:, j], shap[:, j], s=4, alpha=0.25, color=color,
                       edgecolors="none", rasterized=True)
            bw = (hi - lo) * 0.05
            pdp = np.array([
                (np.exp(-0.5 * ((X[:, j] - xv) / bw) ** 2) * shap[:, j]).sum() /
                np.exp(-0.5 * ((X[:, j] - xv) / bw) ** 2).sum() for xv in x_grid])
            ax.plot(x_grid, pdp, color=color, lw=1.8, alpha=0.95)
        ax.set_xlim(lo, hi); ax.axhline(0, color=C_MUTED, lw=0.5, ls="--", alpha=0.7)
        ax.set_xlabel(disp(f)); ax.set_ylabel("SHAP value (logit contribution)")
        ax.set_title(disp(f)); light_grid(ax)
        panel_label(ax, plabel, x=-0.16, fontsize=11)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=C_MIMIC, markersize=6, alpha=0.7, label="MIMIC-IV"),
        Line2D([], [], marker="o", linestyle="", color=C_EICU, markersize=6, alpha=0.7, label="eICU"),
        Line2D([], [], color=C_MUTED, lw=2, label="kernel-smoothed partial dependence"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    save_fig(fig, "figS1_shap_dependence")


# ============================================================================
# Performance metrics summary table
# ============================================================================
def save_metric_table():
    rows = []
    for c in COHORTS:
        b, ic, sl, ece = calibration_metrics(c["y"], c["p"])
        auc_lo, auc_hi = boot_auroc_ci(c["y"], c["p"], n_boot=1000)
        ap_lo, ap_hi = boot_auprc_ci(c["y"], c["p"], n_boot=1000)
        rows.append({
            "cohort": c["name"], "n": len(c["y"]),
            "events": int(c["y"].sum()),
            "prevalence": round(c["y"].mean(), 4),
            "AUROC": round(roc_auc_score(c["y"], c["p"]), 4),
            "AUROC_95CI_low":  round(auc_lo, 4),
            "AUROC_95CI_high": round(auc_hi, 4),
            "AUPRC": round(average_precision_score(c["y"], c["p"]), 4),
            "AUPRC_95CI_low":  round(ap_lo, 4),
            "AUPRC_95CI_high": round(ap_hi, 4),
            "Brier": round(b, 4),
            "calib_intercept": round(ic, 4),
            "calib_slope":     round(sl, 4),
            "ECE": round(ece, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/performance_metrics.csv", index=False)
    print("\n  -> results/performance_metrics.csv")


if __name__ == "__main__":
    print("Generating academic-style figures...")
    make_fig1_consort()
    make_fig2_discrim_calib()
    make_fig3_dca()
    make_fig4_shap_global()
    make_fig5_shap_dependence()
    make_fig6_reference_anchored()
    make_fig7_c2_slopegraph()
    make_fig8_eicu_top5_stability()
    make_figS1_dependence()
    save_metric_table()
    print("\nDone.")
