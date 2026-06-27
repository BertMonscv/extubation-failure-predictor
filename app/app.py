"""
Streamlit app — Extubation Failure Risk Predictor
==================================================
MIMIC-IV trained, eICU externally validated XGBoost model with
SHAP-based individual-patient explanations.

Run locally:        streamlit run app.py
Deploy:             push this repo to GitHub, then connect the repo on
                    https://share.streamlit.io/  (Streamlit Community Cloud)
"""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import streamlit.components.v1 as components

# Heavy ML imports inside the app — kept here so a missing dep raises early
import shap
import xgboost  # noqa: F401  (needed for unpickling the bundle)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
APP_TITLE = "Extubation Failure Risk Predictor"
APP_SUB = ("XGBoost (n=300, depth=5) · trained on MIMIC-IV · "
           "externally validated on eICU-CRD · with SHAP individual explanations")

# Default location for the bundled model file (relative to repo root).
# Streamlit Cloud will see this if you commit the file alongside app.py.
# Try, in order: alongside app.py (Streamlit Cloud), the parent repo's
# models/ directory (local repo clone), then the current working dir.
_CANDIDATE_MODEL_PATHS = [
    Path("xgb_extubation_failure_v2.pkl"),
    Path(__file__).resolve().parent / "xgb_extubation_failure_v2.pkl",
    Path(__file__).resolve().parent.parent / "models" / "xgb_extubation_failure_v2.pkl",
]
DEFAULT_MODEL_PATH = next((p for p in _CANDIDATE_MODEL_PATHS if p.exists()),
                          _CANDIDATE_MODEL_PATHS[0])

# Operating thresholds shipped in the model bundle (for reference / fallback)
DEFAULT_THRESHOLDS = {
    "Sens-80 (high-sensitivity, screening)": 0.0317,
    "Youden (balanced)": 0.0491,
    "Default 0.5 (high-specificity, rarely useful at 4% prevalence)": 0.5,
}

# Per-feature input spec:
#   label, default (used as fallback if imputer doesn't have it), unit,
#   min, max, step, help text
# `default` here is overridden by the bundle's imputer.statistics_ at runtime.
FEATURE_SPECS: dict[str, dict] = {
    # — Mechanical ventilation —
    "mv_duration_hours":   dict(label="MV duration",            unit="hours",      vmin=0.0,   vmax=300.0, step=0.5,
                                  group="Mechanical ventilation",
                                  help="Cumulative invasive ventilation time before extubation"),
    "last_fio2":           dict(label="Last FiO₂",              unit="%",          vmin=21.0,  vmax=100.0, step=1.0,
                                  group="Mechanical ventilation",
                                  help="FiO₂ at the last ventilator setting before extubation (percent)"),
    "spo2_mean":           dict(label="SpO₂ (mean, day 1)",     unit="%",          vmin=70.0,  vmax=100.0, step=0.1,
                                  group="Mechanical ventilation"),
    "resp_rate_mean":      dict(label="Resp rate (mean)",       unit="breaths/min", vmin=5.0,  vmax=50.0,  step=0.1,
                                  group="Mechanical ventilation"),

    # — Comorbidities (binary) —
    "received_rrt":            dict(label="Renal replacement therapy",     binary=True,
                                     group="Comorbidities & treatments"),
    "congestive_heart_failure": dict(label="Congestive heart failure",      binary=True,
                                     group="Comorbidities & treatments"),
    "cerebrovascular_disease":  dict(label="Cerebrovascular disease",       binary=True,
                                     group="Comorbidities & treatments"),

    # — Vitals —
    "sbp_mean":            dict(label="SBP (mean, day 1)",      unit="mmHg",       vmin=50.0,  vmax=220.0, step=1.0,
                                  group="Vital signs"),
    "sbp_max":             dict(label="SBP (max, day 1)",       unit="mmHg",       vmin=60.0,  vmax=260.0, step=1.0,
                                  group="Vital signs"),

    # — Arterial blood gas / oxygenation —
    "po2_min":             dict(label="PO₂ (min)",              unit="mmHg",       vmin=20.0,  vmax=300.0, step=1.0,
                                  group="Arterial blood gas"),
    "po2_max":             dict(label="PO₂ (max)",              unit="mmHg",       vmin=20.0,  vmax=600.0, step=1.0,
                                  group="Arterial blood gas"),
    "pco2_max":            dict(label="PCO₂ (max)",             unit="mmHg",       vmin=20.0,  vmax=120.0, step=0.5,
                                  group="Arterial blood gas"),
    "ph_min":              dict(label="pH (min)",               unit="",           vmin=6.80,  vmax=7.80,  step=0.01,
                                  group="Arterial blood gas"),
    "lactate_max":         dict(label="Lactate (max)",          unit="mmol/L",     vmin=0.3,   vmax=20.0,  step=0.1,
                                  group="Arterial blood gas"),

    # — Hematology / coagulation —
    "hemoglobin_min":      dict(label="Hemoglobin (min)",       unit="g/dL",       vmin=4.0,   vmax=20.0,  step=0.1,
                                  group="Hematology & coagulation"),
    "platelets_max":       dict(label="Platelets (max)",        unit="×10⁹/L",     vmin=10.0,  vmax=1000.0,step=1.0,
                                  group="Hematology & coagulation"),
    "ptt_min":             dict(label="PTT (min)",              unit="sec",        vmin=15.0,  vmax=200.0, step=0.5,
                                  group="Hematology & coagulation"),
    "ptt_max":             dict(label="PTT (max)",              unit="sec",        vmin=15.0,  vmax=200.0, step=0.5,
                                  group="Hematology & coagulation"),
    "pt_max":              dict(label="PT (max)",               unit="sec",        vmin=10.0,  vmax=80.0,  step=0.1,
                                  group="Hematology & coagulation"),
    "inr_max":             dict(label="INR (max)",              unit="",           vmin=0.8,   vmax=10.0,  step=0.05,
                                  group="Hematology & coagulation"),
    "fibrinogen_min":      dict(label="Fibrinogen (min)",       unit="mg/dL",      vmin=50.0,  vmax=1000.0,step=1.0,
                                  group="Hematology & coagulation"),

    # — Chemistry / electrolytes / renal —
    "sodium_max":          dict(label="Sodium (max)",           unit="mmol/L",     vmin=120.0, vmax=170.0, step=1.0,
                                  group="Chemistry & renal"),
    "potassium_min":       dict(label="Potassium (min)",        unit="mmol/L",     vmin=2.0,   vmax=7.0,   step=0.1,
                                  group="Chemistry & renal"),
    "bicarbonate_min":     dict(label="Bicarbonate (min)",      unit="mmol/L",     vmin=5.0,   vmax=45.0,  step=0.5,
                                  group="Chemistry & renal"),
    "bun_max":             dict(label="BUN (max)",              unit="mg/dL",      vmin=2.0,   vmax=200.0, step=1.0,
                                  group="Chemistry & renal"),
    "creatinine_min":      dict(label="Creatinine (min)",       unit="mg/dL",      vmin=0.2,   vmax=15.0,  step=0.05,
                                  group="Chemistry & renal"),
    "aniongap_min":        dict(label="Anion gap (min)",        unit="mmol/L",     vmin=0.0,   vmax=40.0,  step=0.5,
                                  group="Chemistry & renal"),
    "aniongap_max":        dict(label="Anion gap (max)",        unit="mmol/L",     vmin=0.0,   vmax=40.0,  step=0.5,
                                  group="Chemistry & renal"),
}

GROUP_ORDER = [
    "Mechanical ventilation",
    "Comorbidities & treatments",
    "Vital signs",
    "Arterial blood gas",
    "Hematology & coagulation",
    "Chemistry & renal",
]


# ----------------------------------------------------------------------------
# Streamlit page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model bundle…")
def load_bundle_from_path(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading model bundle…")
def load_bundle_from_bytes(file_bytes: bytes) -> dict:
    import io
    return pickle.load(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner="Building SHAP explainer…")
def get_explainer(_model):
    """TreeExplainer is built once. The leading underscore tells Streamlit
    not to hash the un-hashable XGBoost object."""
    return shap.TreeExplainer(_model)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def imputer_defaults(bundle: dict) -> dict[str, float]:
    """Pull median values from the bundle's SimpleImputer."""
    feats = bundle["features"]
    stats = np.asarray(bundle["imputer"].statistics_, dtype=float)
    return dict(zip(feats, stats.tolist()))


def build_input_form(bundle: dict) -> dict[str, float]:
    """Render the input form, return a {feature: value} dict."""
    defaults = imputer_defaults(bundle)
    values: dict[str, float] = {}

    # Build groups
    grouped: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for feat, spec in FEATURE_SPECS.items():
        grouped[spec["group"]].append(feat)
    # Surface any feature not in our spec (defensive)
    extra = [f for f in bundle["features"] if f not in FEATURE_SPECS]
    if extra:
        grouped.setdefault("Other", []).extend(extra)

    for group in [*GROUP_ORDER, "Other"]:
        feats_in_group = grouped.get(group, [])
        if not feats_in_group:
            continue
        with st.expander(f"**{group}**", expanded=(group == "Mechanical ventilation"
                                                    or group == "Comorbidities & treatments")):
            cols = st.columns(2)
            for i, feat in enumerate(feats_in_group):
                spec = FEATURE_SPECS.get(feat, {})
                col = cols[i % 2]
                with col:
                    if spec.get("binary"):
                        v = st.radio(
                            spec.get("label", feat),
                            options=[0, 1],
                            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                            index=int(round(defaults.get(feat, 0))),
                            horizontal=True,
                            key=f"in_{feat}",
                        )
                    else:
                        unit_suffix = (f"  ({spec['unit']})" if spec.get("unit") else "")
                        v = st.number_input(
                            spec.get("label", feat) + unit_suffix,
                            min_value=float(spec.get("vmin", 0.0)),
                            max_value=float(spec.get("vmax", 1e6)),
                            value=float(defaults.get(feat, spec.get("default", 0.0))),
                            step=float(spec.get("step", 1.0)),
                            help=spec.get("help"),
                            key=f"in_{feat}",
                        )
                    values[feat] = float(v)
    return values


def st_shap_html(plot, height: int = 200):
    """Render an interactive (D3-based) SHAP visualization in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


def show_force_plot(explainer, shap_values_row, x_row, feature_names,
                    feature_display_values):
    """Try interactive HTML; fall back to matplotlib if needed."""
    try:
        force = shap.force_plot(
            base_value=float(explainer.expected_value),
            shap_values=shap_values_row,
            features=feature_display_values,
            feature_names=feature_names,
        )
        st_shap_html(force, height=170)
    except Exception as e:
        st.caption(f"Interactive force plot unavailable ({e}); "
                   "falling back to static rendering.")
        fig = plt.figure(figsize=(11, 2.5))
        shap.force_plot(
            base_value=float(explainer.expected_value),
            shap_values=shap_values_row,
            features=feature_display_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        st.pyplot(plt.gcf(), clear_figure=True)


def show_waterfall_plot(explainer, shap_values_row, x_row, feature_names):
    explanation = shap.Explanation(
        values=shap_values_row,
        base_values=float(explainer.expected_value),
        data=x_row,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=14, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # ---- Sidebar ----
    with st.sidebar:
        st.title("⚙️ Settings")

        # Model loader
        st.subheader("Model")
        bundle = None
        if DEFAULT_MODEL_PATH.exists():
            try:
                bundle = load_bundle_from_path(str(DEFAULT_MODEL_PATH))
                st.success(f"Loaded **{DEFAULT_MODEL_PATH.name}**")
            except Exception as e:
                st.error(f"Failed to load default model: {e}")
        if bundle is None:
            uploaded = st.file_uploader(
                "Upload `.pkl` model bundle",
                type=["pkl"],
                help="The v2 bundle: dict with keys `model`, `imputer`, "
                     "`features`, `thresholds`.",
            )
            if uploaded is None:
                st.info("Place `xgb_extubation_failure_v2.pkl` next to "
                         "`app.py` or upload it here.")
                st.stop()
            try:
                bundle = load_bundle_from_bytes(uploaded.getvalue())
                st.success(f"Loaded **{uploaded.name}**")
            except Exception as e:
                st.error(f"Could not unpickle: {e}")
                st.stop()

        # Threshold selector
        st.subheader("Decision threshold")
        thresholds_in_bundle = bundle.get("thresholds", {}) or {}
        # Compose the menu from the bundle's thresholds (fallback to defaults)
        choices = {}
        if "sens80" in thresholds_in_bundle:
            choices[f"Sens-80  (p = {thresholds_in_bundle['sens80']:.3f})"] = \
                thresholds_in_bundle["sens80"]
        if "youden" in thresholds_in_bundle:
            choices[f"Youden  (p = {thresholds_in_bundle['youden']:.3f})"] = \
                thresholds_in_bundle["youden"]
        if "default" in thresholds_in_bundle:
            choices[f"Default  (p = {thresholds_in_bundle['default']:.2f})"] = \
                thresholds_in_bundle["default"]
        if not choices:
            choices = DEFAULT_THRESHOLDS
        choice = st.selectbox("Operating point", list(choices.keys()), index=1)
        chosen_threshold = float(choices[choice])
        st.caption(f"Selected p = **{chosen_threshold:.4f}**")

        # Model meta
        st.subheader("Model info")
        meta = bundle.get("metadata", {}) or {}
        st.markdown(
            f"- Version: `{meta.get('version', 'unknown')}`\n"
            f"- Trained on: {meta.get('trained_on', 'MIMIC-IV')}\n"
            f"- N train: {meta.get('n_train', 'n/a')}\n"
            f"- Positive rate: {meta.get('positive_rate', 'n/a')}\n"
            f"- Features: {len(bundle['features'])}"
        )
        with st.expander("Validation performance"):
            st.markdown(
                "- **MIMIC-IV internal CV (OOF)**  AUROC 0.862 [0.841, 0.881], "
                "AUPRC 0.316, Brier 0.037\n"
                "- **eICU clean (mv > 0, n=756)**  AUROC 0.766 [0.687, 0.836]\n"
                "- **eICU full (n=1,232)**  AUROC 0.615 [0.565, 0.663]"
            )

        st.divider()
        st.caption(
            "⚠️ **Research / educational use only.** Predictions are "
            "model-based estimates; clinical decisions must integrate the "
            "full clinical picture and the calibration drift observed on "
            "the eICU full cohort."
        )

    # ---- Main panel ----
    st.title(f"🫁 {APP_TITLE}")
    st.caption(APP_SUB)

    st.subheader("Patient inputs")
    st.caption("Defaults are the training-cohort medians; adjust the "
               "values you have measured. Click **Predict and explain** "
               "to score the patient — results stay visible while you edit.")

    with st.form("patient_inputs", clear_on_submit=False):
        values = build_input_form(bundle)
        submitted = st.form_submit_button(
            "🩺 Predict and explain",
            type="primary",
            use_container_width=True,
        )

    if not submitted and "last_x_row" not in st.session_state:
        st.info("Fill in the patient's values above, then click "
                "**Predict and explain**.")
        return

    # Build / retrieve the input row in the model's feature order
    feature_names = list(bundle["features"])
    if submitted:
        x_row = np.array([values[f] for f in feature_names], dtype=float)
        st.session_state["last_x_row"] = x_row
        st.session_state["last_values"] = values
    else:
        x_row = st.session_state["last_x_row"]
        values = st.session_state["last_values"]

    # ---- Prediction ----
    model = bundle["model"]
    proba = float(model.predict_proba(x_row.reshape(1, -1))[0, 1])
    above_thr = proba >= chosen_threshold

    # Risk band (uses Sens-80 and Youden as cut-points)
    sens80 = float(thresholds_in_bundle.get("sens80", 0.0317))
    youden = float(thresholds_in_bundle.get("youden", 0.0491))
    if proba < sens80:
        band, band_color = "Low risk",       "#2D6A4F"
    elif proba < youden:
        band, band_color = "Moderate risk",  "#C28A2C"
    else:
        band, band_color = "High risk",      "#A0322F"

    # ---- Results ----
    st.divider()
    st.subheader("Results")
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        st.metric("Predicted probability of extubation failure",
                  f"{proba*100:.1f}%")
    with c2:
        st.markdown(
            f"**Risk band**\n\n"
            f"<span style='display:inline-block;padding:4px 12px;"
            f"border-radius:12px;background:{band_color};color:white;"
            f"font-weight:600;font-size:0.95rem;'>{band}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Bands: low &lt; {sens80:.3f} ≤ moderate &lt; {youden:.3f} ≤ high"
        )
    with c3:
        st.metric(
            f"Above selected threshold (p ≥ {chosen_threshold:.3f})",
            "Yes" if above_thr else "No",
        )

    st.markdown(
        f"_Recommended initial threshold from the model bundle is "
        f"**Youden p = {youden:.3f}** (balanced sens/spec) or "
        f"**Sens-80 p = {sens80:.3f}** (high-sensitivity screening); "
        f"choose based on the clinical cost of missed extubation failures._"
    )

    # ---- SHAP ----
    st.divider()
    st.subheader("Why this prediction? — individual SHAP explanation")

    explainer = get_explainer(model)
    shap_values = explainer.shap_values(x_row.reshape(1, -1))
    # XGBoost binary returns a single 2D array in logit space
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    sv_row = np.asarray(shap_values).reshape(-1)
    base_value = float(np.atleast_1d(np.asarray(explainer.expected_value)).ravel()[0])

    # Display values: round nicely so the force plot reads cleanly
    feature_display_values = np.array([
        round(values[f], 2) for f in feature_names
    ], dtype=float)

    st.markdown("**Force plot** (red = pushes toward higher risk, "
                "blue = lower risk)")
    show_force_plot(explainer, sv_row, x_row, feature_names,
                    feature_display_values)

    # Top-contributing features
    contrib_df = pd.DataFrame({
        "Feature": [f.replace("_", " ") for f in feature_names],
        "Patient value": [values[f] for f in feature_names],
        "SHAP (logit)": sv_row,
        "|SHAP|": np.abs(sv_row),
        "Direction": np.where(sv_row >= 0, "↑ risk", "↓ risk"),
    }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)

    with st.expander("📊 Top contributing features (table)"):
        st.dataframe(
            contrib_df.head(15).style.format({
                "Patient value": "{:.2f}",
                "SHAP (logit)": "{:+.3f}",
                "|SHAP|": "{:.3f}",
            }),
            hide_index=True,
            use_container_width=True,
        )
        # Compose a one-line natural-language summary
        top3 = contrib_df.head(3)
        bullet = " · ".join(
            f"**{row.Feature}** = {row['Patient value']:.2f} "
            f"({row.Direction}, SHAP {row['SHAP (logit)']:+.2f})"
            for _, row in top3.iterrows()
        )
        st.markdown(f"Top-3 drivers for this patient: {bullet}")

    with st.expander("🌊 Waterfall plot (alternative view)"):
        try:
            show_waterfall_plot(explainer, sv_row, x_row, feature_names)
        except Exception as e:
            st.warning(f"Could not render waterfall plot: {e}")

    # Math check (additivity)
    raw_logit_pred = base_value + sv_row.sum()
    proba_from_shap = 1.0 / (1.0 + np.exp(-raw_logit_pred))
    with st.expander("🔬 Sanity check: SHAP additivity"):
        st.markdown(
            f"- base log-odds (model expected_value): `{base_value:+.4f}`\n"
            f"- + Σ SHAP values: `{sv_row.sum():+.4f}`\n"
            f"- = predicted log-odds: `{raw_logit_pred:+.4f}`\n"
            f"- σ(predicted log-odds) = **{proba_from_shap*100:.2f}%** "
            f"(should match the predicted probability above)"
        )


if __name__ == "__main__":
    main()
