"""
Inference script for the extubation-failure XGBoost model (v2).

This version automatically applies preprocessing rules stored in the
model bundle. In particular, mv_duration_hours <= 0 is converted to NaN
before prediction (and then median-imputed), because eICU's
respiratorycare table contains invalid ventstartoffset/ventendoffset
records that would otherwise flip the prediction direction.

Usage
-----
    python predict.py --input new_patients.csv
    python predict.py --input new_patients.csv --threshold youden
    python predict.py --input new_patients.csv --threshold 0.05
    python predict.py --input new_patients.csv --no-clean-input   # skip rules
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("xgb_extubation_failure_v2.pkl")


def load_bundle(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_preprocessing(df, bundle):
    """Apply bundled preprocessing rules in-place on a copy."""
    df = df.copy()
    rules = bundle.get("preprocessing", {}).get("rules", [])
    report = []
    for rule in rules:
        col = rule["column"]
        if col not in df.columns:
            report.append(f"  [skip] {col}: not in input")
            continue
        if "value <= 0" in rule["rule"]:
            mask = df[col] <= 0
            n = int(mask.sum())
            df.loc[mask, col] = np.nan
            report.append(f"  [applied] {col}: {n} non-positive values → NaN")
    return df, report


def prepare_features(df, bundle, clean_input=True):
    """Align an arbitrary input DataFrame to the model's expected feature
    matrix:
      - apply bundled preprocessing rules (if clean_input=True)
      - one-hot encode object columns (drop_first=True)
      - keep only the features the model was trained on
      - fill missing columns with NaN
      - coerce to numeric
      - apply the stored median imputer
    """
    if clean_input:
        df, rep = apply_preprocessing(df, bundle)
        for line in rep:
            print(line, file=sys.stderr)

    cat = df.select_dtypes(include=["object"]).columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat, drop_first=True)
    for f in bundle["features"]:
        if f not in df_enc.columns:
            df_enc[f] = np.nan
    X = df_enc[bundle["features"]].apply(pd.to_numeric, errors="coerce").values
    return bundle["imputer"].transform(X)


def predict(df, bundle, threshold="sens80", clean_input=True):
    """Return (probs, preds, threshold_value)."""
    X = prepare_features(df, bundle, clean_input=clean_input)
    probs = bundle["model"].predict_proba(X)[:, 1]
    if isinstance(threshold, str):
        thr = bundle["thresholds"][threshold]
    else:
        thr = float(threshold)
    preds = (probs >= thr).astype(int)
    return probs, preds, thr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input CSV")
    ap.add_argument("--output", default=None,
                    help="Output CSV (default: <input>_pred.csv)")
    ap.add_argument("--threshold", default="sens80",
                    help="One of {youden, sens80, default} or a float "
                         "(default: sens80)")
    ap.add_argument("--no-clean-input", action="store_true",
                    help="Skip bundled preprocessing rules "
                         "(use raw values as-is)")
    ap.add_argument("--model", default=str(MODEL_PATH),
                    help="Path to model .pkl")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    md = bundle["metadata"]
    print(f"Loaded model v{md.get('version', '?')}: {md['model']}",
          file=sys.stderr)
    print(f"  trained on: {md['trained_on']}", file=sys.stderr)
    if "external_validation" in md:
        ev = md["external_validation"]
        print(f"  external validation AUROC: {ev['AUROC']} "
              f"95% CI {ev['AUROC_95CI']} (n={ev['n_used']})", file=sys.stderr)
    print(f"  features: {len(bundle['features'])}, thresholds: "
          f"{list(bundle['thresholds'])}", file=sys.stderr)

    df = pd.read_csv(args.input)
    print(f"Input rows: {len(df)}", file=sys.stderr)

    probs, preds, thr = predict(df, bundle,
                                 threshold=args.threshold,
                                 clean_input=not args.no_clean_input)
    print(f"Used threshold: {thr:.4f}  "
          f"(predicted positive rate: {preds.mean():.3f})", file=sys.stderr)

    out = df.copy()
    out["prob_extubation_failure"] = probs
    out["pred_extubation_failure"] = preds

    out_path = (args.output
                or Path(args.input).with_suffix("").as_posix() + "_pred.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
