import os
import glob
import json
import pandas as pd
import numpy as np

SWAT_NORMAL_FILE = "experiments/swat_style_minicps/data/raw/normal.csv"
GENERATED_DIR = "experiments/swat_style_minicps/data/processed/swat_style_minicps_runs"
RESULTS_DIR = "experiments/swat_style_minicps/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

NUMERIC_COLUMNS = [
    "FIT101",
    "LIT101",
    "FIT301",
    "LIT301",
]

ACTUATOR_COLUMNS = [
    "MV101",
    "P101",
    "P102",
    "MV301",
    "P301",
    "P302",
]

LABEL_COLUMN = "Normal/Attack"


def clean_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def numeric_profile(df, columns, source_name):
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        rows.append({
            "source": source_name,
            "column": col,
            "count": int(len(values)),
            "min": float(values.min()),
            "p01": float(np.percentile(values, 1)),
            "mean": float(values.mean()),
            "median": float(np.percentile(values, 50)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
            "std": float(values.std()),
        })

    return rows


def actuator_profile(df, columns, source_name):
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        counts = values.value_counts().sort_index()

        for state, count in counts.items():
            rows.append({
                "source": source_name,
                "column": col,
                "state": float(state),
                "count": int(count),
                "percentage": float(count / len(values)) if len(values) else 0.0,
            })

    return rows


def label_profile(df, source_name):
    if LABEL_COLUMN not in df.columns:
        return []

    counts = df[LABEL_COLUMN].astype(str).str.strip().value_counts()

    rows = []

    for label, count in counts.items():
        rows.append({
            "source": source_name,
            "label": label,
            "count": int(count),
            "percentage": float(count / len(df)) if len(df) else 0.0,
        })

    return rows


print("=" * 80)
print("Loading real SWaT normal dataset")
print("=" * 80)

swat = pd.read_csv(SWAT_NORMAL_FILE)
swat = clean_columns(swat)

print("SWaT rows:", len(swat))

print("=" * 80)
print("Loading generated SWaT-style MiniCPS files")
print("=" * 80)

generated_files = sorted(glob.glob(os.path.join(GENERATED_DIR, "swat_style_run_*.csv")))

if not generated_files:
    raise FileNotFoundError(f"No generated files found in {GENERATED_DIR}")

frames = []

for path in generated_files:
    df = pd.read_csv(path)
    df = clean_columns(df)
    frames.append(df)

generated = pd.concat(frames, ignore_index=True)

print("Generated files:", len(generated_files))
print("Generated rows:", len(generated))

print("=" * 80)
print("Creating comparison profiles")
print("=" * 80)

numeric_rows = []
numeric_rows.extend(numeric_profile(swat, NUMERIC_COLUMNS, "real_swat_normal"))
numeric_rows.extend(numeric_profile(generated, NUMERIC_COLUMNS, "generated_swat_style_minicps"))

numeric_df = pd.DataFrame(numeric_rows)
numeric_out = os.path.join(RESULTS_DIR, "swat_vs_generated_numeric_profile.csv")
numeric_df.to_csv(numeric_out, index=False)

actuator_rows = []
actuator_rows.extend(actuator_profile(swat, ACTUATOR_COLUMNS, "real_swat_normal"))
actuator_rows.extend(actuator_profile(generated, ACTUATOR_COLUMNS, "generated_swat_style_minicps"))

actuator_df = pd.DataFrame(actuator_rows)
actuator_out = os.path.join(RESULTS_DIR, "swat_vs_generated_actuator_profile.csv")
actuator_df.to_csv(actuator_out, index=False)

label_rows = []
label_rows.extend(label_profile(swat, "real_swat_normal"))
label_rows.extend(label_profile(generated, "generated_swat_style_minicps"))

label_df = pd.DataFrame(label_rows)
label_out = os.path.join(RESULTS_DIR, "swat_vs_generated_label_profile.csv")
label_df.to_csv(label_out, index=False)

summary = {
    "real_swat_normal_rows": int(len(swat)),
    "generated_swat_style_files": int(len(generated_files)),
    "generated_swat_style_rows": int(len(generated)),
    "numeric_profile": numeric_out,
    "actuator_profile": actuator_out,
    "label_profile": label_out,
    "note": "Generated data is SWaT-style synthetic MiniCPS telemetry. It is compared against SWaT normal-operation ranges and actuator states."
}

summary_out = os.path.join(RESULTS_DIR, "swat_style_validation_summary.json")

with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved numeric profile:", numeric_out)
print("Saved actuator profile:", actuator_out)
print("Saved label profile:", label_out)
print("Saved summary:", summary_out)

print("=" * 80)
print("Numeric comparison preview")
print("=" * 80)
print(numeric_df)

print("=" * 80)
print("Label comparison preview")
print("=" * 80)
print(label_df)

print("=" * 80)
print("Done")
print("=" * 80)
