import os
import glob
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


def profile_numeric(df, source_name):
    rows = []

    for col in NUMERIC_COLUMNS:
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


def profile_actuators(df, source_name):
    rows = []

    for col in ACTUATOR_COLUMNS:
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


print("=" * 80)
print("Loading real SWaT normal")
print("=" * 80)

swat_normal = pd.read_csv(SWAT_NORMAL_FILE)
swat_normal = clean_columns(swat_normal)

print("Real SWaT normal rows:", len(swat_normal))

print("=" * 80)
print("Loading generated SWaT-style MiniCPS")
print("=" * 80)

files = sorted(glob.glob(os.path.join(GENERATED_DIR, "swat_style_run_*.csv")))

if not files:
    raise FileNotFoundError(f"No generated files found in {GENERATED_DIR}")

generated = pd.concat(
    [clean_columns(pd.read_csv(path)) for path in files],
    ignore_index=True
)

print("Generated rows:", len(generated))

generated[LABEL_COLUMN] = generated[LABEL_COLUMN].astype(str).str.strip()

generated_normal = generated[generated[LABEL_COLUMN].str.lower() == "normal"]
generated_attack = generated[generated[LABEL_COLUMN].str.lower() == "attack"]

print("Generated normal rows:", len(generated_normal))
print("Generated attack rows:", len(generated_attack))

numeric_rows = []
numeric_rows.extend(profile_numeric(swat_normal, "real_swat_normal"))
numeric_rows.extend(profile_numeric(generated_normal, "generated_minicps_normal"))
numeric_rows.extend(profile_numeric(generated_attack, "generated_minicps_attack"))

numeric_df = pd.DataFrame(numeric_rows)

numeric_out = os.path.join(
    RESULTS_DIR,
    "swat_vs_generated_by_label_numeric_profile.csv"
)

numeric_df.to_csv(numeric_out, index=False)

actuator_rows = []
actuator_rows.extend(profile_actuators(swat_normal, "real_swat_normal"))
actuator_rows.extend(profile_actuators(generated_normal, "generated_minicps_normal"))
actuator_rows.extend(profile_actuators(generated_attack, "generated_minicps_attack"))

actuator_df = pd.DataFrame(actuator_rows)

actuator_out = os.path.join(
    RESULTS_DIR,
    "swat_vs_generated_by_label_actuator_profile.csv"
)

actuator_df.to_csv(actuator_out, index=False)

print("=" * 80)
print("Numeric profile by label")
print("=" * 80)
print(numeric_df)

print("=" * 80)
print("Saved files")
print("=" * 80)
print(numeric_out)
print(actuator_out)

print("=" * 80)
print("Done")
print("=" * 80)
