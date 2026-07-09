import os
import json
import glob
import pandas as pd
import numpy as np

SWAT_NORMAL_FILE = "experiments/swat_style_minicps/data/raw/normal.csv"

MINICPS_SOURCE_DIR = os.path.expanduser(
    "~/ics_dataset_test/live_final_100_15min_actual_sensed"
)

OUT_DIR = "experiments/swat_style_minicps/data/processed/swat_style_minicps_runs"
RESULTS_DIR = "experiments/swat_style_minicps/results"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CORE_SWAT_COLUMNS = [
    "FIT101",
    "LIT101",
    "MV101",
    "P101",
    "P102",
    "FIT301",
    "LIT301",
    "MV301",
    "P301",
    "P302",
]

NUMERIC_REFERENCE_COLUMNS = [
    "FIT101",
    "LIT101",
    "FIT301",
    "LIT301",
]

ACTUATOR_REFERENCE_COLUMNS = [
    "MV101",
    "P101",
    "P102",
    "MV301",
    "P301",
    "P302",
]

MINICPS_COLUMNS = {
    "s1": "SENSOR1_SENSED",
    "s2": "SENSOR2_SENSED",
    "s3": "SENSOR3_SENSED",
    "a1": "ACTUATOR1",
    "attack": "attack",
}


def clean_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def get_percentile_range(series, low=1, high=99):
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if len(numeric) == 0:
        return {
            "p01": 0.0,
            "p99": 1.0,
            "min": 0.0,
            "max": 1.0,
            "mean": 0.0,
        }

    return {
        "p01": float(np.percentile(numeric, low)),
        "p99": float(np.percentile(numeric, high)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def get_states(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    states = sorted(numeric.unique().tolist())

    if not states:
        return [0.0, 1.0]

    return [float(x) for x in states]


def scale_series(values, src_min, src_max, dst_min, dst_max):
    values = pd.to_numeric(values, errors="coerce").fillna(src_min)

    if src_max == src_min:
        return pd.Series([dst_min] * len(values))

    scaled = (values - src_min) / (src_max - src_min)
    scaled = scaled.clip(0, 1)

    return dst_min + scaled * (dst_max - dst_min)


def read_all_minicps_files():
    files = sorted(glob.glob(os.path.join(MINICPS_SOURCE_DIR, "run_*.csv")))

    if not files:
        raise FileNotFoundError(f"No MiniCPS run_*.csv files found in {MINICPS_SOURCE_DIR}")

    frames = []

    for path in files:
        df = pd.read_csv(path)
        df = clean_columns(df)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    return files, pd.concat(frames, ignore_index=True)


print("=" * 80)
print("Loading SWaT normal dataset")
print("=" * 80)

if not os.path.exists(SWAT_NORMAL_FILE):
    raise FileNotFoundError(f"Missing SWaT normal file: {SWAT_NORMAL_FILE}")

swat = pd.read_csv(SWAT_NORMAL_FILE)
swat = clean_columns(swat)

print("SWaT rows:", len(swat))
print("SWaT columns:", len(swat.columns))

print("=" * 80)
print("Building SWaT reference ranges and actuator states")
print("=" * 80)

swat_reference = {
    "numeric_ranges": {},
    "actuator_states": {},
}

for col in NUMERIC_REFERENCE_COLUMNS:
    swat_reference["numeric_ranges"][col] = get_percentile_range(swat[col])
    print(col, swat_reference["numeric_ranges"][col])

for col in ACTUATOR_REFERENCE_COLUMNS:
    swat_reference["actuator_states"][col] = get_states(swat[col])
    print(col, swat_reference["actuator_states"][col])

reference_path = os.path.join(RESULTS_DIR, "swat_reference_ranges_and_states.json")

with open(reference_path, "w", encoding="utf-8") as f:
    json.dump(swat_reference, f, indent=2)

print("Saved SWaT reference:", reference_path)

print("=" * 80)
print("Loading MiniCPS source runs")
print("=" * 80)

files, minicps_all = read_all_minicps_files()

print("MiniCPS files:", len(files))
print("MiniCPS rows:", len(minicps_all))

required = list(MINICPS_COLUMNS.values())

missing = [c for c in required if c not in minicps_all.columns]
if missing:
    raise ValueError(f"MiniCPS source files are missing required columns: {missing}")

minicps_ranges = {}

for key, col in MINICPS_COLUMNS.items():
    if key == "attack":
        continue

    vals = pd.to_numeric(minicps_all[col], errors="coerce").dropna()
    minicps_ranges[col] = {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }

print("MiniCPS source ranges:")
for col, stats in minicps_ranges.items():
    print(col, stats)

minicps_range_path = os.path.join(RESULTS_DIR, "minicps_source_ranges.json")

with open(minicps_range_path, "w", encoding="utf-8") as f:
    json.dump(minicps_ranges, f, indent=2)

print("Saved MiniCPS ranges:", minicps_range_path)

print("=" * 80)
print("Generating SWaT-style MiniCPS CSV files")
print("=" * 80)

mv101_states = swat_reference["actuator_states"]["MV101"]
p101_states = swat_reference["actuator_states"]["P101"]
p102_states = swat_reference["actuator_states"]["P102"]
mv301_states = swat_reference["actuator_states"]["MV301"]
p301_states = swat_reference["actuator_states"]["P301"]
p302_states = swat_reference["actuator_states"]["P302"]

mv101_off, mv101_on = min(mv101_states), max(mv101_states)
p101_off, p101_on = min(p101_states), max(p101_states)
p102_off, p102_on = min(p102_states), max(p102_states)
mv301_off, mv301_on = min(mv301_states), max(mv301_states)
p301_off, p301_on = min(p301_states), max(p301_states)
p302_off, p302_on = min(p302_states), max(p302_states)

generated_files = []

for path in files:
    df = pd.read_csv(path)
    df = clean_columns(df)

    out = pd.DataFrame()

    if "timestamp" in df.columns:
        out["Timestamp"] = df["timestamp"]
    elif "Timestamp" in df.columns:
        out["Timestamp"] = df["Timestamp"]
    else:
        out["Timestamp"] = pd.date_range(
            start="2026-01-01",
            periods=len(df),
            freq="1s"
        ).astype(str)

    s1 = df[MINICPS_COLUMNS["s1"]]
    s2 = df[MINICPS_COLUMNS["s2"]]
    s3 = df[MINICPS_COLUMNS["s3"]]
    a1 = pd.to_numeric(df[MINICPS_COLUMNS["a1"]], errors="coerce").fillna(0)

    # MiniCPS SENSOR1 -> SWaT LIT101
    lit101_ref = swat_reference["numeric_ranges"]["LIT101"]
    out["LIT101"] = scale_series(
        s1,
        minicps_ranges[MINICPS_COLUMNS["s1"]]["min"],
        minicps_ranges[MINICPS_COLUMNS["s1"]]["max"],
        lit101_ref["p01"],
        lit101_ref["p99"],
    )

    # MiniCPS SENSOR2 -> SWaT FIT101
    fit101_ref = swat_reference["numeric_ranges"]["FIT101"]
    out["FIT101"] = scale_series(
        s2,
        minicps_ranges[MINICPS_COLUMNS["s2"]]["min"],
        minicps_ranges[MINICPS_COLUMNS["s2"]]["max"],
        fit101_ref["p01"],
        fit101_ref["p99"],
    )

    # MiniCPS SENSOR3 -> SWaT LIT301
    lit301_ref = swat_reference["numeric_ranges"]["LIT301"]
    out["LIT301"] = scale_series(
        s3,
        minicps_ranges[MINICPS_COLUMNS["s3"]]["min"],
        minicps_ranges[MINICPS_COLUMNS["s3"]]["max"],
        lit301_ref["p01"],
        lit301_ref["p99"],
    )

    # Downstream flow is derived from SENSOR2, with slight smoothing-like scaling
    fit301_ref = swat_reference["numeric_ranges"]["FIT301"]
    out["FIT301"] = scale_series(
        s2,
        minicps_ranges[MINICPS_COLUMNS["s2"]]["min"],
        minicps_ranges[MINICPS_COLUMNS["s2"]]["max"],
        fit301_ref["p01"],
        fit301_ref["p99"],
    )

    # MiniCPS ACTUATOR1 -> SWaT-style motorized valves and pumps
    out["MV101"] = np.where(a1 >= 0.5, mv101_on, mv101_off)
    out["P101"] = np.where(a1 >= 0.5, p101_on, p101_off)
    out["P102"] = p102_off

    out["MV301"] = np.where(a1 >= 0.5, mv301_on, mv301_off)
    out["P301"] = np.where(a1 >= 0.5, p301_on, p301_off)
    out["P302"] = p302_off

    attack = pd.to_numeric(df[MINICPS_COLUMNS["attack"]], errors="coerce").fillna(0)
    out["Normal/Attack"] = np.where(attack >= 1, "Attack", "Normal")

    # Keep traceability columns at the end
    if "run_id" in df.columns:
        out["source_run_id"] = df["run_id"]
    out["source_file"] = os.path.basename(path)
    if "cycle" in df.columns:
        out["source_cycle"] = df["cycle"]

    ordered_cols = [
        "Timestamp",
        "FIT101",
        "LIT101",
        "MV101",
        "P101",
        "P102",
        "FIT301",
        "LIT301",
        "MV301",
        "P301",
        "P302",
        "Normal/Attack",
    ]

    trace_cols = [c for c in ["source_run_id", "source_file", "source_cycle"] if c in out.columns]
    out = out[ordered_cols + trace_cols]

    out_file = os.path.join(OUT_DIR, "swat_style_" + os.path.basename(path))
    out.to_csv(out_file, index=False)

    generated_files.append(out_file)

print("Generated files:", len(generated_files))
print("Output folder:", OUT_DIR)

summary = {
    "swat_normal_file": SWAT_NORMAL_FILE,
    "minicps_source_dir": MINICPS_SOURCE_DIR,
    "output_dir": OUT_DIR,
    "generated_files": len(generated_files),
    "mapping": {
        "SENSOR1_SENSED": "LIT101",
        "SENSOR2_SENSED": "FIT101",
        "SENSOR3_SENSED": "LIT301",
        "ACTUATOR1": ["MV101", "P101", "MV301", "P301"],
        "attack": "Normal/Attack",
    },
    "note": "This creates SWaT-style synthetic MiniCPS telemetry. It does not copy SWaT rows."
}

summary_path = os.path.join(RESULTS_DIR, "swat_style_minicps_generation_summary.json")

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved summary:", summary_path)
print("=" * 80)
print("Done")
print("=" * 80)
