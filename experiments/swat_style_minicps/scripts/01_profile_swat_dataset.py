import os
import pandas as pd

RAW_DIR = "experiments/swat_style_minicps/data/raw"
OUT_DIR = "experiments/swat_style_minicps/results"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = ["normal.csv", "attack.csv", "merged.csv"]

CORE_COLUMNS = [
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

def clean_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df

for filename in FILES:
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue

    print("=" * 80)
    print(f"Reading {filename}")
    print("=" * 80)

    df = pd.read_csv(path)
    df = clean_columns(df)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("First columns:", list(df.columns[:10]))
    print("Last columns:", list(df.columns[-5:]))

    available = [c for c in CORE_COLUMNS if c in df.columns]
    profile = df[available].describe(include="all").transpose()

    out_path = os.path.join(
        OUT_DIR,
        filename.replace(".csv", "_core_profile.csv")
    )

    profile.to_csv(out_path)

    print("Core columns found:", available)
    print("Saved profile:", out_path)

print("=" * 80)
print("Done. Check experiments/swat_style_minicps/results/")
print("=" * 80)
