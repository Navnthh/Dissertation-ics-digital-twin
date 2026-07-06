import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/home/nr419/ics_rag_llm_pipeline")

PRED_FILE = PROJECT_DIR / "data" / "external_results" / "qwen_lora_feature_summary_predictions_all.jsonl"
TEST_FILE = PROJECT_DIR / "data" / "external_results" / "test_compact.jsonl"

# Fallback to Google Drive exported copies if you manually copied them into the project later.
ALT_PRED_FILE = PROJECT_DIR / "results" / "qwen_lora_feature_summary_predictions_all.jsonl"
ALT_TEST_FILE = PROJECT_DIR / "data" / "finetune" / "test_compact.jsonl"

EXPLAIN_SCRIPT = PROJECT_DIR / "scripts" / "05b_explain_detection_clean_report.py"
OUT_DIR = PROJECT_DIR / "results" / "rag_explanations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = OUT_DIR / "three_rag_explanation_examples_summary.json"
SUMMARY_CSV = OUT_DIR / "three_rag_explanation_examples_summary.csv"


def resolve_file(primary, alt, name):
    if primary.exists():
        return primary

    if alt.exists():
        return alt

    raise FileNotFoundError(
        f"Could not find {name}.\n"
        f"Tried:\n"
        f"  {primary}\n"
        f"  {alt}\n\n"
        f"If this fails, copy these files from Colab/Drive into:\n"
        f"  {PROJECT_DIR}/data/external_results/"
    )


def extract_sensor_csv(user_text):
    match = re.search(
        r"Sensor CSV:\n(.*?)(?:\n\nReturn only JSON:|\Z)",
        user_text,
        flags=re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return user_text.strip()


def extract_source(user_text):
    match = re.search(r"Source:\s*(.+)", user_text)
    return match.group(1).strip() if match else "unknown"


def extract_cycles(user_text):
    match = re.search(r"Cycles:\s*(.+)", user_text)
    return match.group(1).strip() if match else "unknown"


def parse_csv_rows(csv_text):
    rows = []

    for line in csv_text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("step,"):
            continue

        parts = line.split(",")

        if len(parts) < 6:
            continue

        try:
            rows.append({
                "step": int(float(parts[0])),
                "cycle": int(float(parts[1])),
                "S1": float(parts[2]),
                "S2": float(parts[3]),
                "S3": float(parts[4]),
                "A1": float(parts[5]),
            })
        except Exception:
            continue

    return rows


def summary(values):
    return {
        "first": values[0],
        "last": values[-1],
        "change": values[-1] - values[0],
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "range": max(values) - min(values),
    }


def round6(x):
    return round(float(x), 6)


def get_features(example):
    user_text = example["messages"][1]["content"]
    csv_text = extract_sensor_csv(user_text)
    rows = parse_csv_rows(csv_text)

    if not rows:
        raise ValueError("No rows parsed from example.")

    s1 = [r["S1"] for r in rows]
    s2 = [r["S2"] for r in rows]
    s3 = [r["S3"] for r in rows]
    a1 = [r["A1"] for r in rows]

    s1s = summary(s1)
    s2s = summary(s2)
    s3s = summary(s3)

    valve_open_ratio = sum(1 for x in a1 if x == 1.0) / len(a1)
    valve_transitions = sum(1 for i in range(1, len(a1)) if a1[i] != a1[i - 1])

    return {
        "source_file": extract_source(user_text),
        "cycle_range": extract_cycles(user_text),
        "cycle_start": rows[0]["cycle"],
        "cycle_end": rows[-1]["cycle"],

        "s1_change": round6(s1s["change"]),
        "s1_range": round6(s1s["range"]),
        "s1_mean": round6(s1s["mean"]),

        "s2_change": round6(s2s["change"]),
        "s2_range": round6(s2s["range"]),

        "s3_change": round6(s3s["change"]),
        "s3_range": round6(s3s["range"]),

        "a1_open_ratio": round6(valve_open_ratio),
        "a1_transitions": valve_transitions,
    }


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def choose_examples(predictions):
    chosen = {
        "true_positive": None,
        "false_positive": None,
        "false_negative": None,
    }

    for row in predictions:
        gold = row["gold"]
        pred = row["prediction"]

        if chosen["true_positive"] is None and gold == "Attack" and pred == "Attack":
            chosen["true_positive"] = row

        if chosen["false_positive"] is None and gold == "Normal" and pred == "Attack":
            chosen["false_positive"] = row

        if chosen["false_negative"] is None and gold == "Attack" and pred == "Normal":
            chosen["false_negative"] = row

        if all(chosen.values()):
            break

    missing = [key for key, value in chosen.items() if value is None]
    if missing:
        raise RuntimeError(f"Could not find required example types: {missing}")

    return chosen


def run_explanation(example_name, pred_row, test_examples):
    idx = pred_row["index"]
    example = test_examples[idx]
    features = get_features(example)

    cmd = [
        sys.executable,
        str(EXPLAIN_SCRIPT),

        "--prediction", pred_row["prediction"],
        "--source-file", features["source_file"],
        "--cycle-range", features["cycle_range"],

        "--s1-change", str(features["s1_change"]),
        "--s1-range", str(features["s1_range"]),
        "--s1-mean", str(features["s1_mean"]),

        "--s2-change", str(features["s2_change"]),
        "--s2-range", str(features["s2_range"]),

        "--s3-change", str(features["s3_change"]),
        "--s3-range", str(features["s3_range"]),

        "--a1-open-ratio", str(features["a1_open_ratio"]),
        "--a1-transitions", str(features["a1_transitions"]),

        "--top-k", "3",
    ]

    print("\n" + "=" * 100)
    print(f"GENERATING {example_name.upper()}")
    print("=" * 100)
    print("Index:", idx)
    print("Gold:", pred_row["gold"])
    print("Prediction:", pred_row["prediction"])
    print("Source:", features["source_file"])
    print("Cycles:", features["cycle_range"])

    subprocess.run(cmd, check=True)

    safe_source = features["source_file"].replace(".csv", "").replace("/", "_")
    safe_cycles = features["cycle_range"].replace(" ", "").replace("to", "_")

    txt_path = OUT_DIR / f"clean_rag_explanation_{safe_source}_{safe_cycles}.txt"
    json_path = OUT_DIR / f"clean_rag_explanation_{safe_source}_{safe_cycles}.json"

    return {
        "example_type": example_name,
        "index": idx,
        "gold": pred_row["gold"],
        "prediction": pred_row["prediction"],
        "raw_model_output": pred_row.get("raw_model_output", ""),
        **features,
        "txt_report": str(txt_path),
        "json_report": str(json_path),
    }


def main():
    if not EXPLAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing explanation script: {EXPLAIN_SCRIPT}")

    pred_file = resolve_file(PRED_FILE, ALT_PRED_FILE, "predictions file")
    test_file = resolve_file(TEST_FILE, ALT_TEST_FILE, "test compact file")

    print("=" * 100)
    print("INPUT FILES")
    print("=" * 100)
    print("Predictions:", pred_file)
    print("Test compact:", test_file)

    predictions = load_jsonl(pred_file)
    test_examples = load_jsonl(test_file)

    print("Prediction rows:", len(predictions))
    print("Test examples:", len(test_examples))

    chosen = choose_examples(predictions)

    summaries = []

    for example_name, pred_row in chosen.items():
        summaries.append(run_explanation(example_name, pred_row, test_examples))

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 100)
    print("THREE RAG EXPLANATION EXAMPLES COMPLETE")
    print("=" * 100)
    print("Summary JSON:", SUMMARY_JSON)
    print("Summary CSV:", SUMMARY_CSV)

    print("\nGenerated reports:")
    for row in summaries:
        print(f"- {row['example_type']}: {row['txt_report']}")


if __name__ == "__main__":
    main()
