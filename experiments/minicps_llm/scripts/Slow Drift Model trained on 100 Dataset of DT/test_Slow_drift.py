%%bash
mkdir -p /content/drive/MyDrive/ics_llm_outputs/scripts

cat > /content/drive/MyDrive/ics_llm_outputs/scripts/evaluate_minicps_feature_summary.py <<'PY'
import os
import re
import json
import torch
import statistics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_FEATURE_SUMMARY"
TEST_FILE = "/content/drive/MyDrive/ics_llm_outputs/data/finetune_compact/test_compact.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/ics_llm_outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "100"))
SYSTEM_PROMPT = "You are an ICS anomaly detector. Output JSON only."

def extract_sensor_csv(user_text):
    match = re.search(
        r"Sensor CSV:\n(.*?)(?:\n\nReturn only JSON:|\Z)",
        user_text,
        flags=re.DOTALL
    )
    return match.group(1).strip() if match else user_text.strip()

def parse_csv_rows(csv_text):
    rows = []

    for line in csv_text.splitlines():
        line = line.strip()

        if not line or line.lower().startswith("step,"):
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

def series_summary(values):
    if not values:
        return {
            "first": 0.0,
            "last": 0.0,
            "change": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "range": 0.0,
        }

    return {
        "first": values[0],
        "last": values[-1],
        "change": values[-1] - values[0],
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "range": max(values) - min(values),
    }

def fmt(x):
    return f"{x:.6f}"

def make_feature_prompt(messages):
    old_user_text = messages[1]["content"]
    csv_text = extract_sensor_csv(old_user_text)
    rows = parse_csv_rows(csv_text)

    if not rows:
        return (
            "Classify this MiniCPS sensor window as Attack or Normal.\n"
            "No valid rows parsed.\n"
            "Returncat > experiments/minicps_llm/scripts/evaluate_minicps_feature_summary.py exactly: {\"prediction\":\"Attack\"} or {\"prediction\":\"Normal\"}"
        )

    cycles = [r["cycle"] for r in rows]
    s1 = [r["S1"] for r in rows]
    s2 = [r["S2"] for r in rows]
    s3 = [r["S3"] for r in rows]
    a1 = [r["A1"] for r in rows]

    s1s = series_summary(s1)
    s2s = series_summary(s2)
    s3s = series_summary(s3)

    valve_open_ratio = sum(1 for x in a1 if x == 1.0) / len(a1)
    valve_transitions = sum(1 for i in range(1, len(a1)) if a1[i] != a1[i - 1])

    return f"""Classify this MiniCPS sensor window as Attack or Normal.

A stealthy attack may slowly manipulate sensed telemetry while keeping values plausible.
Use only these controller-visible summary features.

cycle_start={cycles[0]}
cycle_end={cycles[-1]}
rows={len(rows)}

S1_tank_sensed:
first={fmt(s1s["first"])}, last={fmt(s1s["last"])}, change={fmt(s1s["change"])}, min={fmt(s1s["min"])}, max={fmt(s1s["max"])}, mean={fmt(s1s["mean"])}, range={fmt(s1s["range"])}

S2_flow_sensed:
first={fmt(s2s["first"])}, last={fmt(s2s["last"])}, change={fmt(s2s["change"])}, min={fmt(s2s["min"])}, max={fmt(s2s["max"])}, mean={fmt(s2s["mean"])}, range={fmt(s2s["range"])}

S3_bottle_sensed:
first={fmt(s3s["first"])}, last={fmt(s3s["last"])}, change={fmt(s3s["change"])}, min={fmt(s3s["min"])}, max={fmt(s3s["max"])}, mean={fmt(s3s["mean"])}, range={fmt(s3s["range"])}

A1_valve:
open_ratio={fmt(valve_open_ratio)}, transitions={valve_transitions}

Return exactly one JSON object:
{{"prediction":"Attack"}} or {{"prediction":"Normal"}}"""

def parse_prediction(text):
    if text is None:
        return "Unknown"

    text = text.strip()

    try:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            pred = str(obj.get("prediction", "")).strip().lower()

            if pred == "attack":
                return "Attack"

            if pred == "normal":
                return "Normal"
    except Exception:
        pass

    lower = text.lower()

    if '"prediction":"attack"' in lower or '"prediction": "attack"' in lower:
        return "Attack"

    if '"prediction":"normal"' in lower or '"prediction": "normal"' in lower:
        return "Normal"

    if "attack" in lower and "normal" not in lower:
        return "Attack"

    if "normal" in lower and "attack" not in lower:
        return "Normal"

    return "Unknown"

def load_examples(path, max_examples):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

            if max_examples > 0 and len(examples) >= max_examples:
                break

    return examples

print("=" * 80)
print("Evaluation setup")
print("=" * 80)
print("ADAPTER_PATH:", ADAPTER_PATH)
print("Adapter exists:", os.path.exists(ADAPTER_PATH))
print("TEST_FILE:", TEST_FILE)
print("Test file exists:", os.path.exists(TEST_FILE))
print("MAX_EXAMPLES:", MAX_EXAMPLES)

if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"Missing adapter path: {ADAPTER_PATH}")

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

print("=" * 80)
print("GPU check")
print("=" * 80)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Use Colab GPU.")

print("GPU:", torch.cuda.get_device_name(0))

print("=" * 80)
print("Loading tokenizer")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("=" * 80)
print("Loading base model")
print("=" * 80)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("=" * 80)
print("Loading LoRA adapter")
print("=" * 80)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

examples = load_examples(TEST_FILE, MAX_EXAMPLES)

print("=" * 80)
print("Loaded test examples:", len(examples))
print("=" * 80)

TP = TN = FP = FN = 0
unknown_count = 0

suffix = "all" if MAX_EXAMPLES <= 0 else str(len(examples))

predictions_path = os.path.join(
    OUTPUT_DIR,
    f"qwen_lora_feature_summary_predictions_{suffix}.jsonl"
)

device = next(model.parameters()).device

with open(predictions_path, "w", encoding="utf-8") as out:
    for idx, example in enumerate(tqdm(examples)):
        messages = example["messages"]
        gold = parse_prediction(messages[-1]["content"])

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_feature_prompt(messages)}
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        pred = parse_prediction(generated_text)

        if pred == "Unknown":
            unknown_count += 1

        if gold == "Attack" and pred == "Attack":
            TP += 1
        elif gold == "Normal" and pred == "Normal":
            TN += 1
        elif gold == "Normal" and pred == "Attack":
            FP += 1
        elif gold == "Attack" and pred == "Normal":
            FN += 1
        else:
            if gold == "Attack":
                FN += 1
            elif gold == "Normal":
                FP += 1

        out.write(json.dumps({
            "index": idx,
            "gold": gold,
            "prediction": pred,
            "raw_model_output": generated_text
        }) + "\n")

total = len(examples)
accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

metrics = {
    "total_examples": total,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,
    "unknown_predictions": unknown_count,
    "adapter_path": ADAPTER_PATH,
    "test_file": TEST_FILE,
    "predictions_file": predictions_path
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    f"qwen_lora_feature_summary_metrics_{suffix}.json"
)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("=" * 80)
print("Evaluation complete")
print("=" * 80)
print(json.dumps(metrics, indent=2))
print("Metrics saved to:", metrics_path)
print("Predictions saved to:", predictions_path)
PY

ls -lh /content/drive/MyDrive/ics_llm_outputs/scripts/evaluate_minicps_feature_summary.py