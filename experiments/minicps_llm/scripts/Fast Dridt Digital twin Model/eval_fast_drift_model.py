# ═══════════════════════════════════════════════════════════════
# EVALUATION — Fast Drift Model (FAST10) on Fast Drift Test Data
# (exact Colab code used, verbatim)
# ═══════════════════════════════════════════════════════════════

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import pandas as pd
import statistics
import json
import os

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FAST10_LORA = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_FAST10"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model_fast = PeftModel.from_pretrained(base, FAST10_LORA)
model_fast.eval()
print("Fast drift model loaded!")

def series_summary(values):
    if not values:
        return {"first":0.0,"last":0.0,"change":0.0,"min":0.0,"max":0.0,"mean":0.0,"range":0.0}
    return {
        "first": values[0], "last": values[-1],
        "change": values[-1]-values[0],
        "min": min(values), "max": max(values),
        "mean": statistics.mean(values),
        "range": max(values)-min(values)
    }

def fmt(x): return f"{x:.6f}"

def detect_fast(window_df, start_cycle=0):
    s1 = window_df['SENSOR1_SENSED'].tolist()
    s2 = window_df['SENSOR2_SENSED'].tolist()
    s3 = window_df['SENSOR3_SENSED'].tolist()
    a1 = window_df['ACTUATOR1'].tolist()

    s1s = series_summary(s1)
    s2s = series_summary(s2)
    s3s = series_summary(s3)

    valve_open_ratio = sum(1 for x in a1 if x >= 1.0) / len(a1)
    valve_transitions = sum(1 for i in range(1, len(a1)) if a1[i] != a1[i-1])

    prompt_content = f"""Classify this MiniCPS sensor window as Attack or Normal.

A stealthy attack may slowly manipulate sensed telemetry while keeping values plausible.
Use only these controller-visible summary features.

cycle_start={start_cycle}
cycle_end={start_cycle + len(window_df) - 1}
rows={len(window_df)}

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

    messages = [
        {"role": "system", "content": "You are an ICS anomaly detector. Output JSON only."},
        {"role": "user", "content": prompt_content}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_fast.device)

    with torch.no_grad():
        outputs = model_fast.generate(**inputs, max_new_tokens=20, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    first_line = response.strip().split('\n')[0].strip()
    if "attack" in first_line.lower(): return "Attack"
    elif "normal" in first_line.lower(): return "Normal"
    return "Unknown"

# ═══════════════════════════════════════════════════════════════
# TEST ON fast_run_009 and fast_run_010
# ═══════════════════════════════════════════════════════════════
test_files = ['fast_run_009.csv', 'fast_run_010.csv']
window_size = 30

all_results = []
all_labels = []

for fname in test_files:
    df = pd.read_csv(f'/content/fast_drift_files/{fname}')
    print(f"\nTesting: {fname}")

    for i in range(window_size, len(df), window_size):
        window = df.iloc[i-window_size:i]
        attack_pct = (window['attack']==1).sum() / len(window) * 100
        if attack_pct == 100:
            label = 'Attack'
        elif attack_pct == 0:
            label = 'Normal'
        else:
            continue
        pred = detect_fast(window, i-window_size)
        all_results.append(pred)
        all_labels.append(label)

# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
TP = sum(1 for t,p in zip(all_labels,all_results) if t=='Attack' and p=='Attack')
TN = sum(1 for t,p in zip(all_labels,all_results) if t=='Normal' and p=='Normal')
FP = sum(1 for t,p in zip(all_labels,all_results) if t=='Normal' and p=='Attack')
FN = sum(1 for t,p in zip(all_labels,all_results) if t=='Attack' and p=='Normal')

total = TP+TN+FP+FN
accuracy  = (TP+TN)/total*100
precision = TP/(TP+FP)*100 if (TP+FP)>0 else 0
recall    = TP/(TP+FN)*100 if (TP+FN)>0 else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

print(f"\n{'='*60}")
print(f"FINAL RESULTS — Fast Drift Model (10 files trained)")
print(f"{'='*60}")
print(f"Total windows: {total}")
print(f"  Attack: {all_labels.count('Attack')}")
print(f"  Normal: {all_labels.count('Normal')}")
print(f"\nAccuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall:    {recall:.2f}%")
print(f"F1-score:  {f1:.2f}%")
print(f"\nConfusion Matrix:")
print(f"  TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")

# ═══════════════════════════════════════════════════════════════
# ACTUAL COLAB OUTPUT:
#
# Fast drift model loaded!
# Testing: fast_run_009.csv
# Testing: fast_run_010.csv
#
# ============================================================
# FINAL RESULTS — Fast Drift Model (10 files trained)
# ============================================================
# Total windows: 58
#   Attack: 22
#   Normal: 36
#
# Accuracy:  100.00%
# Precision: 100.00%
# Recall:    100.00%
# F1-score:  100.00%
#
# Confusion Matrix:
#   TP: 22  TN: 36  FP: 0  FN: 0
# ═══════════════════════════════════════════════════════════════
