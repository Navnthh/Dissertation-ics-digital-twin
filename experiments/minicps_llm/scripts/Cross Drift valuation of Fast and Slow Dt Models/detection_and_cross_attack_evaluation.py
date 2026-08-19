# ═══════════════════════════════════════════════════════════════
# DETECTION FUNCTION + CROSS-ATTACK EVALUATION
# (exact Colab code used to produce the cross-attack matrix)
# ═══════════════════════════════════════════════════════════════

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import os

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SLOW_LORA = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_ANSWER_ONLY_ULTRA"
FAST_LORA = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_FAST10"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def detect(model, window_df, source_name="test.csv", start_cycle=0):
    csv_lines = ["step,cycle,S1,S2,S3,A1"]
    for idx, row in enumerate(window_df.itertuples()):
        csv_lines.append(
            f"{idx},{start_cycle+idx},{row.SENSOR1_SENSED:.4f},"
            f"{row.SENSOR2_SENSED:.4f},{row.SENSOR3_SENSED:.4f},{int(row.ACTUATOR1)}"
        )
    csv_data = "\n".join(csv_lines)

    prompt = f"""<|im_start|>system
You are an ICS cyber-physical anomaly detector. Classify MiniCPS sensor windows as Normal or Attack. Use only controller-visible values. Return valid JSON only.
<|im_end|>
<|im_start|>user
Classify this MiniCPS sensor window.

Signals: S1=tank sensed level, S2=flow sensed value, S3=bottle/tank sensed level, A1=valve state. A stealthy attack may slowly manipulate sensed telemetry while keeping values plausible.

Source: {source_name}
Cycles: {start_cycle} to {start_cycle + len(window_df) - 1}

Sensor CSV:
{csv_data}
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    first_line = response.strip().split('\n')[0].strip()
    if "attack" in first_line.lower():
        return "Attack"
    elif "normal" in first_line.lower():
        return "Normal"
    return "Unknown"


def evaluate_on_files(model, base_dir, files, window_size=30, use_pure_windows=True):
    results, labels = [], []
    for fname in files:
        df = pd.read_csv(os.path.join(base_dir, fname))
        for i in range(window_size, len(df), window_size):
            window = df.iloc[i - window_size:i]
            attack_pct = (window['attack'] == 1).sum() / len(window) * 100
            if use_pure_windows:
                if attack_pct == 100:
                    label = 'Attack'
                elif attack_pct == 0:
                    label = 'Normal'
                else:
                    continue
            else:
                label = 'Attack' if attack_pct > 30 else 'Normal'
            pred = detect(model, window, fname, i - window_size)
            results.append(pred)
            labels.append(label)

    TP = sum(1 for t, p in zip(labels, results) if t == 'Attack' and p == 'Attack')
    TN = sum(1 for t, p in zip(labels, results) if t == 'Normal' and p == 'Normal')
    FP = sum(1 for t, p in zip(labels, results) if t == 'Normal' and p == 'Attack')
    FN = sum(1 for t, p in zip(labels, results) if t == 'Attack' and p == 'Normal')

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total * 100 if total else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"total": total, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


# ═══════════════════════════════════════════════════════════════
# LOAD BOTH MODELS
# ═══════════════════════════════════════════════════════════════
print("Loading slow drift model...")
base1 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
slow_model = PeftModel.from_pretrained(base1, SLOW_LORA)
slow_model.eval()

print("Loading fast drift model...")
base2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
fast_model = PeftModel.from_pretrained(base2, FAST_LORA)
fast_model.eval()

print("Both models loaded!")

# ═══════════════════════════════════════════════════════════════
# CROSS-ATTACK EVALUATION MATRIX
# ═══════════════════════════════════════════════════════════════
MINICPS_PATH = "/content/drive/MyDrive/ics_llm_outputs/minicps_100_csv"
FAST_PATH    = "/content/fast_drift_files/"

slow_test_files = sorted(os.listdir(MINICPS_PATH))[80:]      # runs 081-100
fast_test_files = sorted(os.listdir(FAST_PATH))[8:]          # fast_run_009-010

print("\n" + "=" * 60)
print("1/4  Slow model → Slow drift data")
print("=" * 60)
r1 = evaluate_on_files(slow_model, MINICPS_PATH, slow_test_files)
print(r1)

print("\n" + "=" * 60)
print("2/4  Fast model → Fast drift data")
print("=" * 60)
r2 = evaluate_on_files(fast_model, FAST_PATH, fast_test_files)
print(r2)

print("\n" + "=" * 60)
print("3/4  Slow model → Fast drift data (cross-test)")
print("=" * 60)
r3 = evaluate_on_files(slow_model, FAST_PATH, fast_test_files)
print(r3)

print("\n" + "=" * 60)
print("4/4  Fast model → Slow drift data (cross-test)")
print("=" * 60)
r4 = evaluate_on_files(fast_model, MINICPS_PATH, slow_test_files)
print(r4)

print("\n" + "=" * 60)
print("CROSS-ATTACK EVALUATION MATRIX SUMMARY")
print("=" * 60)
print(f"Slow model → Slow drift:  {r1['accuracy']:.2f}%  (own attack)")
print(f"Fast model → Fast drift:  {r2['accuracy']:.2f}%  (own attack)")
print(f"Slow model → Fast drift:  {r3['accuracy']:.2f}%  (cross)")
print(f"Fast model → Slow drift:  {r4['accuracy']:.2f}%  (cross)")

# ═══════════════════════════════════════════════════════════════
# ACTUAL RESULTS OBTAINED:
#   Slow model → Slow drift:  98.18%  (TP=912 TN=816 FP=3  FN=29)
#   Fast model → Fast drift:  100.00% (TP=22  TN=36  FP=0  FN=0)
#   Slow model → Fast drift:  37.93%  (TP=22  TN=0   FP=36 FN=0)
#   Fast model → Slow drift:  51.22%  (TP=18  TN=3   FP=19 FN=1)
# ═══════════════════════════════════════════════════════════════
