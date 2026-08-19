# ═══════════════════════════════════════════════════════════════
# SLOW DRIFT MODEL — Training Script (exact Colab code used)
# Model saved as: minicps_qwen_lora_ANSWER_ONLY_ULTRA
# Trained on: train_compact.jsonl (7,040 windows, 80 MiniCPS files)
# ═══════════════════════════════════════════════════════════════

import os
import re
import json
import shutil
import statistics
from collections import Counter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "/content/drive/MyDrive/ics_llm_outputs/data/finetune_compact/train_compact.jsonl"

LOCAL_OUTPUT_DIR = "/content/minicps_qwen_lora_FEATURE_SUMMARY_LOCAL"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_FEATURE_SUMMARY"

MAX_LENGTH = 512
SYSTEM_PROMPT = "You are an ICS anomaly detector. Output JSON only."


def parse_label_from_assistant(text):
    try:
        obj = json.loads(text)
        pred = str(obj.get("prediction", "")).strip().lower()
        if pred == "attack":
            return "Attack"
        if pred == "normal":
            return "Normal"
    except Exception:
        pass

    lower = str(text).lower()

    if '"prediction": "attack"' in lower or '"prediction":"attack"' in lower:
        return "Attack"

    if '"prediction": "normal"' in lower or '"prediction":"normal"' in lower:
        return "Normal"

    if "attack" in lower:
        return "Attack"

    return "Normal"


def extract_sensor_csv(user_text):
    match = re.search(
        r"Sensor CSV:\n(.*?)(?:\n\nReturn only JSON:|\Z)",
        user_text,
        flags=re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return user_text.strip()


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

    first = values[0]
    last = values[-1]

    return {
        "first": first,
        "last": last,
        "change": last - first,
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
            "Return exactly: {\"prediction\":\"Attack\"} or {\"prediction\":\"Normal\"}"
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

    prompt = f"""Classify this MiniCPS sensor window as Attack or Normal.

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

    return prompt


def target_json(label):
    if label == "Attack":
        return '{"prediction":"Attack"}'
    return '{"prediction":"Normal"}'


print("=" * 80)
print("Checking paths")
print("=" * 80)
print("TRAIN_FILE:", TRAIN_FILE)
print("TRAIN_FILE exists:", os.path.exists(TRAIN_FILE))

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"Missing training file: {TRAIN_FILE}")

if os.path.exists(LOCAL_OUTPUT_DIR):
    shutil.rmtree(LOCAL_OUTPUT_DIR)

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GPU check")
print("=" * 80)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Use Colab T4 GPU.")

print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

torch.cuda.empty_cache()

print("=" * 80)
print("Loading tokenizer")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("=" * 80)
print("Loading training dataset")
print("=" * 80)

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE}
)

print(dataset)

labels = [
    parse_label_from_assistant(example["messages"][-1]["content"])
    for example in dataset["train"]
]

print("Training label counts:", dict(Counter(labels)))


def tokenize_answer_only(example):
    messages = example["messages"]

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_feature_prompt(messages)}
    ]

    gold_label = parse_label_from_assistant(messages[-1]["content"])
    answer_text = target_json(gold_label) + tokenizer.eos_token

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False
    )["input_ids"]

    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False
    )["input_ids"]

    available_prompt_len = MAX_LENGTH - len(answer_ids)

    if len(prompt_ids) > available_prompt_len:
        prompt_ids = prompt_ids[-available_prompt_len:]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


print("=" * 80)
print("Tokenizing feature-summary answer-only dataset")
print("=" * 80)

tokenized = dataset.map(
    tokenize_answer_only,
    remove_columns=dataset["train"].column_names
)

lengths = [len(x["input_ids"]) for x in tokenized["train"]]

print(tokenized)
print("Min tokens:", min(lengths))
print("Max tokens:", max(lengths))
print("Average tokens:", sum(lengths) / len(lengths))


class AnswerOnlyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


print("=" * 80)
print("Loading base model in FP16")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

print("=" * 80)
print("Adding LoRA adapter")
print("=" * 80)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("=" * 80)
print("Training setup")
print("=" * 80)

training_args = TrainingArguments(
    output_dir=LOCAL_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=100,
    eval_strategy="no",
    save_strategy="steps",
    fp16=True,
    report_to="none",
    warmup_steps=50,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    save_total_limit=2,
    optim="adamw_torch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=AnswerOnlyCollator(tokenizer.pad_token_id),
)

print("=" * 80)
print("Starting feature-summary answer-only LoRA training")
print("=" * 80)

trainer.train()

print("=" * 80)
print("Saving adapter locally")
print("=" * 80)

trainer.save_model(LOCAL_OUTPUT_DIR)
tokenizer.save_pretrained(LOCAL_OUTPUT_DIR)

print("=" * 80)
print("Copying adapter to Google Drive")
print("=" * 80)

if os.path.exists(DRIVE_OUTPUT_DIR):
    shutil.rmtree(DRIVE_OUTPUT_DIR)

shutil.copytree(LOCAL_OUTPUT_DIR, DRIVE_OUTPUT_DIR)

print("=" * 80)
print("Drive saved files")
print("=" * 80)

for filename in os.listdir(DRIVE_OUTPUT_DIR):
    print(filename)

print("Training complete.")
print("Saved to:", DRIVE_OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════
# RESULT: Accuracy 98.18%, Precision 99.67%, Recall 96.92%, F1 98.28%
# Test set: 1,760 windows (TP=912 TN=816 FP=3 FN=29)
# ═══════════════════════════════════════════════════════════════
