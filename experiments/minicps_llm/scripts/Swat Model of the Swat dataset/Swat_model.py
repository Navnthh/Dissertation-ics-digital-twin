# ═══════════════════════════════════════════════════════════════
# SWaT MODEL — Data Preparation + Training (exact Colab code used)
# Model saved as: minicps_qwen_lora_SWAT_v3
# Source: "Swat 900 rows combined1(in).csv"
#   Rows 0-360:   LIT101 fast rising drift attack (361 rows)
#   Rows 361-643: Normal (283 rows)
#   Rows 644-899: LIT301 fast falling drift attack (256 rows)
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import json
import os
import random

df = pd.read_csv('/content/Swat 900 rows combined1(in).csv')
df.columns = df.columns.str.strip()

# ═══════════════════════════════════════════════════════════════
# SPLIT DATA
# ═══════════════════════════════════════════════════════════════
lit101_attack = df.iloc[0:361].reset_index(drop=True)
normal_data   = df.iloc[361:644].reset_index(drop=True)
lit301_attack = df.iloc[644:900].reset_index(drop=True)

print(f"LIT101 attack rows: {len(lit101_attack)}")
print(f"Normal rows: {len(normal_data)}")
print(f"LIT301 attack rows: {len(lit301_attack)}")

WINDOW_SIZE = 30
STEP = 10

SYSTEM = ("You are an ICS cyber-physical anomaly detector. "
          "Classify SWaT sensor windows as Normal or Attack. "
          "Use only controller-visible values. Return valid JSON only.")

USER_TEMPLATE = """Classify this SWaT sensor window.

Signals: LIT101=tank level(mm), FIT101=flow(m3/h), LIT301=bottle level(mm), MV101=valve, P101/P102=pumps.

Source: {source}
Cycles: {cycle_start} to {cycle_end}

Sensor CSV:
{csv_data}"""

def build_swat_example(window_df, source, cycle_start, cycle_end, label, attack_type):
    cols = ['FIT101','LIT101','MV101','P101','P102','DPIT301',
            'FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302']
    csv_lines = ["step," + ",".join(cols)]
    for idx, row in enumerate(window_df.itertuples()):
        vals = [f"{getattr(row, c):.4f}" if isinstance(getattr(row, c), float)
                else str(getattr(row, c)) for c in cols]
        csv_lines.append(f"{idx}," + ",".join(vals))
    csv_data = "\n".join(csv_lines)

    user_msg = USER_TEMPLATE.format(
        source=source, cycle_start=cycle_start,
        cycle_end=cycle_end, csv_data=csv_data
    )

    if label == 'Attack':
        assistant = json.dumps({
            "prediction": "Attack", "attack_type": attack_type,
            "reasoning": f"Sensor readings show {attack_type} inconsistent with normal SWaT operation."
        })
    else:
        assistant = json.dumps({
            "prediction": "Normal", "attack_type": "none",
            "reasoning": "Sensor readings follow expected SWaT normal operating pattern."
        })

    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant}
    ]}

# Build examples
examples = []

for i in range(WINDOW_SIZE, len(lit101_attack), STEP):
    window = lit101_attack.iloc[i-WINDOW_SIZE:i]
    if len(window) == WINDOW_SIZE:
        examples.append(build_swat_example(window, 'LIT101_fast.csv', i-WINDOW_SIZE, i-1, 'Attack', 'fast_rising_LIT101'))

for i in range(WINDOW_SIZE, len(lit301_attack), STEP):
    window = lit301_attack.iloc[i-WINDOW_SIZE:i]
    if len(window) == WINDOW_SIZE:
        examples.append(build_swat_example(window, 'LIT301_fast.csv', i-WINDOW_SIZE, i-1, 'Attack', 'fast_falling_LIT301'))

for i in range(WINDOW_SIZE, len(normal_data), STEP):
    window = normal_data.iloc[i-WINDOW_SIZE:i]
    if len(window) == WINDOW_SIZE:
        examples.append(build_swat_example(window, 'SWaT_normal.csv', i-WINDOW_SIZE, i-1, 'Normal', 'none'))

random.shuffle(examples)

attack_count = sum(1 for e in examples if json.loads(e['messages'][2]['content'])['prediction'] == 'Attack')
normal_count = len(examples) - attack_count
print(f"\nTotal examples: {len(examples)}")
print(f"Attack: {attack_count}, Normal: {normal_count}")

# ═══════════════════════════════════════════════════════════════
# BALANCE — downsample Attack to match Normal count
# (raw split was 71.2% attack; balancing fixed the "always
# predicts Attack" bias seen in the v1/v2 unbalanced runs)
# ═══════════════════════════════════════════════════════════════
attack_examples = [e for e in examples if json.loads(e['messages'][2]['content'])['prediction'] == 'Attack']
normal_examples = [e for e in examples if json.loads(e['messages'][2]['content'])['prediction'] == 'Normal']

n = len(normal_examples)
balanced = random.sample(attack_examples, n) + normal_examples
random.shuffle(balanced)

print(f"\nBalanced total: {len(balanced)} (Attack:{n}, Normal:{n})")

split = int(len(balanced) * 0.8)
train_data = balanced[:split]
test_data  = balanced[split:]

OUTPUT_DIR = "/content/drive/MyDrive/ics_llm_outputs/data/finetune_swat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(OUTPUT_DIR, "train_swat_balanced.jsonl")
test_path  = os.path.join(OUTPUT_DIR, "test_swat_balanced.jsonl")

with open(train_path, 'w') as f:
    for ex in train_data:
        f.write(json.dumps(ex) + '\n')

with open(test_path, 'w') as f:
    for ex in test_data:
        f.write(json.dumps(ex) + '\n')

print(f"\nTrain: {len(train_data)} → {train_path}")
print(f"Test:  {len(test_data)} → {test_path}")


# ═══════════════════════════════════════════════════════════════
# TRAIN MODEL (same LoRA config as slow/fast drift models)
# ═══════════════════════════════════════════════════════════════
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import shutil

DRIVE_OUTPUT = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_SWAT_v3"
LOCAL_OUTPUT = "/content/qwen_swat_v3_local"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 512

if os.path.exists(LOCAL_OUTPUT): shutil.rmtree(LOCAL_OUTPUT)
os.makedirs(LOCAL_OUTPUT, exist_ok=True)
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files={"train": train_path})
print(f"Balanced training examples: {len(dataset['train'])}")

def parse_label(text):
    try:
        pred = json.loads(text).get("prediction","").strip().lower()
        if pred == "attack": return "Attack"
        if pred == "normal": return "Normal"
    except: pass
    return "Attack" if "attack" in str(text).lower() else "Normal"

def tokenize_answer_only(example):
    messages = example["messages"]
    prompt_messages = [
        {"role": "system", "content": messages[0]["content"]},
        {"role": "user",   "content": messages[1]["content"]}
    ]
    gold_label = parse_label(messages[-1]["content"])
    answer_text = ('{"prediction":"Attack"}' if gold_label == "Attack"
                   else '{"prediction":"Normal"}') + tokenizer.eos_token

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    available_prompt_len = MAX_LENGTH - len(answer_ids)
    if len(prompt_ids) > available_prompt_len:
        prompt_ids = prompt_ids[-available_prompt_len:]

    input_ids = prompt_ids + answer_ids
    labels_out = [-100] * len(prompt_ids) + answer_ids

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels_out}

tokenized = dataset.map(tokenize_answer_only, remove_columns=dataset["train"].column_names)

class AnswerOnlyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
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

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=LOCAL_OUTPUT,
    num_train_epochs=3,             # more epochs since only 38 examples
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    save_steps=100,
    eval_strategy="no",
    fp16=True,
    report_to="none",
    warmup_steps=5,
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

print("\nTraining balanced SWaT v3 model...")
trainer.train()

trainer.save_model(LOCAL_OUTPUT)
tokenizer.save_pretrained(LOCAL_OUTPUT)

if os.path.exists(DRIVE_OUTPUT): shutil.rmtree(DRIVE_OUTPUT)
shutil.copytree(LOCAL_OUTPUT, DRIVE_OUTPUT)

print("\nDone!")
for f in os.listdir(DRIVE_OUTPUT):
    print(f"  {f}")

# RESULT: Accuracy 69.00%, Precision 68.97%, Recall 100.00%, F1 81.66%
# Test set: 29 windows (TP=20 TN=0 FP=9 FN=0)
# NOTE: 0% specificity — model flags all normal windows as Attack too.
# Reflects the small training set (38 examples); identified as a
# limitation / direction for future work in the dissertation.
