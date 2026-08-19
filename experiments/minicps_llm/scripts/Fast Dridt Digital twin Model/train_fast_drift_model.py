# ═══════════════════════════════════════════════════════════════
# FAST DRIFT MODEL — Training Script (exact Colab code used)
# Model saved as: minicps_qwen_lora_FAST10
# Trained on: train_fast10.jsonl (680 windows, 8 fast-drift files)
# ═══════════════════════════════════════════════════════════════

import os
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
TRAIN_FILE = "/content/drive/MyDrive/ics_llm_outputs/data/finetune_fast10/train_fast10.jsonl"
LOCAL_OUTPUT_DIR = "/content/qwen_fast10_local"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/ics_llm_outputs/minicps_qwen_lora_FAST10"
MAX_LENGTH = 512

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

if os.path.exists(LOCAL_OUTPUT_DIR): shutil.rmtree(LOCAL_OUTPUT_DIR)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files={"train": TRAIN_FILE})
print(f"Training examples: {len(dataset['train'])}")

def parse_label(text):
    try:
        pred = json.loads(text).get("prediction","").strip().lower()
        if pred == "attack": return "Attack"
        if pred == "normal": return "Normal"
    except: pass
    if "attack" in str(text).lower(): return "Attack"
    return "Normal"

labels = [parse_label(e["messages"][-1]["content"]) for e in dataset["train"]]
print("Label counts:", dict(Counter(labels)))

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

    available = MAX_LENGTH - len(answer_ids)
    if len(prompt_ids) > available:
        prompt_ids = prompt_ids[-available:]

    input_ids = prompt_ids + answer_ids
    labels_out = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels_out
    }

tokenized = dataset.map(tokenize_answer_only, remove_columns=dataset["train"].column_names)
lengths = [len(x["input_ids"]) for x in tokenized["train"]]
print(f"Tokens — min:{min(lengths)} max:{max(lengths)} avg:{sum(lengths)/len(lengths):.0f}")

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=LOCAL_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=100,
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

print("\nTraining fast drift model (10 files)...")
trainer.train()

trainer.save_model(LOCAL_OUTPUT_DIR)
tokenizer.save_pretrained(LOCAL_OUTPUT_DIR)

if os.path.exists(DRIVE_OUTPUT_DIR): shutil.rmtree(DRIVE_OUTPUT_DIR)
shutil.copytree(LOCAL_OUTPUT_DIR, DRIVE_OUTPUT_DIR)

print("\nDone! Files saved:")
for f in os.listdir(DRIVE_OUTPUT_DIR):
    print(f"  {f}")

# RESULT: Accuracy 100.00%, Precision 100.00%, Recall 100.00%, F1 100.00%
# Test set: 58 windows (TP=22 TN=36 FP=0 FN=0)
