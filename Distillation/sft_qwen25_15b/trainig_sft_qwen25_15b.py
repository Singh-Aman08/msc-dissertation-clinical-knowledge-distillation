import torch
import os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096

# Best hyperparameters from Optuna
LEARNING_RATE = 2e-4
LORA_RANK = 8
LORA_ALPHA = 64
EPOCHS = 5

# Directory where the trained LoRA adapter will be saved
OUTPUT_DIR = "qwen25_1.5b_normal_sft_final"


# ============================================================
# LOAD 400 TRAIN + 100 VALIDATION
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_sft_train.json",
        "validation": "kbg_sft_val.json"
    }
)

print("\nOriginal dataset:")
print(dataset)


# ============================================================
# COMBINE INTO 500 TRAINING EXAMPLES
# ============================================================

train_dataset = concatenate_datasets(
    [
        dataset["train"],
        dataset["validation"]
    ]
)

print("\n==========================================")
print("FINAL TRAINING DATASET")
print("==========================================")
print(f"Number of examples: {len(train_dataset)}")


# ============================================================
# LOAD BASE MODEL
# ============================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5"
)


# ============================================================
# FORMAT CHAT DATA
# ============================================================

def formatting_func(examples):

    conversations = examples["messages"]

    texts = []

    for conversation in conversations:

        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        texts.append(text)

    return {"text": texts}


train_dataset = train_dataset.map(
    formatting_func,
    batched=True,
    load_from_cache_file=False
)

print("\nFormatted dataset:")
print(train_dataset)


# ============================================================
# ADD LoRA
# ============================================================

model = FastLanguageModel.get_peft_model(
    model,

    r=LORA_RANK,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],

    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",

    use_gradient_checkpointing="unsloth",

    random_state=42
)


# ============================================================
# CREATE TRAINER
# ============================================================

trainer = SFTTrainer(

    model=model,

    tokenizer=tokenizer,

    train_dataset=train_dataset,

    dataset_text_field="text",

    max_seq_length=MAX_SEQ_LENGTH,

    args=SFTConfig(

        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=2,

        gradient_accumulation_steps=4,

        learning_rate=LEARNING_RATE,

        warmup_ratio=0.05,

        lr_scheduler_type="cosine",

        bf16=torch.cuda.is_bf16_supported(),

        fp16=not torch.cuda.is_bf16_supported(),

        optim="adamw_8bit",

        save_strategy="epoch",

        logging_steps=10,

        report_to="none"
    )
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\n==========================================")
print("FINAL NORMAL SFT TRAINING")
print("==========================================")

print(f"Base model        : {MODEL_NAME}")
print(f"Training examples : {len(train_dataset)}")
print(f"Learning rate     : {LEARNING_RATE}")
print(f"LoRA rank         : {LORA_RANK}")
print(f"LoRA alpha        : {LORA_ALPHA}")
print(f"Epochs            : {EPOCHS}")
print(f"Output directory  : {OUTPUT_DIR}")

print("==========================================\n")


# ============================================================
# TRAIN
# ============================================================

trainer.train()


# ============================================================
# SAVE LoRA ADAPTER
# ============================================================

print("\n==========================================")
print("SAVING FINAL MODEL")
print("==========================================")

trainer.save_model(OUTPUT_DIR)

# Save tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nFinal LoRA model saved to:")
print(OUTPUT_DIR)

print("\nTraining complete.")