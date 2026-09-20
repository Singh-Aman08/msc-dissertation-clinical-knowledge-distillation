import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WGHT_ADAPTER = "weighted_sft_qwen25_1.5b/checkpoint-50"

MAX_SEQ_LENGTH = 8192
MAX_PROMPT_LENGTH = 4096

LEARNING_RATE = 1e-7
BETA = 0.05
EPOCHS = 1

OUTPUT_DIR = "qwen25_1.5b_dpo_wght_conservative"
FINAL_OUTPUT = "qwen25_1.5b_dpo_wght_conservative_adapter"


class DPOConsolePrinterCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):

        if not logs:
            return

        if "loss" in logs:
            print(f"\n--- STEP {state.global_step} ---")
            print(f"Loss: {logs.get('loss', 'N/A')}")
            print(f"Chosen reward: {logs.get('rewards/chosen', 'N/A')}")
            print(f"Rejected reward: {logs.get('rewards/rejected', 'N/A')}")
            print(f"Reward margin: {logs.get('rewards/margins', 'N/A')}")

        if "eval_loss" in logs:
            print(f"\n>>> VALIDATION @ STEP {state.global_step}")
            print(f"Validation loss: {logs.get('eval_loss', 'N/A')}")
            print(
                f"Val chosen reward: "
                f"{logs.get('eval_rewards/chosen', 'N/A')}"
            )
            print(
                f"Val rejected reward: "
                f"{logs.get('eval_rewards/rejected', 'N/A')}"
            )
            print(
                f"Val reward margin: "
                f"{logs.get('eval_rewards/margins', 'N/A')}"
            )
            print("=" * 60)

print("Loading DPO preference datasets...")

dataset = load_dataset(
    "json",
    data_files={
        "train": "dpo_preference_pairs_train_450.jsonl",
        "validation": "dpo_preference_pairs_val_50.jsonl"
    }
)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

print(f"Training examples   : {len(train_dataset)}")
print(f"Validation examples : {len(val_dataset)}")

if len(train_dataset) != 450:
    raise ValueError(
        f"Expected 450 training examples, found {len(train_dataset)}"
    )

if len(val_dataset) != 50:
    raise ValueError(
        f"Expected 50 validation examples, found {len(val_dataset)}"
    )


required_columns = {"prompt", "chosen", "rejected"}

missing_columns = required_columns - set(train_dataset.column_names)

if missing_columns:
    raise ValueError(
        f"Missing required DPO columns: {missing_columns}"
    )

print("\nDPO columns:")
print(train_dataset.column_names)

print("\nExample lengths:")

example = train_dataset[0]

print(f"Prompt characters  : {len(example['prompt'])}")
print(f"Chosen characters  : {len(example['chosen'])}")
print(f"Rejected characters: {len(example['rejected'])}")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print(f"PAD token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

use_bf16 = torch.cuda.is_bf16_supported()

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"BF16 supported: {use_bf16}")

print("\nLoading active Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
)

model.config.use_cache = False

print("\nLoading SFT LoRA adapter:")
print(WGHT_ADAPTER)

model = PeftModel.from_pretrained(
    model,
    WGHT_ADAPTER,
    is_trainable=True
)

print("\nActive SFT model:")
model.print_trainable_parameters()


print("\nLoading frozen reference model...")

ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
)

ref_model.config.use_cache = False

ref_model = PeftModel.from_pretrained(
    ref_model,
    WGHT_ADAPTER,
    is_trainable=False
)

print("Frozen reference SFT model loaded.")

ref_model.print_trainable_parameters()


training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    learning_rate=LEARNING_RATE,
    beta=BETA,
    loss_type="sigmoid",
    num_train_epochs=EPOCHS,

    
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,


    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    
    bf16=use_bf16,
    fp16=not use_bf16,

    
    optim="adamw_8bit",

    gradient_checkpointing=True,

    
    logging_strategy="steps",
    logging_steps=5,

    
    eval_strategy="steps",
    eval_steps=10,

    save_strategy="steps",
    save_steps=10,

    report_to="none",

    remove_unused_columns=False,

    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    load_best_model_at_end=False
)


print("\nCreating DPOTrainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    callbacks=[DPOConsolePrinterCallback()]
)


print("\n" + "=" * 70)
print("DPO CONFIGURATION")
print("=" * 70)

print(f"Base model        : {BASE_MODEL}")
print(f"SFT adapter       : {WGHT_ADAPTER}")
print(f"Training examples : {len(train_dataset)}")
print(f"Validation        : {len(val_dataset)}")
print(f"Learning rate     : {LEARNING_RATE}")
print(f"Beta              : {BETA}")
print(f"Epochs            : {EPOCHS}")
print(f"Batch size        : 2")
print(f"Grad accumulation : 4")
print(f"Effective batch   : 8")
print(f"Max sequence      : {MAX_SEQ_LENGTH}")
print(f"Max prompt        : {MAX_PROMPT_LENGTH}")
print("=" * 70)


print("\nStarting DPO training...\n")

trainer.train()


print("\nDPO training completed.")

trainer.save_model(FINAL_OUTPUT)
tokenizer.save_pretrained(FINAL_OUTPUT)

print("\n" + "=" * 70)
print("DPO TRAINING COMPLETE")
print("=" * 70)

print(f"Final adapter saved to: {FINAL_OUTPUT}")
print("DONE.")
