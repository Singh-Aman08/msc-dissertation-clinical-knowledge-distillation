

import torch
import gc
import os
import shutil
import json

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096

# Previous hyperparameters
LEARNING_RATE = 0.00002
LORA_RANK = 8
LORA_ALPHA = 16
EPOCHS = 5

OUTPUT_DIR = "model_sft_1"


dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_sft_train.json",
        "validation": "kbg_sft_val.json"
    }
)

print(dataset)


if __name__ == "__main__":

    model = None
    trainer = None
    tokenizer = None

    try:


        use_bf16 = torch.cuda.is_bf16_supported()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            load_in_4bit=False
        )

        print("\nModel loaded successfully.")


        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen-2.5"
        )

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


        train_dataset = dataset["train"].map(
            formatting_func,
            batched=True,
            load_from_cache_file=False
        )

        val_dataset = dataset["validation"].map(
            formatting_func,
            batched=True,
            load_from_cache_file=False
        )

        print("\nDatasets formatted successfully.")
        print("Training examples:", len(train_dataset))
        print("Validation examples:", len(val_dataset))


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


        print("\nLoRA configuration:")
        print("LoRA rank:", LORA_RANK)
        print("LoRA alpha:", LORA_ALPHA)

        class LossHistoryCallback(TrainerCallback):

            def __init__(self):
                self.training_losses = []
                self.validation_losses = []

            def on_log(
                self,
                args,
                state,
                control,
                logs=None,
                **kwargs
            ):

                if logs is None:
                    return

                # Training loss
                if "loss" in logs:

                    self.training_losses.append({
                        "step": int(state.global_step),
                        "epoch": float(state.epoch),
                        "loss": float(logs["loss"])
                    })

                # Validation loss
                if "eval_loss" in logs:

                    self.validation_losses.append({
                        "step": int(state.global_step),
                        "epoch": float(state.epoch),
                        "loss": float(logs["eval_loss"])
                    })


        loss_callback = LossHistoryCallback()


        trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

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

    
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_8bit",

        

        eval_strategy="epoch",

        save_strategy="epoch",     
        save_total_limit=None,

        logging_strategy="steps",
        logging_steps=10,

        # TensorBoard
        report_to="tensorboard",
        logging_dir=f"normal_sft/tensorboard",

    )
)

        trainer.add_callback(loss_callback)

        print("\n")
        print("=" * 60)
        print("PREVIOUS SFT CONFIGURATION")
        print("=" * 60)

        print(f"Learning rate : {LEARNING_RATE}")
        print(f"LoRA rank     : {LORA_RANK}")
        print(f"LoRA alpha    : {LORA_ALPHA}")
        print(f"Epochs        : {EPOCHS}")
        print(f"Batch size    : 2")
        print(f"Gradient accum: 4")

        print("=" * 60)
        print("\nStarting training...\n")

        trainer.train()


        evaluation = trainer.evaluate()

        final_val_loss = evaluation["eval_loss"]


        print("\n")
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(f"Final validation loss: {final_val_loss:.4f}")

        print("\nTraining losses:")
        for item in loss_callback.training_losses:
            print(
                f"Epoch {item['epoch']:.2f} | "
                f"Step {item['step']} | "
                f"Loss {item['loss']:.4f}"
            )

        print("\nValidation losses:")
        for item in loss_callback.validation_losses:
            print(
                f"Epoch {item['epoch']:.2f} | "
                f"Step {item['step']} | "
                f"Loss {item['loss']:.4f}"
            )

        results = {
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "epochs": EPOCHS,
                "batch_size": 2,
                "gradient_accumulation_steps": 4
            },

            "training_loss": loss_callback.training_losses,

            "validation_loss": loss_callback.validation_losses,

            "final_validation_loss": final_val_loss
        }


        with open(
            "previous_sft_loss_history.json",
            "w"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )


        print("\n")
        print("=" * 60)
        print("LOSS HISTORY SAVED")
        print("=" * 60)

        print(
            "File: previous_sft_loss_history.json"
        )


    finally:

        if model is not None:
            del model

        if trainer is not None:
            del trainer

        if tokenizer is not None:
            del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nGPU memory cleaned.")