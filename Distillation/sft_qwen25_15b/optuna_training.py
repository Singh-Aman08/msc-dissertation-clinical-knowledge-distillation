import optuna
import torch
import gc
import os
import shutil
import json

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
max_seq_length = 4096

dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_sft_train.json",
        "validation": "kbg_sft_val.json"
    }
)

print(dataset)


def objective(trial):
    learning_rate = trial.suggest_categorical(
        "learning_rate",
        [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    )
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    epochs = trial.suggest_categorical("epochs", [2, 3, 5, 7])

    print("\n==============================")
    print(f"Trial {trial.number}")
    print(f"LR={learning_rate}, Rank={lora_rank}, Alpha={lora_alpha}, Epochs={epochs}")
    print("==============================")

    model = None
    trainer = None
    tokenizer = None
    val_loss = None # Safe variable initialization to decouple from the cleanup block

    try:
        # Fresh model for every trial
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False
        )

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

        # Keep these map transformations cleanly local to the trial loop
        train_dataset = dataset["train"].map(formatting_func, batched=True, load_from_cache_file=False)
        val_dataset = dataset["validation"].map(formatting_func, batched=True, load_from_cache_file=False)

        # LoRA adapter mapping
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42
        )

        output_dir = f"optuna_trial_{trial.number}"

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,  # Reverted parameter name to standard tokenizer for TRL stability
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=SFTConfig(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                warmup_ratio=0.05,
                lr_scheduler_type="cosine",
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                optim="adamw_8bit",
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=10,
                report_to="none"
            )
        )

        trainer.train()
        evaluation = trainer.evaluate()
        val_loss = evaluation["eval_loss"]

        print(f"Trial {trial.number} validation loss: {val_loss}")

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise e  # Ensures Optuna registers a failed trial correctly
        
    finally:
        # Always clean memory, even if a runtime error breaks execution
        del model
        del trainer
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

        output_dir = f"optuna_trial_{trial.number}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    return val_loss # Returned safely here outside of the active memory purging context


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\n================================")
    print("OPTUNA COMPLETE")
    print("================================")

    print("Best parameters:")
    print(study.best_params)

    print("Best validation loss:", study.best_value)

    with open("best_optuna_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    print("Saved best parameters.")
