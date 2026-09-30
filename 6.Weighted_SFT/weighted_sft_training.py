import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1' 
import torch
import gc
import json
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 0.00002
LORA_RANK = 8
LORA_ALPHA = 16
EPOCHS = 5
OUTPUT_DIR = "weighted_sft_qwen25_1.5b"

dataset = load_dataset("json", data_files={"train": "kbg_weighted_sft_train.json", "validation": "kbg_weighted_sft_val.json"})
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


        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

        def tokenization_mapping_func(example):
            full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(full_text, max_length=MAX_SEQ_LENGTH, truncation=True)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = list(input_ids)
            prompt_messages = example["messages"][:-1]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_len = len(tokenizer(prompt_text)["input_ids"])
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "quality_weight": float(example["quality_weight"])}

        train_dataset = dataset["train"].map(tokenization_mapping_func, remove_columns=dataset["train"].column_names, load_from_cache_file=False)
        val_dataset = dataset["validation"].map(tokenization_mapping_func, remove_columns=dataset["validation"].column_names, load_from_cache_file=False)
        print("\nDatasets tokenized successfully.")
        print("Training examples:", len(train_dataset))
        print("Validation examples:", len(val_dataset))

    
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42
        )
        print("\nLoRA configuration loaded.")

        class SingleWeightDataCollator(DataCollatorForSeq2Seq):
            def __call__(self, features, return_tensors=None):
                quality_weights = [f.pop("quality_weight") for f in features]
                batch = super().__call__(features, return_tensors=return_tensors)
                batch["quality_weight"] = torch.tensor(quality_weights, dtype=torch.float32)
                return batch

        custom_collator = SingleWeightDataCollator(tokenizer=tokenizer, model=model)

        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                quality_weights = inputs.pop("quality_weight", None)
                labels = inputs.get("labels")

                outputs = model(**inputs)
                
                if labels is None or quality_weights is None:
                    loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
                    return (loss, outputs) if return_outputs else loss

                logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
                
                try:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    batch_size, seq_len = shift_labels.shape[0], shift_labels.shape[1]
                    
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    
                    raw_losses = loss_fct(flat_logits, flat_labels)
                    reshaped_losses = raw_losses.view(batch_size, seq_len)
                except (NotImplementedError, AttributeError):
    
                    loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
                    return (loss, outputs) if return_outputs else loss

                quality_weights = quality_weights.to(device=reshaped_losses.device, dtype=torch.float32)
                batch_mean = quality_weights.mean()
                normalized_weights = quality_weights / batch_mean if batch_mean > 0 else quality_weights

                loss_mask = (shift_labels != -100).float()
                weighted_losses = reshaped_losses * normalized_weights.unsqueeze(1)
                
                numerator = (weighted_losses * loss_mask).sum()
                denominator = loss_mask.sum()
                loss = numerator / (denominator + 1e-8)

                return (loss, outputs) if return_outputs else loss


        class LossHistoryCallback(TrainerCallback):
            def __init__(self):
                self.training_losses = []
                self.validation_losses = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                if "loss" in logs:
                    self.training_losses.append({"step": int(state.global_step), "epoch": float(state.epoch), "loss": float(logs["loss"])})
                if "eval_loss" in logs:
                    self.validation_losses.append({"step": int(state.global_step), "epoch": float(state.epoch), "loss": float(logs["eval_loss"])})

        loss_callback = LossHistoryCallback()
        
        trainer = WeightedTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=custom_collator,
            args=TrainingArguments(
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
                report_to="tensorboard",
                logging_dir="weighted_sft/tensorboard",
                remove_unused_columns=False
            )
        )
        trainer.add_callback(loss_callback)

        print("\n" + "="*60)
        print("WEIGHTED SFT CONFIGURATION")
        print("="*60)
        print(f"Learning rate        : {LEARNING_RATE}")
        print(f"LoRA rank            : {LORA_RANK}")
        print(f"LoRA alpha           : {LORA_ALPHA}")
        print(f"Epochs               : {EPOCHS}")
        print(f"Training examples    : {len(train_dataset)}")
        print(f"Validation examples  : {len(val_dataset)}")
        print("Batch size           : 2")
        print("Gradient accumulation: 4")
        print("="*60)
        print("\nStarting weighted SFT training...\n")

        trainer.train()

        evaluation = trainer.evaluate()
        final_val_loss = evaluation["eval_loss"]

        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("\n" + "="*60)
        print("WEIGHTED SFT TRAINING COMPLETE")
        print("="*60)
        print(f"Final validation loss: {final_val_loss:.4f}")
        print("\nTraining losses:")
        for item in loss_callback.training_losses:
            print(f"Epoch {item['epoch']:.2f} | Step {item['step']} | Loss {item['loss']:.4f}")

        print("\nValidation losses:")
        for item in loss_callback.validation_losses:
            print(f"Epoch {item['epoch']:.2f} | Step {item['step']} | Loss {item['loss']:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    finally:
        # CLEAN GPU MEMORY
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