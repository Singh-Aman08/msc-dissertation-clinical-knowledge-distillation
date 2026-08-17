import os
import gc
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"

MAX_SEQ_LENGTH = 4096

OUTPUT_DIR = "qwen25_1.5b_weighted_sft_final"

LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 64
EPOCHS = 7


# ============================================================
# LOAD TRAIN + VALIDATION DATA
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_weighted_sft_train.json",
        "validation": "kbg_weighted_sft_val.json",
    }
)

print(dataset)

# Combine 400 train + 100 validation = 500 examples
full_dataset = concatenate_datasets(
    [
        dataset["train"],
        dataset["validation"],
    ]
)

print("\nFull training dataset:")
print(full_dataset)

print("Number of examples:", len(full_dataset))


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):

        # ----------------------------------------------------
        # Extract quality weights
        # ----------------------------------------------------

        quality_weights = inputs.pop(
            "quality_weight",
            None
        )

        # Fallback to normal HF loss
        if quality_weights is None:

            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )

        quality_weights = quality_weights.to(
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Normalize weights
        # ----------------------------------------------------

        batch_mean = quality_weights.mean()

        if batch_mean > 0:

            normalized_weights = (
                quality_weights / batch_mean
            )

        else:

            normalized_weights = quality_weights


        # ====================================================
        # GET INPUTS
        # ====================================================

        input_ids = inputs["input_ids"]

        attention_mask = inputs["attention_mask"]

        labels = inputs.get("labels")


        # ====================================================
        # GET UNDERLYING QWEN MODEL
        # ====================================================

        # model = PEFT/LoRA model
        #
        # get_base_model() -> Qwen CausalLM
        #
        # base_model.model -> Qwen transformer
        #
        # transformer -> hidden states
        #
        # lm_head -> vocabulary logits

        base_model = model.get_base_model()

        transformer = base_model.model


        # ====================================================
        # FORWARD THROUGH TRANSFORMER
        # ====================================================

        transformer_outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )


        # [batch, sequence, hidden_size]
        hidden_states = (
            transformer_outputs.last_hidden_state
        )


        # ====================================================
        # HIDDEN STATES -> LOGITS
        # ====================================================

        # [batch, sequence, hidden_size]
        #
        #             ↓ lm_head
        #
        # [batch, sequence, vocab_size]

        logits = base_model.lm_head(
            hidden_states
        )


        # ====================================================
        # TOKEN-LEVEL CROSS ENTROPY
        # ====================================================

        if labels is not None:

            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none"
            )


            # ------------------------------------------------
            # Causal language-model shift
            # ------------------------------------------------

            shift_logits = (
                logits[..., :-1, :]
                .contiguous()
            )

            shift_labels = (
                labels[..., 1:]
                .contiguous()
            )


            batch_size = shift_labels.shape[0]

            seq_len = shift_labels.shape[1]


            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------

            flat_logits = shift_logits.view(
                -1,
                shift_logits.size(-1)
            )

            flat_labels = shift_labels.view(-1)


            # ------------------------------------------------
            # Ignore masked tokens
            # ------------------------------------------------

            loss_mask = (
                flat_labels != -100
            )


            # ------------------------------------------------
            # Individual token losses
            # ------------------------------------------------

            raw_losses = loss_fct(
                flat_logits,
                flat_labels
            )


            # [batch, sequence]
            reshaped_losses = (
                raw_losses.view(
                    batch_size,
                    seq_len
                )
            )


            reshaped_mask = (
                loss_mask.view(
                    batch_size,
                    seq_len
                )
            )


            # =================================================
            # APPLY EXAMPLE-LEVEL QUALITY WEIGHTS
            # =================================================

            weighted_losses = (
                reshaped_losses
                * normalized_weights.unsqueeze(1)
            )


            # ------------------------------------------------
            # Only valid target tokens
            # ------------------------------------------------

            numerator = weighted_losses[
                reshaped_mask
            ].sum()


            denominator = (
                reshaped_mask.sum()
            )


            loss = numerator / denominator


        else:

            loss = torch.tensor(
                0.0,
                device=hidden_states.device,
                requires_grad=True,
            )


        # ====================================================
        # RETURN
        # ====================================================

        if return_outputs:

            from transformers.modeling_outputs import (
                CausalLMOutputWithPast
            )

            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
            )

            return loss, outputs

        return loss


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model, tokenizer = (
    FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
)


# ============================================================
# CHAT TEMPLATE
# ============================================================

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenization_mapping_func(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


    tokenized = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
    )


    return {
        "input_ids": tokenized["input_ids"],

        "attention_mask": tokenized[
            "attention_mask"
        ],

        "labels": tokenized[
            "input_ids"
        ].copy(),

        "quality_weight": float(
            example["quality_weight"]
        ),
    }


print("\nTokenizing full 500-example dataset...")

train_dataset = full_dataset.map(
    tokenization_mapping_func,
    remove_columns=full_dataset.column_names,
)

print(train_dataset)

print(
    "Training examples:",
    len(train_dataset)
)


# ============================================================
# ADD LoRA
# ============================================================

print("\nAdding LoRA...")

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
        "down_proj",
    ],

    lora_alpha=LORA_ALPHA,

    lora_dropout=0,

    bias="none",

    use_gradient_checkpointing="unsloth",

    random_state=42,
)


# ============================================================
# DATA COLLATOR
# ============================================================

class SingleWeightDataCollator(
    DataCollatorForSeq2Seq
):

    def __call__(
        self,
        features,
        return_tensors=None,
    ):

        # Extract quality weights
        quality_weights = [
            f.pop("quality_weight")
            for f in features
        ]


        # Standard padding
        batch = super().__call__(
            features,
            return_tensors=return_tensors,
        )


        # Put weights back into batch
        batch["quality_weight"] = torch.tensor(
            quality_weights,
            dtype=torch.float32,
        )

        return batch


custom_collator = SingleWeightDataCollator(
    tokenizer=tokenizer,
    model=model,
)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    tokenizer=tokenizer,

    train_dataset=train_dataset,

    data_collator=custom_collator,

    args=TrainingArguments(

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

        report_to="none",

        remove_unused_columns=False,
    ),
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING FINAL WEIGHTED TRAINING")
print("========================================")

print(f"Examples       : {len(train_dataset)}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"LoRA rank      : {LORA_RANK}")
print(f"LoRA alpha     : {LORA_ALPHA}")
print(f"Epochs         : {EPOCHS}")

trainer.train()


# ============================================================
# SAVE FINAL LoRA MODEL
# ============================================================

print("\n========================================")
print("SAVING FINAL WEIGHTED MODEL")
print("========================================")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Final model saved to:",
    OUTPUT_DIR
)


# ============================================================
# CLEAN MEMORY
# ============================================================

del trainer
del model
del tokenizer

gc.collect()
torch.cuda.empty_cache()