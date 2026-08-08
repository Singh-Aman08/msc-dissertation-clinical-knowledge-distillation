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
from transformers import DataCollatorForSeq2Seq

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
max_seq_length = 4096

dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_weighted_sft_train.json",
        "validation": "kbg_weighted_sft_val.json"}
)
print(dataset)
class WeightedSFTTrainer(SFTTrainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        quality_weights = inputs.pop(
            "quality_weight",
            None
        )

        if quality_weights is None:

            return super().compute_loss(
                model,
                inputs,
                return_outputs,
                **kwargs
            )

        quality_weights = quality_weights.to(
            model.device
        )

        batch_mean = quality_weights.mean()


        if batch_mean > 0:

            normalized_weights = (
                quality_weights / batch_mean
            )

        else:

            normalized_weights = quality_weights


        outputs = model(**inputs)

        logits = outputs.logits

        labels = inputs.get("labels")


        if labels is not None:


            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none"
            )

            shift_logits = (
                logits[..., :-1, :]
                .contiguous()
            )

            shift_labels = (
                labels[..., 1:]
                .contiguous()
            )

            flat_logits = shift_logits.view(
                -1,
                shift_logits.size(-1)
            )

            flat_labels = shift_labels.view(-1)

            loss_mask = flat_labels != -100

            raw_losses = loss_fct(
                flat_logits,
                flat_labels
            )

            batch_size = labels.shape[0]


            reshaped_losses = raw_losses.view(
                batch_size,
                -1
            )


            reshaped_mask = loss_mask.view(
                batch_size,
                -1
            )

            weighted_losses = (
                reshaped_losses *
                normalized_weights.unsqueeze(1)
            )

            loss = (
                weighted_losses[reshaped_mask].sum()
                /
                reshaped_mask.sum()
            )

        else:

            loss = outputs.loss

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def objective(trial):

    learning_rate = trial.suggest_categorical(
        "learning_rate",
        [
            1e-5,
            2e-5,
            5e-5,
            1e-4,
            2e-4
        ]
    )


    lora_rank = trial.suggest_categorical(
        "lora_rank",
        [
            8,
            16,
            32
        ]
    )


    lora_alpha = trial.suggest_categorical(
        "lora_alpha",
        [
            16,
            32,
            64
        ]
    )


    epochs = trial.suggest_categorical(
        "epochs",
        [
            2,
            3,
            5,
            7
        ]
    )

    print("\n==============================")
    print(f"Trial {trial.number}")
    print(
        f"LR={learning_rate}, "
        f"Rank={lora_rank}, "
        f"Alpha={lora_alpha}, "
        f"Epochs={epochs}"
    )
    print("==============================")



    model = None
    trainer = None
    tokenizer = None
    val_loss = None
    output_dir = None


    try:


        # Fresh model for every Optuna trial

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
    
        
        
        def formatting_func(example):

            text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

            return {
        "text": text,
        "quality_weight": float(example["quality_weight"])
    }



        print("Formatting datasets...")
        
        print(dataset["train"][0]["messages"])
        print(type(dataset["train"][0]["messages"]))


        train_dataset = dataset["train"].map(
    formatting_func,
    remove_columns=dataset["train"].column_names
)

        val_dataset = dataset["validation"].map(
    formatting_func,
    remove_columns=dataset["validation"].column_names
)
        
        print(train_dataset.column_names)
        print(val_dataset.column_names)


        model = FastLanguageModel.get_peft_model(

            model,

            r=lora_rank,

            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],

            lora_alpha=lora_alpha,

            lora_dropout=0,

            bias="none",

            use_gradient_checkpointing="unsloth",

            random_state=42

        )



        output_dir = (
            f"weighted_optuna_trial_{trial.number}"
        )


        class SingleWeightDataCollator(DataCollatorForSeq2Seq):
            def __call__(self, features, return_tensors=None):
                # 1. Safely extract the custom report weights
                quality_weights = [
                    f.pop("quality_weight")
                    for f in features
                ]

                # 2. CRUCIAL FIX: Strip out the raw untokenized "text" string 
                # This stops transformers from trying to pad raw text sentences into tensors!
                for f in features:
                    if "text" in f:
                        del f["text"]

                # 3. Now the parent class can pad the input_ids and labels safely
                batch = super().__call__(
                    features,
                    return_tensors=return_tensors
                )

                # 4. Bind the quality weights tensor to the batch dictionary
                batch["quality_weight"] = torch.tensor(
                    quality_weights,
                    dtype=torch.bfloat16
                )

                return batch





        custom_collator = SingleWeightDataCollator(
            tokenizer=tokenizer,
            model=model
        )


        trainer = WeightedSFTTrainer(

            model=model,

            tokenizer=tokenizer,

            train_dataset=train_dataset,

            eval_dataset=val_dataset,

            dataset_text_field="text",

            data_collator=custom_collator,

            max_seq_length=max_seq_length,


            args=SFTConfig(

                output_dir=output_dir,


                num_train_epochs=epochs,


                per_device_train_batch_size=2,


                gradient_accumulation_steps=4,


                learning_rate=learning_rate,
                packing=False,


                warmup_ratio=0.05,


                lr_scheduler_type="cosine",


                bf16=torch.cuda.is_bf16_supported(),


                fp16=not torch.cuda.is_bf16_supported(),


                optim="adamw_8bit",


                eval_strategy="epoch",


                save_strategy="no",


                logging_steps=10,


                report_to="none",


                # Important:
                # keeps quality_weight column
                # available for collator

                remove_unused_columns=False

            )

        )
        
        
        print(train_dataset[0].keys())
        print(type(train_dataset[0]["text"]))


        trainer.train()



        evaluation = trainer.evaluate()


        val_loss = evaluation["eval_loss"]



        print(
            f"Trial {trial.number} "
            f"validation loss: {val_loss}"
        )



    except Exception as e:


        print(
            f"Trial {trial.number} failed: {e}"
        )

        raise e



    finally:

        if model is not None:

            del model


        if trainer is not None:

            del trainer


        if tokenizer is not None:

            del tokenizer



        gc.collect()

        torch.cuda.empty_cache()



        if output_dir is not None and os.path.exists(output_dir):

            shutil.rmtree(output_dir)



    return val_loss


if __name__ == "__main__":


    study = optuna.create_study(

        direction="minimize"

    )


    study.optimize(

        objective,

        n_trials=2

    )



    print(
        "\n================================"
    )

    print(
        "WEIGHTED OPTUNA SWEEP COMPLETE"
    )

    print(
        "================================"
    )



    print(
        "Best parameters:"
    )


    print(
        study.best_params
    )


    print(
        "Best validation loss:",
        study.best_value
    )



    with open(
        "best_weighted_optuna_params.json",
        "w"
    ) as f:


        json.dump(
            study.best_params,
            f,
            indent=4
        )



    print(
        "Saved best weighted Optuna parameters."
    )
    
    
############################################################################## 
import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

print("LOGITS FLAG:", os.environ.get("UNSLOTH_RETURN_LOGITS"))

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
from transformers import DataCollatorForSeq2Seq



MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
max_seq_length = 4096

dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_weighted_sft_train.json",
        "validation": "kbg_weighted_sft_val.json"}
)
print(dataset)

class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        quality_weights = inputs.pop("quality_weight", None)

        if quality_weights is None:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        quality_weights = quality_weights.to(model.device)
        batch_mean = quality_weights.mean()

        if batch_mean > 0:
            normalized_weights = quality_weights / batch_mean
        else:
            normalized_weights = quality_weights

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            loss_mask = flat_labels != -100

            raw_losses = loss_fct(flat_logits, flat_labels)

            batch_size = labels.shape[0]
            seq_len_minus_1 = shift_labels.shape[1] # Robustly dynamic sequence calculation

            reshaped_losses = raw_losses.view(batch_size, seq_len_minus_1)
            reshaped_mask = loss_mask.view(batch_size, seq_len_minus_1)

            weighted_losses = reshaped_losses * normalized_weights.unsqueeze(1)

            loss = weighted_losses[reshaped_mask].sum() / reshaped_mask.sum()
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 5e-5, 1e-4, 2e-4])
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
    val_loss = None
    output_dir = f"weighted_optuna_trial_{trial.number}"

    try:
        # Fresh model for every Optuna trial
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False
        )

        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    
        # FIXED: Directly tokenize strings to integer lists inside mapping, avoiding SFT text injection errors
        def tokenization_mapping_func(example):
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            tokenized = tokenizer(text, max_length=max_seq_length, truncation=True)
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy(), # Ground-truth labels duplicate input tracking IDs
                "quality_weight": float(example["quality_weight"])
            }

        print("Formatting and tokenizing datasets explicitly...")
        train_dataset = dataset["train"].map(tokenization_mapping_func, remove_columns=dataset["train"].column_names)
        val_dataset = dataset["validation"].map(tokenization_mapping_func, remove_columns=dataset["validation"].column_names)

                # Existing LoRA initialization block
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

        # =====================================================================
        # CRUCIAL FIX: MANUALLY FORCE UNSLOTH TO RETAIN LOGITS IN MEMORY
        # =====================================================================
        if hasattr(model, "config"):
            model.config.return_dict = True


        class SingleWeightDataCollator(DataCollatorForSeq2Seq):
            def __call__(self, features, return_tensors=None):
                quality_weights = [f.pop("quality_weight") for f in features]
                
                # Super call executes standard padding metrics cleanly on integer lists
                batch = super().__call__(features, return_tensors=return_tensors)
                
                batch["quality_weight"] = torch.tensor(quality_weights, dtype=torch.bfloat16)
                return batch

        custom_collator = SingleWeightDataCollator(tokenizer=tokenizer, model=model)

        trainer = WeightedSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=custom_collator, # Swapped out dataset_text_field for complete manual collation control
            max_seq_length=max_seq_length,
            args=SFTConfig(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                packing=False,
                warmup_ratio=0.05,
                lr_scheduler_type="cosine",
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                optim="adamw_8bit",
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=10,
                report_to="none",
                remove_unused_columns=False
            )
        )
    

        trainer.train()
        evaluation = trainer.evaluate()
        val_loss = evaluation["eval_loss"]

        print(f"Trial {trial.number} validation loss: {val_loss}")

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise e

    finally:
        if model is not None:
            del model
        if trainer is not None:
            del trainer
        if tokenizer is not None:
            del tokenizer
            
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\n================================")
    print("WEIGHTED OPTUNA SWEEP COMPLETE")
    print("================================")
    print("Best parameters:", study.best_params)
    print("Best validation loss:", study.best_value)

    with open("best_weighted_optuna_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

########################################

import os

# Keep this enabled, although we do not rely on Unsloth returning logits.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

print(
    "LOGITS FLAG VERIFIED:",
    os.environ.get("UNSLOTH_RETURN_LOGITS")
)

import gc
import json
import shutil

import optuna
import torch

from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096



dataset = load_dataset(
    "json",
    data_files={
        "train": "kbg_weighted_sft_train.json",
        "validation": "kbg_weighted_sft_val.json",
    },
)

print(dataset)


class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):

        quality_weights = inputs.pop("quality_weight", None)

        # If no quality weights are present, use normal HF loss.
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

        # Normalize weights so their mean within the batch is 1.
        batch_mean = quality_weights.mean()

        if batch_mean > 0:
            normalized_weights = quality_weights / batch_mean
        else:
            normalized_weights = quality_weights

        # --------------------------------------------------------
        # Get inputs
        # --------------------------------------------------------

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels")


        base_model = model.get_base_model()

        # Qwen2.5 CausalLM -> transformer backbone
        transformer = base_model.model

        transformer_outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = transformer_outputs.last_hidden_state

        # Apply the original Qwen LM head.
        # This produces the actual vocabulary logits.
        logits = base_model.lm_head(hidden_states)

        if labels is not None:

            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none"
            )

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            batch_size = shift_labels.shape[0]
            seq_len = shift_labels.shape[1]

            # Flatten for CrossEntropyLoss
            flat_logits = shift_logits.view(
                -1,
                shift_logits.size(-1)
            )

            flat_labels = shift_labels.view(-1)

            # Ignore padding / masked tokens
            loss_mask = flat_labels != -100

            # Individual TOKEN losses
            raw_losses = loss_fct(
                flat_logits,
                flat_labels
            )

            # Back into:
            #
            # [batch_size, sequence_length]
            #
            reshaped_losses = raw_losses.view(
                batch_size,
                seq_len
            )

            reshaped_mask = loss_mask.view(
                batch_size,
                seq_len
            )


            weighted_losses = (
                reshaped_losses
                * normalized_weights.unsqueeze(1)
            )

            # Only include valid target tokens.
            numerator = weighted_losses[
                reshaped_mask
            ].sum()

            denominator = reshaped_mask.sum()

            loss = numerator / denominator

        else:

            loss = torch.tensor(
                0.0,
                device=hidden_states.device,
                requires_grad=True,
            )

    

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



def objective(trial):

    learning_rate = trial.suggest_categorical(
        "learning_rate",
        [
            1e-5,
            2e-5,
            5e-5,
            1e-4,
            2e-4,
        ],
    )

    lora_rank = trial.suggest_categorical(
        "lora_rank",
        [
            8,
            16,
            32,
        ],
    )

    lora_alpha = trial.suggest_categorical(
        "lora_alpha",
        [
            16,
            32,
            64,
        ],
    )

    epochs = trial.suggest_categorical(
        "epochs",
        [
            2,
            3,
            5,
            7,
        ],
    )

    print("\n==============================")
    print(f"Trial {trial.number}")
    print(f"LR     = {learning_rate}")
    print(f"Rank   = {lora_rank}")
    print(f"Alpha  = {lora_alpha}")
    print(f"Epochs = {epochs}")
    print("==============================")

    model = None
    trainer = None
    tokenizer = None
    val_loss = None

    output_dir = (
        f"weighted_optuna_trial_{trial.number}"
    )

    try:

        model, tokenizer = (
            FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=False,
            )
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen-2.5",
        )

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
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy(),
                "quality_weight": float(
                    example["quality_weight"]
                ),
            }

        print(
            "Formatting and tokenizing datasets explicitly..."
        )

        train_dataset = dataset["train"].map(
            tokenization_mapping_func,
            remove_columns=dataset["train"].column_names,
        )

        val_dataset = dataset["validation"].map(
            tokenization_mapping_func,
            remove_columns=dataset["validation"].column_names,
        )
        model = FastLanguageModel.get_peft_model(
            model,

            r=lora_rank,

            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],

            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",

            use_gradient_checkpointing="unsloth",

            random_state=42,
        )

        if hasattr(model, "config"):
            model.config.return_dict = True

        class SingleWeightDataCollator(
            DataCollatorForSeq2Seq
        ):

            def __call__(
                self,
                features,
                return_tensors=None,
            ):

                quality_weights = [
                    f.pop("quality_weight")
                    for f in features
                ]

                batch = super().__call__(
                    features,
                    return_tensors=return_tensors,
                )

                batch["quality_weight"] = torch.tensor(
                    quality_weights,
                    dtype=torch.float32,
                )

                return batch

        custom_collator = SingleWeightDataCollator(
            tokenizer=tokenizer,
            model=model,
        )


        trainer = WeightedTrainer(
            model=model,
            tokenizer=tokenizer,

            train_dataset=train_dataset,
            eval_dataset=val_dataset,

            data_collator=custom_collator,

            args=TrainingArguments(
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

                report_to="none",

                remove_unused_columns=False,
            ),
        )


        trainer.train()
        evaluation = trainer.evaluate()

        val_loss = evaluation["eval_loss"]

        print(
            f"Trial {trial.number} "
            f"validation loss: {val_loss}"
        )

    except Exception as e:

        print(
            f"Trial {trial.number} failed: {e}"
        )

        raise

    finally:

        if trainer is not None:
            del trainer

        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    return val_loss


if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=20,
    )

    print("\n================================")
    print("WEIGHTED OPTUNA SWEEP COMPLETE")
    print("================================")

    print(
        "Best parameters:",
        study.best_params
    )

    print(
        "Best validation loss:",
        study.best_value
    )

    with open(
        "best_weighted_optuna_params.json",
        "w",
    ) as f:

        json.dump(
            study.best_params,
            f,
            indent=4,
        )