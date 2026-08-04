import json
import torch
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROFILE_FILE = " "
OUTPUT_FILE = " " 
HF_TOKEN = "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"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")


with open(PROFILE_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)

print(f"Loaded {len(profiles)} patient profiles")


print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = HF_TOKEN )

# Ensure correct padding configurations for Qwen text generation
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)


print("Loading model onto GPU")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",     #cuda
    token = HF_TOKEN,
    low_cpu_mem_usage=True
    
)
model.eval()

# Clear/Initialize the target output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pass

print(f"Starting pipeline. Streaming outputs straight to: {OUTPUT_FILE}")


for idx, profile in enumerate(profiles):
    print(f"\nGenerating consultation {idx + 1}/{len(profiles)} (Patient ID: {profile.get('patient_id')})")

    messages = [
        {
            "role": "system",
            "content": ("""You are an expert evaluator of clinical patient reports.

Your task is to evaluate the conciseness of a report by identifying unnecessary repetitions.

A repetition is considered unnecessary ONLY when the same clinical information is repeated across different sections without adding any new information, context, interpretation, or clinical purpose.

Consider a repetition unnecessary when:

- The same patient-specific finding is mentioned multiple times in the report.
- The repeated mention does not provide additional details, explanation, consequences, or clinical relevance.
- Removing one occurrence would not reduce the reader's understanding of the patient's condition.

Do NOT consider a repetition unnecessary when:

- The information is repeated to explain a different aspect of the condition.
- A patient finding is linked to its impact, management, treatment, or follow-up.
- The same finding appears in a recommendation or care plan because it justifies the recommended action.
- The information is repeated in a different context that provides additional clinical value.

Identify ONLY genuine redundancy and ignore clinically meaningful repetition.

Return the output strictly as valid JSON:
{
  "redundant_items_count": 0,
  "redundant_items": [
    {
      "information": "",
      "sections": [],
      "reason": ""
    }
  ]
}
""" )
        },
        
        {
            "role": "user",
            "content": f"PATIENT_PROFILE\n\n{json.dumps(profile, indent=2)}\n\nGenerate the consultation."
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            
            # Critical adjustments for strict data extraction tasks
            temperature=0.75,  
            do_sample=True,   
            top_p=0.9,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only the newly generated text block
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # Create record object
    record = {
        "patient_id": profile.get("patient_id", idx),
        "profile": profile,
        "synthetic_transcript": generated_text
    }

    # Instantly append line record to disk
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
    # if idx == 10:
    #     break
print("\nAll generations finished successfully.")

######################################################################################

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_FILE = "factuality_scores_02.jsonl"
OUTPUT_FILE = "conciseness_results_testing.jsonl"
HF_TOKEN = "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"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")


# Load JSONL file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    profiles = [json.loads(line) for line in f]


print(f"Loaded {len(profiles)} patient reports")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# Quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    token=HF_TOKEN,
    low_cpu_mem_usage=True
)

model.eval()


system_prompt = """
You are an expert evaluator of clinical patient reports.

Your task is to evaluate the conciseness of a report by identifying unnecessary repetitions.

A repetition is considered unnecessary ONLY when the same clinical information is repeated across different sections without adding any new information, context, interpretation, or clinical purpose.

Consider a repetition unnecessary when:

- The same patient-specific finding is mentioned multiple times.
- The repeated mention does not provide additional details, explanation, consequences, or clinical relevance.
- Removing one occurrence would not reduce the reader's understanding of the patient's condition.

Do NOT consider a repetition unnecessary when:

- The information is repeated to explain a different aspect of the condition.
- A finding is linked to its impact, management, treatment, or follow-up.
- The repeated information provides additional clinical value.

Identify ONLY genuine redundancy and ignore clinically meaningful repetition.

Return ONLY valid JSON.

Output format:

{
  "redundant_items_count": 0,
  "redundant_items": [
    {
      "information": "",
      "sections": [],
      "reason": ""
    }
  ]
}
"""


# Clear output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pass


print("Starting conciseness evaluation...")


for idx, profile in enumerate(profiles):

    print(
        f"Evaluating report {idx + 1}/{len(profiles)}"
    )


    report_text = profile.get("report", "")


    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
Evaluate the following patient report:

{report_text}
"""
        }
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,

            # Deterministic judge
            do_sample=False,

            repetition_penalty=1.05,

            pad_token_id=tokenizer.pad_token_id
        )


    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()


    # Parse JSON output
    try:
        evaluation = json.loads(generated_text)

    except Exception:
        evaluation = {
            "parse_error": True,
            "raw_output": generated_text
        }


    result = {
        "patient_id": profile.get("patient_id", idx),
        "conciseness_evaluation": evaluation
    }


    # Save immediately
    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(result, ensure_ascii=False)
            + "\n"
        )


print("All conciseness evaluations completed successfully.")