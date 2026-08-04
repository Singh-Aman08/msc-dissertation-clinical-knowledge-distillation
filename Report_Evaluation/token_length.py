import json
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_FILE = "factuality_scores_01.jsonl"
OUTPUT_FILE = "report_token_lengths_01.jsonl"
HF_TOKEN = "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"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)
print("Tokenizer loaded successfully\n", flush=True)

# Load all profiles into memory
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    profiles = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(profiles)} reports. Starting calculation...", flush=True)

# Open the file ONCE and write line-by-line (Much faster)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for idx, data in enumerate(profiles):
        report_index = data.get("index", idx)
        patient_report = data.get("report", "")

        # Fast token length calculation without creating PyTorch tensors
        tokens = tokenizer.encode(patient_report, add_special_tokens=True)
        token_length = len(tokens)

        output_data = {
            "index": report_index,
            "token_length": token_length
        }

        f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        
        print(f"Index {report_index}: {token_length} tokens", flush=True)

print("\nToken length calculation completed successfully.", flush=True)
