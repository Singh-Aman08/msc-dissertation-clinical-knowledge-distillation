import json

INPUT_FILE = "kbg_synthetic_conversations-5_llama.jsonl"
OUTPUT_FILE = "doctor_parent_conversations_llama.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line_num, line in enumerate(infile, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)

            transcript = record.get("synthetic_transcript", "")

            outfile.write(f"=== Patient {record.get('patient_id', line_num)} ===\n")
            outfile.write(transcript)
            outfile.write("\n\n")

        except json.JSONDecodeError:
            print(f"Skipping invalid JSON on line {line_num}")

print(f"Conversations saved to: {OUTPUT_FILE}")