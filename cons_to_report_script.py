import json 
INPUT_FILE = r"C:\Users\Aman Kumar Singh\Desktop\testing_Report_3B.jsonl"
OUTPUT_FILE = "./patient_report_check_3B.txt"

with open (INPUT_FILE, "r", encoding="utf-8") as infile:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for i,j in enumerate(infile):
            temp_dict = json.loads(j)
            output = temp_dict.get("output", "")
            consult = temp_dict.get("input")
            outfile.write(f"############################ { i+1 }. PATIENT ###########################\n\n")
            outfile.write(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$ DOCTOR CONSULTATION $$$$$$$$$$$$$$$$$$$$$$$$\n")
            outfile.write(consult)
            outfile.write(f"\n\n")
            outfile.write(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PATIENT REPORT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
            outfile.write(output)
            outfile.write("\n\n\n")
            if i ==10:
                break
print(f"All the outputs are successfully uploaded in the file-{OUTPUT_FILE}")














































# import json

# INPUT_FILE = "kbg_synthetic_conversations-5_llama.jsonl"
# OUTPUT_FILE = "doctor_parent_conversations_llama.txt"

# with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
#      open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

#     for line_num, line in enumerate(infile, start=1):
#         line = line.strip()

#         if not line:
#             continue

#         try:
#             record = json.loads(line)

#             transcript = record.get("synthetic_transcript", "")

#             outfile.write(f"=== Patient {record.get('patient_id', line_num)} ===\n")
#             outfile.write(transcript)
#             outfile.write("\n\n")

#         except json.JSONDecodeError:
#             print(f"Skipping invalid JSON on line {line_num}")

# print(f"Conversations saved to: {OUTPUT_FILE}")