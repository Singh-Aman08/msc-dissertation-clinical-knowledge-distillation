import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REPORT_FILE = "patient_report_07.jsonl"
OUTPUT_FILE = "claim_decomposition_07.jsonl"
HF_TOKEN = "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"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,
    token = HF_TOKEN
)

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("\nBeginning Claim Decomposition Pipeline...\n", flush=True)

SYSTEM_PROMPT = """ You are an expert clinical information extraction assistant.

Your task is to analyse a generated clinical report and extract all factual claims. The report contains two distinct types of information, and each extracted claim must be assigned to the correct category.

Category 1: Patient-Specific Claims

These claims are extracted from sections describing:
"How this affects the patient"

They represent factual statements about the individual patient.

Category 2: Syndrome-Specific Claims

These claims are extracted from sections describing:
"How this affects others with the syndrome" and "Recommendations for Screening and Treatments"

They represent factual statements about KBG syndrome, its effects on affected individuals in general, and general recommendations for screening, monitoring, or management of the syndrome.

Instructions:

- Extract all factual claims explicitly stated in the report.
- Across all clinical categories, claims from "How this affects the patient" sections must be classified as Patient-Specific Claims.
- Across all clinical categories, claims from "How this affects others with the syndrome" sections and the "Recommendations for Screening and Treatments" section must be classified as Syndrome-Specific Claims.
- Decompose complex sentences into atomic claims, where each claim contains only one factual statement.
- Extract claims faithfully. Do not add, infer, explain, or reinterpret information. Preserve the original meaning and only perform minimal restructuring for claim decomposition.
- Ignore "N/A" sections as they do not contain factual claims.
- Return only the extracted claims in the required JSON format.
- Ensure the final output is a complete and valid JSON object with all brackets and quotes properly closed."""


few_shot_user = """

# CLINICAL PATIENT REPORT

## 1) RESPIRATORY
a) How this affects the patient: N/A
b) How this affects others with the syndrome: N/A

## 2) CARDIOLOGY
a) How this affects the patient: The patient has a structural abnormality in her heart.
b) How this affects others with the syndrome: Around 38% of people with KBG syndrome have heart conditions, including changes to the heart valves, holes in the heart chambers, and structural changes.

## 3) GASTROENTEROLOGY
a) How this affects the patient: The patient experiences bowel movements problems, causing discomfort and irregularity.
b) How this affects others with the syndrome: Some people with KBG syndrome have feeding difficulties and may require short-term nasogastric tube feeding to supplement oral feeds.

## 4) IMMUNOLOGY
a) How this affects the patient: N/A
b) How this affects others with the syndrome: N/A

## 5) NEUROLOGY
a) How this affects the patient: The patient gets exhausted quickly when playing or running around, more so than her friends.
b) How this affects others with the syndrome: Around 20-40% of people with KBG syndrome have seizures, and some may experience developmental delay and behavioural differences.

## 6) EAR NOSE THROAT
a) How this affects the patient: The patient has fluid in her ear, which has been a recurring issue.
b) How this affects others with the syndrome: A significant proportion of people with KBG syndrome have recurrent otitis media or glue ear, which can cause conductive hearing loss.

## 7) OPHTHALMOLOGY AND VISION
a) How this affects the patient: The patient has trouble seeing the board in class, indicating potential eyesight problems.
b) How this affects others with the syndrome: People with KBG syndrome are more likely to have vision problems such as astigmatism, short- or long-sightedness, and squint.

## 8) DERMATOLOGY
a) How this affects the patient: N/A
b) How this affects others with the syndrome: N/A

## 9) DENTAL
a) How this affects the patient: The patient's teeth break or chip very easily, even with normal eating and drinking.
b) How this affects others with the syndrome: People with KBG syndrome often have large front teeth, weak enamel, and other dental concerns, making regular dental check-ups important.

## 10) EDUCATION
a) How this affects the patient: The patient seems behind other kids her age in terms of growth and development.
b) How this affects others with the syndrome: Children with KBG syndrome typically need extra help in school, and some may require special schooling or Educational Health Care Plans.

## 11) BEHAVIOUR AND DEVELOPMENT
a) How this affects the patient: The patient seems behind other kids her age in terms of growth and development.
b) How this affects others with the syndrome: People with KBG syndrome often have developmental delay, learning difficulties, and behavioural differences, such as autism spectrum disorder, ADHD, or anxiety.

## 12) SKELETAL
a) How this affects the patient: N/A
b) How this affects others with the syndrome: Some people with KBG syndrome have skeletal anomalies, such as an unusual structure of their spinal bones, short fingers, or hip dysplasia.

## 13) RECOMMENDATIONS FOR SCREENING AND TREATMENTS
* Regular dental check-ups to monitor tooth health and address any concerns.
* Regular hearing reviews to age 5 to address any hearing concerns.
* Eyesight review to assess and address any vision problems.
* Cardiac review to monitor the patient's heart condition.
* Consider referral for a skeletal review (X-ray of the wrist, hip, spine, and skull) to assess for any skeletal anomalies.
* Monitor growth velocity and consider referral for endocrine investigations if height is below the 2nd centile.
* Consider physiotherapy, occupational therapy, speech therapy, and behavioural therapy to support the patient's development and address any delays or difficulties.


"""

few_shot_assistant = """ {
  "patient_specific_claims": [
    "The patient has a structural abnormality in her heart.",
    "The patient experiences bowel movements problems.",
    "The patient experiences discomfort.",
    "The patient experiences irregularity.",
    "The patient gets exhausted quickly when playing or running around.",
    "The patient gets exhausted more than her friends when playing or running around.",
    "The patient has fluid in her ear.",
    "The patient's fluid in her ear has been a recurring issue.",
    "The patient has trouble seeing the board in class.",
    "The patient has potential eyesight problems.",
    "The patient's teeth break very easily.",
    "The patient's teeth chip very easily.",
    "The patient's teeth break or chip even with normal eating and drinking.",
    "The patient seems behind other kids her age in terms of growth.",
    "The patient seems behind other kids her age in terms of development."
  ],
  "syndrome_specific_claims": [
    "Around 38% of people with KBG syndrome have heart conditions.",
    "People with KBG syndrome have changes to the heart valves.",
    "People with KBG syndrome have holes in the heart chambers.",
    "People with KBG syndrome have structural changes.",
    
    "Some people with KBG syndrome have feeding difficulties.",
    "Some people with KBG syndrome may require short-term nasogastric tube feeding to supplement oral feeds.",
    
    "Around 20-40% of people with KBG syndrome have seizures.",
    "Some people with KBG syndrome may experience developmental delay.",
    "Some people with KBG syndrome may experience behavioural differences.",
    
    "A significant proportion of people with KBG syndrome have recurrent otitis media or glue ear.",
    "Recurrent otitis media or glue ear can cause conductive hearing loss.",
    
    "People with KBG syndrome are more likely to have vision problems.",
    "People with KBG syndrome may have astigmatism.",
    "People with KBG syndrome may have short-sightedness.",
    "People with KBG syndrome may have long-sightedness.",
    "People with KBG syndrome may have squint.",
    
    "People with KBG syndrome often have large front teeth.",
    "People with KBG syndrome often have weak enamel.",
    "People with KBG syndrome may have other dental concerns.",
    "Regular dental check-ups are important.",
    
    "Children with KBG syndrome typically need extra help in school.",
    "Some children with KBG syndrome may require special schooling.",
    "Some children with KBG syndrome may require Educational Health Care Plans.",
    
    "People with KBG syndrome often have developmental delay.",
    "People with KBG syndrome often have learning difficulties.",
    "People with KBG syndrome often have behavioural differences.",
    "People with KBG syndrome may have autism spectrum disorder.",
    "People with KBG syndrome may have ADHD.",
    "People with KBG syndrome may have anxiety.",
    
    "Some people with KBG syndrome have skeletal anomalies.",
    "Some people with KBG syndrome have an unusual structure of their spinal bones.",
    "Some people with KBG syndrome have short fingers.",
    "Some people with KBG syndrome have hip dysplasia.",
    
    "Regular dental check-ups to monitor tooth health and address any concerns.",
    "Regular hearing reviews to age 5 to address any hearing concerns.",
    "Eyesight review to assess and address any vision problems.",
    "Cardiac review to monitor the patient's heart condition.",
    "Consider referral for a skeletal review.",
    "X-ray of the wrist, hip, spine, and skull.",
    "Monitor growth velocity.",
    "Consider referral for endocrine investigations if height is below the 2nd centile.",
    "Consider physiotherapy.",
    "Consider occupational therapy.",
    "Consider speech therapy.",
    "Consider behavioural therapy."
  ]
} """


with open(REPORT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for idx, line in enumerate(infile):
        if not line.strip():
            continue
        data = json.loads(line)
        patient_report = data.get("output", "")
        if not patient_report:
            print(f"Skipping row {idx}: empty report", flush=True)
            continue
        
        print(
                    f"Processing row {idx}",
                    flush=True
                )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": few_shot_user},
            {"role": "assistant", "content": few_shot_assistant},
            {
                "role": "user", 
                "content": f"Generated Clinical Report:\n\n{patient_report}\n\nExtract all Patient-Specific Claims and Syndrome-Specific Claims.\nReturn ONLY valid JSON."
            }
        ]
        print(
                    "Applying chat template...",
                    flush=True
                )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(
                            "Tokenizing prompt...",
                            flush=True
                        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192 # Safeguard against GPU OOM crashing
        ).to(model.device)
        
        print(
                    "Starting generation...",
                    flush=True
                )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=2048,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True).strip()
        print(
                    "Generation completed.",
                    flush=True
                )
        
    

        parsed_json = None

        try:
            # Step 1: Extract JSON block
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                generated_text,
                re.DOTALL
            )

            if json_match:
                json_string = json_match.group(1)

            else:
                start = generated_text.find("{")
                end = generated_text.rfind("}")

                if start == -1:
                    raise ValueError("No JSON object found")

                json_string = generated_text[start:end+1]


            # Step 2: Try normal JSON parsing
            try:
                parsed_json = json.loads(json_string)


            # Step 3: Repair incomplete JSON
            except json.JSONDecodeError:

                repaired_json = json_string.strip()

                # Remove accidental trailing commas
                repaired_json = re.sub(
                    r",\s*([}\]])",
                    r"\1",
                    repaired_json
                )

                # Close missing square brackets
                missing_square = (
                    repaired_json.count("[") -
                    repaired_json.count("]")
                )

                if missing_square > 0:
                    repaired_json += "]" * missing_square


                # Close missing curly brackets
                missing_curly = (
                    repaired_json.count("{") -
                    repaired_json.count("}")
                )

                if missing_curly > 0:
                    repaired_json += "}" * missing_curly


                parsed_json = json.loads(repaired_json)


            # Step 4: Validate expected structure
            if (
                "patient_specific_claims" not in parsed_json
                or
                "syndrome_specific_claims" not in parsed_json
            ):
                raise ValueError(
                    "Missing required JSON keys"
                )


        except Exception as e:

            print(
                f"Row {idx}: JSON parsing failed - {e}", flush=True
            )

            parsed_json = {
                "error": "json_parsing_failed",
                "raw_output": generated_text
            }


        output = {
    "consultation": data.get("input", ""),
    "report": data.get("output", ""),
    "claim_decomposition": parsed_json
}

        outfile.write(
            json.dumps(
                output,
                ensure_ascii=False
            ) + "\n"
        )
        outfile.flush()
        #if idx == 2:
            #break
        print(f"Claim decomposition completed for index {idx + 1}", flush=True)

print("\nFinished claim decomposition.", flush=True)
