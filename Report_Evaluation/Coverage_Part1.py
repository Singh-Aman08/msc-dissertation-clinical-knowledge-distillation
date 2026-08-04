
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REPORT_FILE = "factuality_scores_02.jsonl" #"patient_report_01.jsonl"
OUTPUT_FILE = "coverage_testing_02.jsonl"
HF_TOKEN = "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"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)


SYSTEM_PROMPT = (
"You are a clinical information extraction assistant. "
"Extract patient-specific clinical claims from a doctor-parent consultation.\n\n"

"Rules:\n"
"- Extract only clinical information explicitly stated by the parent about the child.\n"
"- Do not extract parent's emotions, uncertainty, opinions, concerns, intentions, or plans.\n"
"- Do not infer information or use external medical knowledge.\n"
"- Ignore doctor's questions and ignore N/A responses.\n"
"- Each claim should represent one clinically meaningful fact.\n"
"- Split a statement only when it contains multiple distinct clinical facts.\n"
"- Do not create multiple claims that describe the same clinical fact using different wording.\n"
"- Do not split out minor details such as duration, examples, comparisons, or context unless they represent an important clinical fact.\n"
"- Merge closely related symptoms or details into a single meaningful clinical claim.\n"
"- Keep the original meaning with minimal rewriting.\n"
"- Return only valid JSON containing the extracted claims.\n"
)
SYSTEM_VALIDATION_PROMPT = (
    "You are a clinical summary coverage evaluator. "
    "Your task is to evaluate whether a generated patient summary covers a clinical claim extracted from a doctor-parent consultation.\n\n"

    "Rules:\n"
    "- Evaluate ONLY the provided Clinical Patient Report and Target Claim.\n"
    "- Do not use external medical knowledge or add assumptions.\n"
    "- Mark YES if the patient report conveys the same clinical meaning as the target claim, even if different wording is used.\n"
    "- Treat clinically equivalent expressions as the same information.\n"
    "Examples:\n"
    '- "tooth pain when eating cold foods" ↔ "tooth sensitivity to cold foods" → YES\n'
    '- "valve issue" ↔ "heart valve abnormality" → YES\n'
    '- "delayed milestones" ↔ "takes longer to achieve milestones" → YES\n'
    "- Do not require exact wording.\n"
    "- Ignore minor differences in wording, grammar, tense, pronouns, or non-essential details.\n"
    "- Mark NO if the clinical information in the claim is not present or not supported by the patient report.\n"
    "You MUST respond ONLY with valid JSON:\n"
    "{\n"
    '  "results": [\n'
    '    {\n'
    '      "claim": "The patient has fluid in her ear.",\n'
    '      "covered": "YES"\n'
    '    }\n'
    '  ]\n'
    "}"
)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")


print("Loading tokenizer and model...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, token = HF_TOKEN
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config = quantization_config,
    token = HF_TOKEN
)

model.eval()


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id



def get_model_response(messages_list):

    prompt = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)



    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )



    generated_tokens = outputs[
        0
    ][
        inputs.input_ids.shape[-1]:
    ]


    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    return generated_text


def extract_and_parse_json(generated_text):


    json_match = re.search(
        r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```",
        generated_text,
        re.DOTALL
    )


    if json_match:

        json_string = json_match.group(1)


    else:

        start_candidates = [
            generated_text.find("{"),
            generated_text.find("[")
        ]


        start_candidates = [
            x for x in start_candidates
            if x != -1
        ]


        if not start_candidates:

            return None


        start = min(start_candidates)


        end = max(
            generated_text.rfind("}"),
            generated_text.rfind("]")
        )


        if end == -1:

            return None


        json_string = generated_text[
            start:end+1
        ]



    try:

        return json.loads(
            json_string
        )



    except json.JSONDecodeError:


        # Basic JSON repair

        repaired_json = json_string.strip()


        # Remove trailing commas

        repaired_json = re.sub(
            r",\s*([}\]])",
            r"\1",
            repaired_json
        )


        # Balance square brackets

        missing_square = (
            repaired_json.count("[")
            -
            repaired_json.count("]")
        )


        if missing_square > 0:

            repaired_json += "]" * missing_square



        # Balance curly brackets

        missing_curly = (
            repaired_json.count("{")
            -
            repaired_json.count("}")
        )


        if missing_curly > 0:

            repaired_json += "}" * missing_curly



        try:

            return json.loads(
                repaired_json
            )


        except json.JSONDecodeError:

            return None


print("\nStarting clinical coverage evaluation pipeline...\n")


with open(
    REPORT_FILE,
    "r",
    encoding="utf-8"
) as infile, open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as outfile:


    for idx, line in enumerate(infile):

        if not line.strip():
            continue


        data = json.loads(line)
        factuality_scores = data.get(
    "factuality_scores",
    {}
)

        factuality_breakdown = data.get(
    "audit_breakdown",
    {}
)

        

        consultation_text = data.get("consultation", "")

        patient_report = data.get("report", "")

        if not consultation_text or not patient_report:

            print(
                f"Skipping row {idx}: missing consultation or report"
            )

            continue


        extraction_messages = [

    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },

    {
        "role": "user",
        "content":
            f"Now extract atomic patient claims from this consultation:\n\n"
            f"{consultation_text}"
    }

]


        raw_extraction_text = get_model_response(
            extraction_messages
        )


        claims_list = extract_and_parse_json(
            raw_extraction_text
        )



        if not isinstance(claims_list, list):

            print(
                f"Claim extraction failed for row {idx}"
            )

            print(raw_extraction_text)

            claims_list = []

        coverage_results = []

        covered_count = 0



        for claim in claims_list:


            validation_messages = [

    {
        "role": "system",
        "content": SYSTEM_VALIDATION_PROMPT
    },

    {
        "role": "user",
        "content":
            f"Clinical Patient Report:\n"
            f"{patient_report}\n\n"
            f"Target Claim:\n"
            f"{claim}\n\n"
            "Determine whether the patient report includes this clinical information. "
            "Return the evaluation only in the required JSON format."
    }

]



            raw_validation_text = get_model_response(
                validation_messages
            )


            validation_json = extract_and_parse_json(
                raw_validation_text
            )



            covered = "NO"



            if isinstance(validation_json, dict):

                results = validation_json.get(
        "results",
        []
    )

                if results:

                    covered = str(
            results[0].get(
                "covered",
                "NO"
            )
        ).upper()



            coverage_results.append(

                {
                    "claim": claim,

                    "covered": covered,

                }

            )



            if covered == "YES":

                covered_count += 1


        total_claims = len(
            claims_list
        )


        coverage_score = (

            covered_count / total_claims

            if total_claims > 0

            else 1.0

        )


        output_data = {

    "index": data.get(
        "index",
        idx
    ),

    "consultation": consultation_text,

    "report": patient_report,


    # Original factuality claims
    "claim_decomposition": data.get(
        "claim_decomposition",
        {}
    ),


    # Factuality results
    "factuality_scores": factuality_scores,


    # Coverage results
    "coverage_scores": {

        "patient_coverage_score": round(
            coverage_score,
            4
        )

    },


    # Coverage claims generated separately
    "extracted_coverage_claims": claims_list,


    "coverage_summary_counts": {

        "total_claims": total_claims,

        "covered_claims": covered_count

    },


    # Detailed evaluations
    "factuality_audit_breakdown": factuality_breakdown,

    "coverage_breakdown": coverage_results

}



        outfile.write(

            json.dumps(
                output_data,
                ensure_ascii=False
            )

            +

            "\n"

        )
        outfile.flush()



        print(

            f"Processed row {idx} | "
            f"Coverage: {round(coverage_score,4)} "
            f"({covered_count}/{total_claims})"

        )

        if idx == 5:
            break 

print(
    "\nClinical coverage evaluation completed successfully."
)


############################################################################################

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_FILE = "factuality_scores_02.jsonl"
OUTPUT_FILE = "conciseness_testing_results.jsonl"
HF_TOKEN = "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"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

print(
    "CUDA available:",
    torch.cuda.get_device_name(0),
    flush=True
)

profiles = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            profiles.append(json.loads(line))


print(
    f"Loaded {len(profiles)} reports",
    flush=True
)

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    token=HF_TOKEN,
    low_cpu_mem_usage=True
)


model.eval()

print(
    "Model loaded successfully",
    flush=True
)
system_prompt = (
    "You are an expert evaluator of clinical patient reports.\n\n"
    "Your task is to evaluate the conciseness of a report by identifying ONLY unnecessary repetitions.\n\n"
    "A repetition is considered unnecessary ONLY when the same clinical information is repeated without adding any new information, context, interpretation, clinical purpose, or value.\n\n"
    "Evaluate each potential repetition by considering:\n\n"
    "- Whether the same clinical information appears multiple times.\n"
    "- Whether each occurrence serves the same purpose.\n"
    "- Whether removing one occurrence would reduce the reader's understanding of the patient's condition.\n\n"
    "Do NOT consider repetition unnecessary when:\n\n"
    "- The same finding appears in different sections with different clinical purposes.\n"
    "- A finding is repeated to explain impact, functional consequences, management, treatment, monitoring, or follow-up.\n"
    "- A recommendation refers to a patient condition to justify an action.\n"
    "- The repeated information provides additional clinical context or interpretation.\n\n"
    "Do NOT mark a repetition only because the same symptom or diagnosis appears in multiple sections.\n"
    "Assess the purpose and context of each occurrence.\n\n"
    "Identify ONLY genuine redundancy.\n\n"
    "You MUST respond ONLY with valid JSON:\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "information": "",\n'
    '      "sections": [],\n'
    '      "reasoning": ""\n'
    "    }\n"
    "  ],\n"
    '  "total_redundant_items": 0\n'
    "}"
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pass


print(
    "Starting conciseness evaluation...",
    flush=True
)

for idx, data in enumerate(profiles):

    report_index = data.get(
        "index",
        idx
    )

    print(
        f"\n[{idx + 1}/{len(profiles)}] Evaluating index: {report_index}",
        flush=True
    )


    # Extract report
    patient_report = data.get(
        "report",
        ""
    )


    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
Evaluate the following clinical patient report:

{patient_report}
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

            max_new_tokens=512,

            # Deterministic LLM judge
            do_sample=False,

            repetition_penalty=1.05,

            pad_token_id=tokenizer.pad_token_id
        )


    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()


    try:

        conciseness_result = json.loads(
            generated_text
        )

    except Exception:

        # fallback if model adds text around JSON
        try:

            start = generated_text.index("{")
            end = generated_text.rindex("}") + 1

            conciseness_result = json.loads(
                generated_text[start:end]
            )

        except Exception:

            conciseness_result = {
                "parse_error": True,
                "raw_output": generated_text
            }


    if "redundant_items_count" in conciseness_result:

        repetition_count = conciseness_result[
            "redundant_items_count"
        ]

        print(
            f"Index {report_index}: "
            f"Redundant items = {repetition_count}",
            flush=True
        )


        if repetition_count > 0:

            for item in conciseness_result.get(
                "redundant_items",
                []
            ):

                print(
                    "  -",
                    item.get("information", ""),
                    "| Sections:",
                    item.get("sections", []),
                    flush=True
                )


    else:

        print(
            f"Index {report_index}: JSON parsing failed",
            flush=True
        )


    output_data = {

        "index": report_index,

        "conciseness_evaluation": conciseness_result,

        "report": patient_report

    }


    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                output_data,
                ensure_ascii=False
            )
            + "\n"
        )

        f.flush()


    print(
        f"Saved index {report_index}",
        flush=True
    )


print(
    "\nAll conciseness evaluations completed successfully.",
    flush=True
)


############################################################################
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


#############################################################################################

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_FILE = "factuality_scores_02.jsonl"
OUTPUT_FILE = "conciseness_testing_results.jsonl"
HF_TOKEN = "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"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")


print(
    "CUDA available:",
    torch.cuda.get_device_name(0),
    flush=True
)


profiles = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            profiles.append(json.loads(line))


print(
    f"Loaded {len(profiles)} reports",
    flush=True
)


print("Loading tokenizer...", flush=True)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


print("Loading model...", flush=True)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    token=HF_TOKEN,
    low_cpu_mem_usage=True
)


model.eval()


print(
    "Model loaded successfully",
    flush=True
)
system_prompt = (
"You are an expert evaluator of clinical patient reports. "
"Identify ONLY unnecessary repetition within each section.\n\n"

"Evaluate redundancy separately for each section. "
"Do NOT compare information across different sections.\n\n"

"Count a repetition ONLY when the same information is repeated within the same section "
"without adding any new detail, explanation, or clinical value.\n\n"

"Do NOT mark repetition when similar information appears in different sections, "
"as each section may describe a condition from a different clinical perspective.\n\n"

"Return ONLY valid JSON:\n"
"{\n"
'  \"total_redundant_items\": 0,\n'
'  \"results\": [\n'
'    {\n'
'      \"information\": \"\",\n'
'      \"sections\": [],\n'
'      \"reasoning\": \"\"\n'
'    }\n'
'  ]\n'
"}"
)

def extract_and_parse_json(generated_text):

    json_match = re.search(
        r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```",
        generated_text,
        re.DOTALL
    )


    if json_match:

        json_string = json_match.group(1)

    else:

        start_candidates = [
            generated_text.find("{"),
            generated_text.find("[")
        ]


        start_candidates = [
            x for x in start_candidates
            if x != -1
        ]


        if not start_candidates:
            return None


        start = min(start_candidates)


        end = max(
            generated_text.rfind("}"),
            generated_text.rfind("]")
        )


        if end == -1:
            return None


        json_string = generated_text[start:end+1]


    try:

        return json.loads(json_string)


    except json.JSONDecodeError:


        repaired_json = json_string.strip()


        repaired_json = re.sub(
            r",\s*([}\]])",
            r"\1",
            repaired_json
        )


        missing_square = (
            repaired_json.count("[")
            -
            repaired_json.count("]")
        )


        if missing_square > 0:
            repaired_json += "]" * missing_square


        missing_curly = (
            repaired_json.count("{")
            -
            repaired_json.count("}")
        )


        if missing_curly > 0:
            repaired_json += "}" * missing_curly


        try:

            return json.loads(repaired_json)


        except json.JSONDecodeError:

            return None



with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    pass


print(
    "Starting conciseness evaluation...",
    flush=True
)


for idx, data in enumerate(profiles):

    report_index = data.get(
        "index",
        idx
    )


    print(
        f"\n[{idx+1}/{len(profiles)}] Evaluating index: {report_index}",
        flush=True
    )


    patient_report = data.get(
        "report",
        ""
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                "Evaluate the following clinical patient report:\n\n"
                f"{patient_report}"
            )
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

            # Deterministic evaluation
            do_sample=False,

            repetition_penalty=1.05,

            pad_token_id=tokenizer.pad_token_id
        )



    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()



    conciseness_result = extract_and_parse_json(
        generated_text
    )



    if conciseness_result is None:

        conciseness_result = {
            "parse_error": True,
            "raw_output": generated_text
        }

        print(
            f"Index {report_index}: JSON parsing failed",
            flush=True
        )


    else:

        repetition_count = conciseness_result.get(
            "total_redundant_items",
            0
        )


        print(
            f"Index {report_index}: "
            f"Redundant items = {repetition_count}",
            flush=True
        )


        for item in conciseness_result.get(
            "results",
            []
        ):

            print(
                "  -",
                item.get("information", ""),
                "| Sections:",
                item.get("sections", []),
                flush=True
            )



    output_data = {

        "index": report_index,

        "conciseness_evaluation": conciseness_result,

        "report": patient_report

    }



    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                output_data,
                ensure_ascii=False
            )
            + "\n"
        )

        f.flush()



    print(
        f"Saved index {report_index}",
        flush=True
    )



print(
    "\nAll conciseness evaluations completed successfully.",
    flush=True
)