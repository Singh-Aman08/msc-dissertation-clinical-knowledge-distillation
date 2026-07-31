import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

REPORT_FILE = "patient_report_01.jsonl"
OUTPUT_FILE = "coverage_testing_01.jsonl"
HF_TOKEN = "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"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)


SYSTEM_PROMPT = SYSTEM_PROMPT = (
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
"- Mark YES if the patient report explicitly mentions the clinical information in the claim or provides a direct semantic paraphrase.\n"
"- Ignore minor differences in wording, grammar, tense, pronouns, or non-essential details.\n"
"- Mark NO if the claim contains information, interpretation, diagnosis, comparison, or meaning that is not supported by the patient report.\n"
"- For multi-part claims, mark YES only when all clinically important information is covered in the patient report.\n\n"

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


few_shot_user = """ Doctor: Please tell us about any problems with your child’s vision or eyes.
Parent: She has trouble seeing the board in class, so we're concerned her eyesight might not be perfect.

Doctor: Please tell us about problems with your child's hearing or ears.
Parent: There's fluid in her ear, which has been a recurring issue.

Doctor: Please tell us about the problems with your child’s teeth.
Parent: Her teeth break or chip very easily, even with normal eating and drinking.

Doctor: Please tell us about any problems with your child’s heart.
Parent: Doctors have mentioned a structural abnormality in her heart.

Doctor: Please tell us about any problems with your child's hormones.
Parent: N/A.

Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
Parent: She gets exhausted quickly when playing or running around, more so than her friends.

Doctor: Please tell us about any problems with your child's kidneys.
Parent: N/A.

Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
Parent: N/A.

Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
Parent: Bowel movements have been a problem for her, causing discomfort and irregularity.

Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above.
Parent: Overall, she seems a bit behind other kids her age in terms of growth and development.
"""

few_shot_assistant = """
[
  "The patient has trouble seeing the board in class.",
  "The patient has recurring fluid in her ear.",
  "The patient's teeth break or chip easily.",
  "The patient has a structural abnormality in her heart.",
  "The patient gets exhausted more quickly than other children her age during physical activity.",
  "The patient has bowel movement difficulties causing discomfort and irregularity.",
  "The patient is behind other children her age in terms of growth and development."
]
"""



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



        consultation_text = data.get("input", "")

        patient_report = data.get("output", "")

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
            f"Example Doctor-Parent Consultation:\n\n"
            f"{few_shot_user}"
    },

    {
        "role": "assistant",
        "content": few_shot_assistant
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


            "row_index": idx,


            "consultation": consultation_text,


            "patient_report": patient_report,



            "extracted_claims": claims_list,



            "summary_counts": {

                "total_claims": total_claims,

                "covered_claims": covered_count

            },



            "patient_coverage_score": round(
                coverage_score,
                4
            ),



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

        if idx == 7:
            break 

print(
    "\nClinical coverage evaluation completed successfully."
)

#######################################################################################
#####################################################################################
SYSTEM_PROMPT = (
"You are a clinical information extraction assistant. "
"Extract atomic patient-specific clinical claims from a doctor-parent consultation.\n\n"

"Rules:\n"
"- Extract only clinical information explicitly stated by the parent about the child.\n"
"- Do not extract parent's emotions, uncertainty, opinions, intentions, or plans.\n"
"- Do not infer information or use external medical knowledge.\n"
"- Ignore doctor's questions and any N/A responses.\n"
"- Split a claim into separate claims only when it contains multiple distinct clinical facts.\n"
"- Keep each claim focused on one clinical fact.\n"
"- Keep the original meaning with minimal rewriting.\n"
"- Return only valid JSON containing the extracted claims.\n"
)
SYSTEM_VALIDATION_PROMPT = (
"You are a clinical summary coverage evaluator. "
"Your task is to evaluate whether a generated patient summary covers a clinical claim extracted from a doctor-parent consultation.\n\n"

"Rules:\n"
"- Evaluate ONLY the provided Clinical Patient Report and Target Claim.\n"
"- Do not use external medical knowledge or add assumptions.\n"
"- Mark YES if the patient report explicitly mentions the clinical information in the claim or provides a direct semantic paraphrase.\n"
"- Ignore minor differences in wording, grammar, tense, pronouns, or non-essential details.\n"
"- Mark NO if the claim contains information, interpretation, diagnosis, comparison, or meaning that is not supported by the patient report.\n"
"- For multi-part claims, mark YES only when all clinically important information is covered in the patient report.\n\n"

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


few_shot_user = """ Doctor: Please tell us about any problems with your child’s vision or eyes.
Parent: She has trouble seeing the board in class, so we're concerned her eyesight might not be perfect.

Doctor: Please tell us about problems with your child's hearing or ears.
Parent: There's fluid in her ear, which has been a recurring issue.

Doctor: Please tell us about the problems with your child’s teeth.
Parent: Her teeth break or chip very easily, even with normal eating and drinking.

Doctor: Please tell us about any problems with your child’s heart.
Parent: Doctors have mentioned a structural abnormality in her heart.

Doctor: Please tell us about any problems with your child's hormones.
Parent: N/A.

Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
Parent: She gets exhausted quickly when playing or running around, more so than her friends.

Doctor: Please tell us about any problems with your child's kidneys.
Parent: N/A.

Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
Parent: N/A.

Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
Parent: Bowel movements have been a problem for her, causing discomfort and irregularity.

Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above.
Parent: Overall, she seems a bit behind other kids her age in terms of growth and development.
"""

few_shot_assistant = """
[
  "The patient has trouble seeing the board in class.",
  "The patient has fluid in her ear.",
  "The patient has recurring fluid in her ear.",
  "The patient's teeth break or chip easily.",
  "The patient has a structural abnormality in her heart.",
  "The patient gets exhausted quickly when playing or running around.",
  "The patient gets tired more easily than other children her age.",
  "The patient has bowel movement difficulties causing discomfort and irregularity.",
  "The patient is behind other children her age in terms of growth and development."
]
"""
############################################################################################

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

REPORT_FILE = "patient_report_15.jsonl"#"patient_report_01.jsonl"
OUTPUT_FILE = "coverage_testing_01.jsonl"
HF_TOKEN = "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"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)


SYSTEM_PROMPT = SYSTEM_PROMPT = (
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
"- Mark YES if the patient report explicitly mentions the clinical information in the claim or provides a direct semantic paraphrase.\n"
"- Ignore minor differences in wording, grammar, tense, pronouns, or non-essential details.\n"
"- Mark NO if the claim contains information, interpretation, diagnosis, comparison, or meaning that is not supported by the patient report.\n"
"- For multi-part claims, mark YES only when all clinically important information is covered in the patient report.\n\n"

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



        consultation_text = data.get("input", "")

        patient_report = data.get("output", "")

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


            "row_index": idx,


            "consultation": consultation_text,


            "patient_report": patient_report,



            "extracted_claims": claims_list,



            "summary_counts": {

                "total_claims": total_claims,

                "covered_claims": covered_count

            },



            "patient_coverage_score": round(
                coverage_score,
                4
            ),



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

######################################################################################
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

REPORT_FILE = "testing_Report_3B.jsonl"#"patient_report_01.jsonl"
OUTPUT_FILE = "coverage_testing_01.jsonl"
HF_TOKEN = "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"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)


SYSTEM_PROMPT = (
"You are a clinical information extraction assistant. "
"Extract atomic patient-specific clinical claims from a doctor-parent consultation.\n\n"

"Rules:\n"
"- Extract only clinical information explicitly stated by the parent about the child.\n"
"- Do not extract parent's emotions, uncertainty, opinions, intentions, or plans.\n"
"- Do not infer information or use external medical knowledge.\n"
"- Ignore doctor's questions and any N/A responses.\n"
"- Split a claim into separate claims only when it contains multiple distinct clinical facts.\n"
"- Keep each claim focused on one clinical fact.\n"
"- Keep the original meaning with minimal rewriting.\n"
"- Return only valid JSON containing the extracted claims.\n"
)
SYSTEM_VALIDATION_PROMPT = (
"You are a clinical summary coverage evaluator. "
"Your task is to evaluate whether a generated patient summary covers a clinical claim extracted from a doctor-parent consultation.\n\n"

"Rules:\n"
"- Evaluate ONLY the provided Clinical Patient Report and Target Claim.\n"
"- Do not use external medical knowledge or add assumptions.\n"
"- Mark YES if the patient report explicitly mentions the clinical information in the claim or provides a direct semantic paraphrase.\n"
"- Ignore minor differences in wording, grammar, tense, pronouns, or non-essential details.\n"
"- Mark NO if the claim contains information, interpretation, diagnosis, comparison, or meaning that is not supported by the patient report.\n"
"- For multi-part claims, mark YES only when all clinically important information is covered in the patient report.\n\n"

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


few_shot_user = """ Doctor: Please tell us about any problems with your child’s vision or eyes.
Parent: She has trouble seeing the board in class, so we're concerned her eyesight might not be perfect.

Doctor: Please tell us about problems with your child's hearing or ears.
Parent: There's fluid in her ear, which has been a recurring issue.

Doctor: Please tell us about the problems with your child’s teeth.
Parent: Her teeth break or chip very easily, even with normal eating and drinking.

Doctor: Please tell us about any problems with your child’s heart.
Parent: Doctors have mentioned a structural abnormality in her heart.

Doctor: Please tell us about any problems with your child's hormones.
Parent: N/A.

Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
Parent: She gets exhausted quickly when playing or running around, more so than her friends.

Doctor: Please tell us about any problems with your child's kidneys.
Parent: N/A.

Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
Parent: N/A.

Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
Parent: Bowel movements have been a problem for her, causing discomfort and irregularity.

Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above.
Parent: Overall, she seems a bit behind other kids her age in terms of growth and development.
"""

few_shot_assistant = """
[
  "The patient has trouble seeing the board in class.",
  "The patient has fluid in her ear.",
  "The patient has recurring fluid in her ear.",
  "The patient's teeth break or chip easily.",
  "The patient has a structural abnormality in her heart.",
  "The patient gets exhausted quickly when playing or running around.",
  "The patient gets tired more easily than other children her age.",
  "The patient has bowel movement difficulties causing discomfort and irregularity.",
  "The patient is behind other children her age in terms of growth and development."
]
"""


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



        consultation_text = data.get("input", "")

        patient_report = data.get("output", "")

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
            f"Example Doctor-Parent Consultation:\n\n"
            f"{few_shot_user}"
    },

    {
        "role": "assistant",
        "content": few_shot_assistant
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


            "row_index": idx,


            "consultation": consultation_text,


            "patient_report": patient_report,



            "extracted_claims": claims_list,



            "summary_counts": {

                "total_claims": total_claims,

                "covered_claims": covered_count

            },



            "patient_coverage_score": round(
                coverage_score,
                4
            ),



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

        if idx == 7:
            break 

print(
    "\nClinical coverage evaluation completed successfully."
)
#3B
#Processed row 0 | Coverage: 0.9545 (21/22)
#Processed row 1 | Coverage: 0.8182 (9/11)
#Processed row 2 | Coverage: 0.7692 (10/13)
#Processed row 3 | Coverage: 0.4545 (10/22)
#Processed row 4 | Coverage: 0.7647 (13/17)

#72B
#Processed row 0 | Coverage: 0.9 (9/10)
#Processed row 1 | Coverage: 0.9444 (17/18)
#Processed row 2 | Coverage: 0.7059 (12/17)
#Processed row 3 | Coverage: 1.0 (11/11)
#Processed row 4 | Coverage: 0.9167 (11/12)
#Processed row 5 | Coverage: 0.9091 (30/33)
#Processed row 6 | Coverage: 0.8571 (12/14)
#Processed row 7 | Coverage: 0.8 (8/10)




#########################################################################################
#########################################################################################

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

REPORT_FILE = "factuality_scores_02.jsonl"#"patient_report_01.jsonl"
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
"- Extract only clinical facts claimed by the parent about the child.\n"
"- Do not infer or add medical knowledge.\n"
"- Ignore N/A responses.\n"
"- Each claim must represent one unique clinical fact.\n"
"- Output only valid JSON.\n"
)

SYSTEM_VALIDATION_PROMPT = (
    "You are a clinical summary coverage evaluator. "
    "Your task is to evaluate whether a generated patient summary covers a clinical claim extracted from a doctor-parent consultation.\n\n"

    "Rules:\n"
    "- Evaluate ONLY the provided Clinical Patient Report and Target Claim.\n"
    "- Do not use external medical knowledge or add assumptions.\n"
    "- Mark YES if the patient report contains a semantically equivalent clinical fact, regardless of wording differences.\n"
    "- Mark NO if the patient report does not contain a semantically equivalent clinical fact supporting the claim."
    "Examples:\n"
    '- "tooth pain when eating cold foods" ↔ "tooth sensitivity to cold foods" → YES\n'
    '- "valve issue" ↔ "heart valve abnormality" → YES\n'
    '- "delayed milestones" ↔ "takes longer to achieve milestones" → YES\n'
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


#############################################################
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
