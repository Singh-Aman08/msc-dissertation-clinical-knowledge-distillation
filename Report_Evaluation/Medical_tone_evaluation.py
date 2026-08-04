import json
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_FILE = "coverage_part1_01.jsonl"
OUTPUT_FILE = "medical_tone_scores_01.jsonl"
HF_TOKEN = "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"


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


SYSTEM_PROMPT = """
You are an expert clinical report extraction assistant.

Your task is to extract ONLY patient-specific clinical information from a clinical report.

Include:

- Every statement under sections titled "How this affects the patient".
- Every recommendation under sections titled "Recommendations for screening and treatment".

Exclude:

- Any information under "How this affects others with the syndrome".
- Prevalence statistics or general background medical information.

Rules:

1. Copy the text exactly as written in the report.
2. Preserve the original wording, punctuation, and sentence boundaries.
3. Do not merge, split, rewrite, summarize, or deduplicate statements.
4. Ignore sections whose content is "N/A".
5. Return the extracted statements in the same order that they appear in the report.
6. Return exactly one JSON object.
7. The value of "extracted_sections" must be a flat list containing only strings.
8. Do not include nested lists, nested JSON objects, repeated keys, explanations, or additional text outside the JSON.
9. Return only valid JSON.

Output format:

{
  "extracted_sections": [
    "...",
    "..."
  ]
}
"""

FEWSHOT_USER_EXTRACTION = """

# CLINICAL PATIENT REPORT

## 1) RESPIRATORY
a) How this affects the patient: N/A
b) How this affects others with the syndrome: N/A

## 2) CARDIOLOGY
a) How this affects the patient: The patient has a heart condition that requires regular check-ups at the cardiologist.
b) How this affects others with the syndrome: Around 38% of people with KBG syndrome have heart conditions, including changes to the heart valves, holes in the heart chambers, and structural changes.

## 3) GASTROENTEROLOGY
a) How this affects the patient: The patient experienced feeding difficulties as a baby, but this has since resolved.
b) How this affects others with the syndrome: Many babies with KBG syndrome have feeding difficulties and some require short-term nasogastric tube feeding to supplement oral feeds.

## 4) IMMUNOLOGY
a) How this affects the patient: N/A
b) How this affects others with the syndrome: N/A

## 5) NEUROLOGY
a) How this affects the patient: The patient has hypotonia, which means her muscles are too stiff or rigid, and also experiences speech delay.
b) How this affects others with the syndrome: Around 20-40% of people with KBG syndrome have seizures, and some may have neurological differences such as autism spectrum disorder (ASD) or ADHD.

## 6) EAR NOSE THROAT
a) How this affects the patient: The patient had glue ear when she was younger, which took several rounds of antibiotics to clear up.
b) How this affects others with the syndrome: A significant proportion of people with KBG syndrome have recurrent otitis media or glue ear, which can cause conductive hearing loss.

## 7) OPHTHALMOLOGY AND VISION
a) How this affects the patient: The patient has hypermetropia (long-sightedness) and has trouble seeing things up close.
b) How this affects others with the syndrome: People with KBG syndrome are more likely to have vision problems such as astigmatism, short- or long-sightedness, and squint.

## 8) DERMATOLOGY
a) How this affects the patient: N/A
b) How this affects others with the syndrome: N/A

## 9) DENTAL
a) How this affects the patient: The patient's teeth are brittle and break or chip easily.
b) How this affects others with the syndrome: Many people with KBG syndrome have dental concerns, including weak enamel and large front teeth.

## 10) EDUCATION
a) How this affects the patient: The patient experiences speech delay.
b) How this affects others with the syndrome: Children with KBG syndrome typically need extra help in school, and some may require special schooling or Educational Health Care Plans.

## 11) BEHAVIOUR AND DEVELOPMENT
a) How this affects the patient: The patient experiences speech delay and has hypotonia.
b) How this affects others with the syndrome: People with KBG syndrome often have behavioural differences such as autism spectrum disorder (ASD), ADHD, or anxiety.

## 12) SKELETAL
a) How this affects the patient: N/A
b) How this affects others with the syndrome: Some people with KBG syndrome have an unusual structure of their spinal bones, which can give rise to an increased curvature of the spine (scoliosis), and some have short fingers (brachydactyly) with a curvature of the 5th fingers (clinodactyly).

## 13) RECOMMENDATIONS FOR SCREENING AND TREATMENTS
* Regular dental check-ups to monitor the patient's brittle teeth
* Regular hearing reviews to monitor the patient's hearing and address any potential hearing loss
* Regular eyesight reviews to monitor the patient's vision and address any potential vision problems
* Regular check-ups at the cardiologist to monitor the patient's heart condition
* Speech therapy to address the patient's speech delay
* Physiotherapy and occupational therapy to address the patient's hypotonia and developmental delays

"""

FEWSHOT_ASSISTANT_EXTRACTION = """
{
  "extracted_sections": [
    "The patient has a heart condition that requires regular check-ups at the cardiologist.",
    "The patient experienced feeding difficulties as a baby, but this has since resolved.",
    "The patient has hypotonia, which means her muscles are too stiff or rigid, and also experiences speech delay.",
    "The patient had glue ear when she was younger, which took several rounds of antibiotics to clear up.",
    "The patient has hypermetropia (long-sightedness) and has trouble seeing things up close.",
    "The patient's teeth are brittle and break or chip easily.",
    "The patient experiences speech delay.",
    "The patient experiences speech delay and has hypotonia.",
    "Regular dental check-ups to monitor the patient's brittle teeth.",
    "Regular hearing reviews to monitor the patient's hearing and address any potential hearing loss.",
    "Regular eyesight reviews to monitor the patient's vision and address any potential vision problems.",
    "Regular check-ups at the cardiologist to monitor the patient's heart condition.",
    "Speech therapy to address the patient's speech delay.",
    "Physiotherapy and occupational therapy to address the patient's hypotonia and developmental delays."
  ]
}
"""

SYSTEM_VALIDATION_PROMPT = """
You are an expert evaluator of medical tone.

Evaluate the following patient-specific sections.

For each section, answer the following questions:

1. Professionalism:
Does this section use professional and appropriate clinical language?
(YES/NO)

2. Respectfulness:
Does this section describe the patient in a respectful and non-stigmatising manner?
(YES/NO)

3. Patient-centred communication:
Does this section communicate the patient's condition in a supportive and patient-focused manner?
(YES/NO)

Return only valid JSON.

"""

FEWSHOT_USER_VALIDATION = """
{
  "extracted_sections": [
    "The patient has a heart condition that requires regular check-ups at the cardiologist.",
    "The patient experienced feeding difficulties as a baby, but this has since resolved.",
    "The patient has hypotonia, which means her muscles are too stiff or rigid, and also experiences speech delay.",
    "The patient had glue ear when she was younger, which took several rounds of antibiotics to clear up.",
    "The patient has hypermetropia (long-sightedness) and has trouble seeing things up close.",
    "The patient's teeth are brittle and break or chip easily.",
    "The patient experiences speech delay.",
    "The patient experiences speech delay and has hypotonia.",
    "Regular dental check-ups to monitor the patient's brittle teeth.",
    "Regular hearing reviews to monitor the patient's hearing and address any potential hearing loss.",
    "Regular eyesight reviews to monitor the patient's vision and address any potential vision problems.",
    "Regular check-ups at the cardiologist to monitor the patient's heart condition.",
    "Speech therapy to address the patient's speech delay.",
    "Physiotherapy and occupational therapy to address the patient's hypotonia and developmental delays."
  ]
}
"""
FEWSHOT_ASSISTANT_VALIDATION = """
{
  "evaluations": [
    {
      "section": "The patient has a heart condition that requires regular check-ups at the cardiologist.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient experienced feeding difficulties as a baby, but this has since resolved.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient has hypotonia, which means her muscles are too stiff or rigid, and also experiences speech delay.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient had glue ear when she was younger, which took several rounds of antibiotics to clear up.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient has hypermetropia (long-sightedness) and has trouble seeing things up close.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient's teeth are brittle and break or chip easily.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient experiences speech delay.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "The patient experiences speech delay and has hypotonia.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "Regular dental check-ups to monitor the patient's brittle teeth.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "Regular hearing reviews to monitor the patient's hearing and address any potential hearing loss.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "Regular eyesight reviews to monitor the patient's vision and address any potential vision problems.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "Regular check-ups at the cardiologist to monitor the patient's heart condition.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "Speech therapy to address the patient's speech delay.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    },
    {
      "section": "Physiotherapy and occupational therapy to address the patient's hypotonia and developmental delays.",
      "professionalism": "YES",
      "respectfulness": "YES",
      "patient_centred_communication": "YES"
    }
  ]
}
"""


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected."
    )


print("Loading model...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    token=HF_TOKEN
)


model.eval()


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


print("Model loaded")

def generate_response(messages):
    # This returns a BatchEncoding dictionary object containing tensors
    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    
        
    
    inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(model.device)
    
    
    
    

    # Get the length of the prompt safely from the input_ids tensor
    prompt_length = inputs.input_ids.shape[1]

    with torch.inference_mode():
        output = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=2048,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

    # Cleanly slice off the prompt using the 1D index integer
    response = tokenizer.decode(
        output[0][prompt_length:],
        skip_special_tokens=True
    )

    del inputs
    del output
    torch.cuda.empty_cache()

    return response


def clean_json(response):

    response = response.strip()


    if "```json" in response:

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )


    start = response.find("{")
    end = response.rfind("}")


    if start == -1 or end == -1:
        return ""


    return response[start:end+1].strip()


def run_extraction(report):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": FEWSHOT_USER_EXTRACTION
        },

        {
            "role": "assistant",
            "content": FEWSHOT_ASSISTANT_EXTRACTION
        },

        {
            "role": "user",
            "content": report
        }

    ]


    response = generate_response(
        messages
    )


    try:

        cleaned_response = clean_json(response)

        print("\n========== RAW MODEL RESPONSE ==========")
        print(response)

        print("\n========== CLEANED JSON ==========")
        print(cleaned_response)

        result = json.loads(cleaned_response)

        return result


    except Exception as e:

        print(
            "Extraction JSON error:",
            e
        )

        print(
            "Raw model response was:"
        )

        print(response)

        return {
            "extracted_sections": []
        }

def run_tone_evaluation(extracted_sections):


    user_input = json.dumps(
        {
            "extracted_sections":
            extracted_sections
        },
        indent=2
    )


    messages = [

        {
            "role": "system",
            "content": SYSTEM_VALIDATION_PROMPT
        },

        {
            "role": "user",
            "content": FEWSHOT_USER_VALIDATION
        },

        {
            "role": "assistant",
            "content": FEWSHOT_ASSISTANT_VALIDATION
        },

        {
            "role": "user",
            "content": user_input
        }

    ]


    response = generate_response(
        messages
    )


    try:

        return json.loads(
            clean_json(response)
        )


    except Exception as e:

        print(
            "Tone JSON error:",
            e
        )


        return {
            "evaluations": []
        }


def calculate_scores(evaluations):

    if not evaluations:

        return {

            "professionalism_score": 0.0,

            "respectfulness_score": 0.0,

            "patient_centred_score": 0.0

        }


    total = len(evaluations)


    professionalism = 0
    respectfulness = 0
    patient_centred = 0



    for item in evaluations:


        if str(
            item.get(
                "professionalism",
                ""
            )
        ).upper() == "YES":

            professionalism += 1



        if str(
            item.get(
                "respectfulness",
                ""
            )
        ).upper() == "YES":

            respectfulness += 1



        if str(
            item.get(
                "patient_centred_communication",
                ""
            )
        ).upper() == "YES":

            patient_centred += 1



    return {

        "professionalism_score":
        round(
            professionalism / total,
            4
        ),


        "respectfulness_score":
        round(
            respectfulness / total,
            4
        ),


        "patient_centred_score":
        round(
            patient_centred / total,
            4
        )

    }


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as infile, \
open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as outfile:


    processed = 0
    skipped = 0
    extraction_failures = 0



    for idx, line in enumerate(infile):


        if not line.strip():
            continue



        data = json.loads(line)



        index = data.get(
            "index",
            idx
        )


        report = data.get(
            "report",
            ""
        )



        if not report:

            print(
                f"Skipping index {index}: missing report"
            )

            skipped += 1

            continue


        print(
            f"Starting extraction index {index}",
            flush=True
        )


        extraction_result = run_extraction(
            report
        )


        extracted_sections = extraction_result.get(
            "extracted_sections",
            []
        )



        if not isinstance(
            extracted_sections,
            list
        ):

            print(
                f"Skipping index {index}: extraction failed"
            )

            extraction_failures += 1

            continue



        if len(extracted_sections) == 0:

            print(
                f"Warning index {index}: no extracted sections"
            )

            extraction_failures += 1



        print(
            f"Extraction completed index {index}",
            flush=True
        )

        print(
            f"Starting tone evaluation index {index}",
            flush=True
        )


        tone_result = run_tone_evaluation(
            extracted_sections
        )



        evaluations = tone_result.get(
            "evaluations",
            []
        )



        if not isinstance(
            evaluations,
            list
        ):

            print(
                f"Skipping index {index}: tone evaluation failed"
            )

            skipped += 1

            continue



        print(
            f"Tone evaluation completed index {index}",
            flush=True
        )


        tone_scores = calculate_scores(
            evaluations
        )


        output_record = {


            "index": index,


            "report": report,


            "extracted_sections":
            extracted_sections,


            "tone_evaluations":
            evaluations,


            "medical_tone_scores":
            tone_scores

        }



        outfile.write(
            json.dumps(
                output_record,
                ensure_ascii=False
            )
            + "\n"
        )


        outfile.flush()



        processed += 1



        print(
            f"Processed index {index} | "
            f"Professionalism: {tone_scores['professionalism_score']} | "
            f"Respectfulness: {tone_scores['respectfulness_score']} | "
            f"Patient-centred: {tone_scores['patient_centred_score']}",
            flush=True
        )
        if idx == 5:
            break


print(
    "\nFinished medical tone evaluation successfully."
)


print(
    "Processed:",
    processed
)


print(
    "Skipped:",
    skipped
)


print(
    "Extraction failures:",
    extraction_failures
)
