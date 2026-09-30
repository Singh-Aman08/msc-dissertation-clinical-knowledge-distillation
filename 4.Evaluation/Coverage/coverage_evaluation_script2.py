import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

REPORT_FILE = "testing_Report_3B.jsonl"
OUTPUT_FILE = "Coverage_second_part_testing_01.jsonl"
HF_TOKEN = "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"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

SYSTEM_PROMPT = (
    "You are a clinical question generation assistant. "
    "Your task is to read a doctor-parent consultation and generate evaluation questions "
    "to assess whether the patient report provides:\n"
    "1. Background information about how the relevant feature affects other people with KBG syndrome.\n"
    "2. Relevant recommendations, screening, treatment, or management advice for that feature.\n\n"

    "Rules:\n"
    "- Use only information explicitly stated by the parent.\n"
    "- Ignore doctor's questions and all N/A responses.\n"
    "- Do not use external medical knowledge.\n"
    "- Do not infer missing information.\n"
    "- For each clinical feature mentioned by the parent:"
    "  • Generate a syndrome_background question.\n"
    "  • Generate a recommendation question.\n"
    "- Keep questions short, direct, and specific.\n"
    "- Return only valid JSON.\n"
)

few_shot_user = """  
Doctor: Please tell us about any problems with your child’s vision or eyes.
Parent: Honestly, I'm getting a bit overwhelmed with all these questions... Um, yes, his vision. He complains that things look blurry sometimes. I don't know if it's because he needs glasses or what.

Doctor: Please tell us about problems with your child's hearing or ears.
Parent: Oh, right. His ears. There's been some fluid in his ear for a while now. I've been meaning to take him to get it checked out.

Doctor: Please tell us about the problems with your child’s teeth.
Parent: Teeth... yes. He says his teeth hurt when he eats cold things like ice cream. I think it might be sensitive or something.

Doctor: Please tell us about any problems with your child’s heart.
Parent: His heart... well, when he plays, he breathes really fast. Faster than I expect, anyway. Is that normal?

Doctor: Please tell us about any problems with your child's hormones.
Parent: N/A.

Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
Parent: He gets tired really easily when running around outside. Not like other kids his age who can just keep going.

Doctor: Please tell us about any problems with your child's kidneys.
Parent: N/A.

Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
Parent: N/A.

Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
Parent: Food... yes. He's always been a bit slow to gain weight compared to his friends. And he's just so picky about what he eats.

Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above.
Parent: Speech... it's taking him longer to talk than others. And overall, he just seems behind where the other kids his age are. Developmentally, I mean. Sorry, I'm just feeling a bit... overwhelmed with all of this.
"""
few_shot_assistant = """ 
[
  {
    "question": "Does the report mention how vision problems affect others with the syndrome?",
    "category": "syndrome_background"
  },
  {
    "question": "Does the report include recommendations for vision problems?",
    "category": "recommendation"
  },
  {
    "question": "Does the report mention how ear problems affect others with the syndrome?",
    "category": "syndrome_background"
  },
  {
    "question": "Does the report include recommendations for ear problems?",
    "category": "recommendation"
  },
  {
    "question": "Does the report mention how dental problems affect others with the syndrome?",
    "category": "syndrome_background"
  },
  {
    "question": "Does the report include recommendations for dental problems?",
    "category": "recommendation"
  },
  {
    "question": "Does the report mention how heart problems affect others with the syndrome?",
    "category": "syndrome_background"
  },
  {
    "question": "Does the report include recommendations for heart problems?",
    "category": "recommendation"
  },
  {
    "question": "Does the report mention how neurological or developmental problems affect others with the syndrome?",
    "category": "syndrome_background"
  },
  {
    "question": "Does the report include recommendations for neurological or developmental problems?",
    "category": "recommendation"
  },
  {
    "question": "Does the report mention how feeding or growth problems affect others with the syndrome?",
    "category": "syndrome_background"
  },
  {
    "question": "Does the report include recommendations for feeding or growth problems?",
    "category": "recommendation"
  }
]

"""

SYSTEM_VALIDATION_PROMPT = (
"You are a clinical report coverage evaluator. "
"Your task is to determine whether a patient report contains information that answers the given question.\n\n"

"Rules:\n"
"- Use only information explicitly stated in the patient report.\n"
"- Do not use external medical knowledge.\n"
"- Do not infer missing information.\n"
"- Give YES only if the report provides the information requested in the question.\n"
"- Give NO if the information is missing, unclear, or not directly addressed in the report.\n"
"- For recommendation questions, give YES if the report provides any explicit recommendation, screening, monitoring, treatment, referral, therapy, or management advice relevant to the feature.\n"
"- For syndrome_background questions, check the section describing 'how this affects others with the syndrome' . Give YES only if this section explains how the feature affects people with KBG syndrome.\n"
"- Preserve the category field from each input question in the output.\n\n"

"Return ONLY valid JSON in the following format:\n"
"{\n"
'  "results": [\n'
'    {\n'
'      "question": "string",\n'
'      "category": "syndrome_background or recommendation",\n'
'      "covered": "YES"\n'
'    }\n'
"  ]\n"
"}"
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    token=HF_TOKEN
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

            return json.loads(
                repaired_json
            )

        except json.JSONDecodeError:

            return None


print(
    "\nStarting Clinical Coverage Evaluation...\n",
    flush=True
)

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

        consultation_text = data.get(
            "input",
            ""
        )

        patient_report = data.get(
            "output",
            ""
        )

        if not consultation_text:

            print(
                f"Skipping row {idx}: missing consultation",
                flush=True
            )

            continue

        if not patient_report:

            print(
                f"Skipping row {idx}: missing report",
                flush=True
            )

            continue

        
        print(
            f"Generating questions for row {idx}",
            flush=True
        )

        extraction_messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": few_shot_user
            },

            {
                "role": "assistant",
                "content": few_shot_assistant
            },

            {
                "role": "user",
                "content":
                "Generate coverage questions for the following consultation.\n\n"
                f"{consultation_text}"
            }

        ]

        raw_question_text = get_model_response(
            extraction_messages
        )

        question_list = extract_and_parse_json(
            raw_question_text
        )

        if not isinstance(
            question_list,
            list
        ):

            print(
                f"Question generation failed for row {idx}",
                flush=True
            )

            print(
                raw_question_text,
                flush=True
            )

            continue

        print(
            f"Generated {len(question_list)} questions for row {idx}",
            flush=True
        )

        print(
            f"Validating report for row {idx}",
            flush=True
        )

        validation_messages = [

            {
                "role": "system",
                "content": SYSTEM_VALIDATION_PROMPT
            },

            {
                "role": "user",
                "content":
                f"Patient Report:\n\n"
                f"{patient_report}\n\n"
                f"Questions:\n"
                f"{json.dumps(question_list, ensure_ascii=False, indent=2)}\n\n"
                "Evaluate every question against the report and return the results "
                "only in the required JSON format."
            }

        ]

        raw_validation_text = get_model_response(
            validation_messages
        )

        validation_json = extract_and_parse_json(
            raw_validation_text
        )

        if (
            not isinstance(validation_json, dict)
            or
            "results" not in validation_json
        ):

            print(
                f"Validation failed for row {idx}",
                flush=True
            )

            print(
                raw_validation_text,
                flush=True
            )

            continue

        coverage_results = validation_json["results"]

        covered_count = sum(
        1
        for item in coverage_results
        if str(
            item.get(
                "covered",
                "NO"
            )
        ).upper() == "YES"
    )

        total_questions = len(question_list)

        coverage_score = (
        covered_count / total_questions
        if total_questions > 0
        else 1.0
    )



        syndrome_results = [
        item for item in coverage_results
        if item.get("category") == "syndrome_background"
    ]

        recommendation_results = [
        item for item in coverage_results
        if item.get("category") == "recommendation"
    ]


        syndrome_covered = sum(
        1
        for item in syndrome_results
        if str(
            item.get(
                "covered",
                "NO"
            )
        ).upper() == "YES"
    )


        recommendation_covered = sum(
        1
        for item in recommendation_results
        if str(
            item.get(
                "covered",
                "NO"
            )
        ).upper() == "YES"
    )


        syndrome_total = len(syndrome_results)

        recommendation_total = len(recommendation_results)


        syndrome_score = (
        syndrome_covered / syndrome_total
        if syndrome_total > 0
        else 1.0
    )


        recommendation_score = (
        recommendation_covered / recommendation_total
        if recommendation_total > 0
        else 1.0
    )


        output_data = {

        "row_index": idx,

        "consultation": consultation_text,

        "patient_report": patient_report,

        "generated_questions": question_list,


        "summary_counts": {

            "total_questions": total_questions,

            "covered_questions": covered_count,

            "syndrome_background": {
                "total": syndrome_total,
                "covered": syndrome_covered
            },

            "recommendation": {
                "total": recommendation_total,
                "covered": recommendation_covered
            }

        },


        "coverage_scores": {

            "overall_coverage_score": round(
                coverage_score,
                4
            ),

            "syndrome_background_score": round(
                syndrome_score,
                4
            ),

            "recommendation_score": round(
                recommendation_score,
                4
            )

        },


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
        f"Overall: {round(coverage_score,4)} "
        f"({covered_count}/{total_questions}) | "
        f"Syndrome: {round(syndrome_score,4)} "
        f"({syndrome_covered}/{syndrome_total}) | "
        f"Recommendation: {round(recommendation_score,4)} "
        f"({recommendation_covered}/{recommendation_total})",

        flush=True

    )


    # Uncomment for debugging
        if idx == 5:
            break
print(
    "\nClinical Coverage Evaluation Completed Successfully.",
    flush=True
)


