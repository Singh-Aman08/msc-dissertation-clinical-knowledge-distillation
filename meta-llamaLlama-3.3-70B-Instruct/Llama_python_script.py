import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PROFILE_FILE = "kbg_patient_profiles_15-4.json"
OUTPUT_FILE = "kbg_synthetic_conversations-5_llama.jsonl" # Changed to JSONL for progressive saving
HF_TOKEN = "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"

# ==========================================================
# GPU CHECK
# ==========================================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

# ==========================================================
# LOAD PROFILES
# ==========================================================
with open(PROFILE_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)

print(f"Loaded {len(profiles)} patient profiles")

# ==========================================================
# LOAD MODEL
# ==========================================================
print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = HF_TOKEN )

# Ensure correct padding configurations for Qwen text generation
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading model onto GPU")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="cuda",
    token = HF_TOKEN,
    low_cpu_mem_usage=True
    
)
model.eval()

# Clear/Initialize the target output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pass

print(f"Starting pipeline. Streaming outputs straight to: {OUTPUT_FILE}")

# ==========================================================
# GENERATION LOOP
# ==========================================================
for idx, profile in enumerate(profiles):
    print(f"\nGenerating consultation {idx + 1}/{len(profiles)} (Patient ID: {profile.get('patient_id')})")

    messages = [
        {
            "role": "system",
            "content": (
                "You generate realistic, highly specific doctor-parent consultations for children with KBG syndrome.\n\n"
                "Rules:\n"
                "- The Doctor MUST ask the exact 10 questions provided below, in order, verbatim. Do not change a single word.\n"
                "- Generate the complete consultation from start to finish. The output is incomplete unless all 10 doctor questions and all 10 corresponding parent answers are present and answered in order."
                "- No greetings, introductions, or pleasantries.\n"
                "- No doctor follow-up questions or empathy transitions.\n"
                "- No medical advice, next steps, or summaries.\n"
                "- Use very natural, conversational language for the parent's voice.\n"
                "- Do not invent symptoms.\n"
                "- If a category contains no symptoms or an empty list [], write a short parent response indicating no current problems.\n"
                "- Each parent response must be exactly 2–3 short sentences.\n"
                "- Treat every consultation as a conversation with a different parent. Do not reuse common phrases, fixed templates, or repetitive response patterns across patients. Vary the wording, sentence structure, tone, and level of detail so that each consultation feels naturally written and distinct, even when patients share similar symptoms.\n"
                "- Parents may use a mix of medical terms and everyday language. They do not need to know or use exact medical terminology, so responses should naturally combine clinical terms (when available) with simple, real-world descriptions of the child’s symptoms and difficulties.\n"
                "- Output only the Doctor and Parent dialogue turns.\n\n"
                "The 10 Verbatim Doctor Questions:\n"
                "1. Doctor: Please tell us about any problems with your child’s vision or eyes.\n"
                "2. Doctor: Please tell us about problems with your child's hearing or ears.\n"
                "3. Doctor: Please tell us about the problems with your child’s teeth.\n"
                "4. Doctor: Please tell us about any problems with your child’s heart.\n"
                "5. Doctor: Please tell us about any problems with your child's hormones.\n"
                "6. Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).\n"
                "7. Doctor: Please tell us about any problems with your child's kidneys.\n"
                "8. Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).\n"
                "9. Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).\n"
                "10. Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above."
            )
        },
        {
            "role": "user",
            "content": """
PROFILE
{
    "patient_id": 1,
    "age": 4,
    "gender": "Male",
    "vision": [],
    "hearing": [{"symptom": "Speech delay", "severity": "Moderate"}],
    "teeth": [
        {"symptom": "Weak enamel", "severity": "Mild"},
        {"symptom": "Large front teeth", "severity": "Moderate"}
    ],
    "heart": [{"symptom": "Valve abnormality", "severity": "Moderate"}],
    "neurological": [
        {"symptom": "Developmental delay", "severity": "Severe"},
        {"symptom": "Tethered cord", "severity": "Moderate"}
    ],
    "kidneys": [],
    "immune": [],
    "gastrointestinal": [],
    "other_medical": [
        {"symptom": "Palatal abnormalities", "severity": "Moderate"},
        {"symptom": "Undescended testes", "severity": "Mild"}
    ]
}
Generate the consultation.
"""
        },
        {
            "role": "assistant",
            "content": """
Doctor: Please tell us about any problems with your child’s vision or eyes.
Parent: No problems with his vision have been noticed so far.

Doctor: Please tell us about problems with your child's hearing or ears.
Parent: Communication is affected because speech development is delayed. He struggles to express himself as clearly as other children.

Doctor: Please tell us about the problems with your child’s teeth.
Parent: His front teeth are larger than expected. We have to pay extra attention to dental care because the enamel is quite weak.

Doctor: Please tell us about any problems with your child’s heart.
Parent: A valve problem has been identified and is being monitored. At the moment it has not caused major difficulties.

Doctor: Please tell us about any problems with your child's hormones.
Parent: No hormone-related concerns have been reported.

Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
Parent: Development is noticeably delayed compared with other children. He also has a tethered cord which affects some aspects of his physical development.

Doctor: Please tell us about any problems with your child's kidneys.
Parent: No kidney problems have been reported.

Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
Parent: No concerns with his immune system at present.

Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
Parent: There are no significant feeding or digestive problems currently.

Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above.
Parent: He has a palate abnormality which affects speech. He also has an undescended testis that is being monitored.
"""
        },
        {
            "role": "user",
            "content": f"PROFILE\n\n{json.dumps(profile, indent=2)}\n\nGenerate the consultation."
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            
            # Critical adjustments for strict data extraction tasks
            temperature=0.1,  
            do_sample=True,   
            top_p=0.9,
            
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

print("\nAll generations finished successfully.")