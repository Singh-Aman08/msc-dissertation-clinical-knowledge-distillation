import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PROFILE_FILE = "kbg_patient_profile_current.json"
OUTPUT_FILE = "kbg_synthetic_conversations-9_llama.jsonl" # Changed to JSONL for progressive saving
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
                "You generate realistic doctor-parent consultations for children with KBG syndrome.\n\n"
                """Rules:

                - The Doctor MUST ask the exact 10 questions provided below, in order, verbatim.
                - Generate the complete consultation from start to finish. The output is incomplete unless all 10 doctor questions and all 10 corresponding parent answers are present and answered in order.
                - Write as a real parent speaking during a clinic appointment.
                - The parent persona MUST be strictly followed throughout the entire consultation.
                - Use the "patient_persona" field from the profile to determine the parent's communication style.
                - Every parent response must reflect the tone, communication style, and behaviour of the assigned persona.
                - Parents may use a mixture of everyday language but should primarily describe what they notice in daily life.
                - Do not assume that every parents know clinical terminology. Translate symptoms from the profile into realistic parent language whenever possible and avoid unnecessary medical jargon.
                - Focus on practical effects, observations, and concerns rather than simply repeating symptom names.
                - Treat every consultation as a different family.
                - Avoid repetitive sentence structures and repeated opening phrases.
                - Vary vocabulary, sentence length, level of detail, and speaking style.
                - Use natural, conversational language while strictly following all structural constraints.
                - All 10 doctor questions MUST be answered in order without omission or reordering.
                - If a category contains no symptoms or an empty list [], provide a brief natural response indicating no concerns at present.
                - Each parent response must contain exactly 2–3 short sentences.

                The 10 Verbatim Doctor Questions:
                1. Doctor: Please tell us about any problems with your child’s vision or eyes.
                2. Doctor: Please tell us about problems with your child's hearing or ears.
                3. Doctor: Please tell us about the problems with your child’s teeth.
                4. Doctor: Please tell us about any problems with your child’s heart.
                5. Doctor: Please tell us about any problems with your child's hormones.
                6. Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
                7. Doctor: Please tell us about any problems with your child's kidneys.
                8. Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
                9. Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
                10. Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above."""
               

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
    "patient_persona": "detailed_observant",
    "vision": [],
    "hearing": [{"symptom": "Speech delay"}],
    "teeth": [
        {"symptom": "Weak enamel"},
        {"symptom": "Large front teeth"}
    ],
    "heart": [{"symptom": "Valve abnormality"}],
    "neurological": [
        {"symptom": "Developmental delay"},
        {"symptom": "Tethered cord"}
    ],
    "kidneys": [],
    "immune": [],
    "gastrointestinal": [],
    "other_medical": [
        {"symptom": "Palatal abnormalities"},
        {"symptom": "Undescended testes"}
    ]
}
Generate the consultation.
"""
        },
        {
            "role": "assistant",
            "content": """
Doctor: Please tell us about any problems with your child’s vision or eyes.
Parent: We haven't noticed any issues with his eyesight. He seems to spot things normally and doesn't struggle with seeing objects or people.

Doctor: Please tell us about problems with your child's hearing or ears.
Parent: We don't think hearing is the problem, but he's quite behind when it comes to talking. He understands some of what we say, but getting words out is much harder for him than other children his age.

Doctor: Please tell us about the problems with your child’s teeth.
Parent: His front teeth look much bigger than expected for his age. We've also been told the teeth are quite fragile, so we have to be careful with brushing and dental check-ups.

Doctor: Please tell us about any problems with your child’s heart.
Parent: The doctors found an issue with one of the valves in his heart. We go for regular check-ups, and so far it hasn't stopped him from doing his usual activities.

Doctor: Please tell us about any problems with your child's hormones.
Parent: Nothing has come up in that area so far. We haven't been told about any hormone-related concerns.

Doctor: Please tell us about any neurological problem your child has (brain, nerves, muscle tone, fits).
Parent: He's been much slower to reach milestones than other children. We've also been told there's a problem affecting the lower part of his spine, and that's something the specialists keep an eye on.

Doctor: Please tell us about any problems with your child's kidneys.
Parent: No kidney problems that we're aware of. Everything has been fine so far.

Doctor: Please tell us about any problems with your child’s immune system (ability to fight infection and vaccination responses).
Parent: We haven't noticed any unusual problems with infections. Vaccinations and common illnesses have all been fairly straightforward.

Doctor: Please tell us about problems with your child’s gastrointestinal system (feeding, stomach, bowel).
Parent: No major concerns there. Eating and bowel habits have generally been normal.

Doctor: Please use this box to tell us about any other medical problems that you haven’t been able to mention above.
Parent: The roof of his mouth developed differently, which has made speaking a bit harder for him. He also has one testicle that hasn't moved down properly, and that's being monitored.
"""
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

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            
            # Critical adjustments for strict data extraction tasks
            temperature=0.75,  
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