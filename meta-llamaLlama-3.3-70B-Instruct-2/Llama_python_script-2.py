import json
import torch
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PROFILE_FILE = "kbg_profile_04.json"
OUTPUT_FILE = "syn_con_04.jsonl" 
HF_TOKEN = "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"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")


with open(PROFILE_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)

print(f"Loaded {len(profiles)} patient profiles")


print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = HF_TOKEN )

# Ensure correct padding configurations for Qwen text generation
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("Loading model onto GPU")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",     #cuda
    token = HF_TOKEN,
    low_cpu_mem_usage=True
    
)
model.eval()

# Clear/Initialize the target output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pass

print(f"Starting pipeline. Streaming outputs straight to: {OUTPUT_FILE}")


for idx, profile in enumerate(profiles):
    print(f"\nGenerating consultation {idx + 1}/{len(profiles)} (Patient ID: {profile.get('patient_id')})")

    messages = [
        {
            "role": "system",
            "content": (
                "You generate realistic doctor–parent consultations for children with KBG syndrome."

                """ Rules:

                - The doctor MUST ask the exact 10 questions provided below, in the given order, without modification.
                - Generate the complete consultation from start to finish. Every doctor question must have exactly one corresponding parent response.
                - Base every parent response only on the provided patient profile. Do not add, remove, or alter patient information.
                - Use the "parent_persona" field to determine the parent's communication style, tone, and vocabulary.
                - Each parent response must contain exactly 2-5 short sentences depending on the parent_persona.
                - If the corresponding category in the patient profile is empty or contains no symptoms, the parent's entire response must be exactly N/A. Do not add any explanation, punctuation, extra words, or follow-up sentences. The response must consist only of N/A.
                - Treat every consultation as a different family by varying sentence structure, wording, and expression while preserving the underlying patient information.
                - Do not introduce symptoms, diagnoses, or concerns that are not present in the patient profile.

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
    "parent_persona": "detailed_observant",
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            
            # Critical adjustments for strict data extraction tasks
            temperature=0.75,  
            do_sample=True,   
            top_p=0.9,
            repetition_penalty=1.12,
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
        
    # if idx == 10:
    #     break
print("\nAll generations finished successfully.")