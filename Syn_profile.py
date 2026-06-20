import json
import random

# ==========================================================
# CONFIG
# ==========================================================

NUM_PROFILES = 20
OUTPUT_FILE = "kbg_patient_profiles_15.json"

# ==========================================================
# SYMPTOM POOLS
# ==========================================================

VISION = ["Astigmatism", "Myopia", "Hypermetropia", "Squint"]

HEARING = [
    "otitis media",
    "Glue ear",
    "fluid in the ear"
    
    
]

TEETH = ["Large front teeth", "Weak enamel"]

HEART = ["Valve abnormality", "Septal defect", "Structural heart abnormality"]

NEUROLOGICAL = ["Seizures", "Epilepsy", "Hypotonia", "Tethered cord"]

KIDNEYS = []  # kept empty intentionally (KBG rarely has kidney issues)

IMMUNE = [] #no immune

GASTROINTESTINAL = [
    "Feeding difficulties",
    "Poor feeding in infancy",
    "Feeding aversion",
    "Bowel problems"
]

OTHER_MALE = [
    "Scoliosis",
    "Hip dysplasia",
    "Short fingers",
    "Speech delay",
    "Delayed fontanelle closure",
    "Undescended testes",
    "Palatal abnormalities"
]

OTHER_FEMALE = [
    "Scoliosis",
    "Hip dysplasia",
    "Short fingers",
    "Speech delay",
    "Delayed fontanelle closure",
    "Palatal abnormalities"
]

# ==========================================================
# SEVERITY
# ==========================================================

SEVERITY_LEVELS = ["Mild", "Moderate", "Severe"]

# ==========================================================
# SAFE SYMPTOM SAMPLER
# ==========================================================

def sample_symptoms(symptoms, max_count):
    """
    Safely sample symptoms and attach severity.
    Handles empty lists without crashing.
    """
    if not symptoms:
        return []

    count = random.randint(1, min(max_count, len(symptoms)))
    chosen = random.sample(symptoms, count)

    return [
        {
            "symptom": s,
            "severity": random.choice(SEVERITY_LEVELS)
        }
        for s in chosen
    ]

# ==========================================================
# PROFILE GENERATOR
# ==========================================================

def generate_profile(pid):
    gender = random.choice(["Male", "Female"])
    if gender == "Male":
        other_medical = sample_symptoms(OTHER_MALE, 2)
    else:
        other_medical = sample_symptoms(OTHER_FEMALE,2)
        
    
    return {
        "patient_id": pid,
        "age": random.randint(3, 9),
        "gender": gender,
        "vision": sample_symptoms(VISION, 1),
        "hearing": sample_symptoms(HEARING, 1),
        "teeth": sample_symptoms(TEETH, 1),
        "heart": sample_symptoms(HEART, 1),
        "neurological": sample_symptoms(NEUROLOGICAL, 1),
        "kidneys": sample_symptoms(KIDNEYS, 1),
        "immune": sample_symptoms(IMMUNE, 1),
        "gastrointestinal": sample_symptoms(GASTROINTESTINAL, 1),
        "other_medical": other_medical
    }

# ==========================================================
# GENERATE PROFILES
# ==========================================================

profiles = []

for i in range(1, NUM_PROFILES + 1):
    profiles.append(generate_profile(i))

# ==========================================================
# SAVE OUTPUT
# ==========================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=4, ensure_ascii=False)

# ==========================================================
# PRINT SAMPLE
# ==========================================================

print(f"\nGenerated {NUM_PROFILES} patient profiles successfully.")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSample profile:\n")
print(json.dumps(profiles[0], indent=4))





