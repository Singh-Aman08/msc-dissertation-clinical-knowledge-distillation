import json
import random

NUM_PROFILES = 20
OUTPUT_FILE = "kbg_patient_profile_current.json"
VISION = [
    "Astigmatism",
    "Myopia",
    "Hypermetropia",
    "Squint",
    "Struggles to see the board at school",
    "Often squints when looking at things",
    "Sits very close to the TV or books",
    "Complains that things look blurry",
    "Rubs eyes frequently when focusing"
]

HEARING = [
    "otitis media",
    "Glue ear",
    "fluid in the ear",
    "Doesn't always respond when called",
    "Asks people to repeat themselves often",
    "Turns volume of TV very high",
    "Seems to hear some sounds but not others clearly",
    "Gets frustrated in noisy places"
]
    
TEETH = [
    "Large front teeth",
    "Weak enamel",
    "Teeth break or chip easily",
    "Complains of tooth sensitivity with cold food",
    "Needs frequent dental visits",
    "Has trouble chewing harder foods"
]

HEART = [
    "Valve abnormality",
    "Septal defect",
    "Structural heart abnormality",
    "Gets tired more quickly than other children",
    "Needs regular heart check-ups",
    "Breathes faster during play than expected",
    "Can't keep up with physical activity for long"
]

NEUROLOGICAL = [
    "Seizures",
    "Epilepsy",
    "Hypotonia",
    "Tethered cord",
    "Slower to reach milestones like walking or sitting",
    "Falls more often than other children",
    "Seems less coordinated when moving",
    "Gets tired quickly during physical activity",
    "Has episodes of unusual staring or blank spells"
]
KIDNEYS = []

IMMUNE = [] 

GASTROINTESTINAL = [
    "Feeding difficulties",
    "Poor feeding in infancy",
    "Feeding aversion",
    "Bowel problems",
    "Mealtimes are often a struggle",
    "Very picky with food textures",
    "Often refuses to eat certain foods",
    "Constipation issues",
    "Slow weight gain compared to peers"
]


OTHER_MALE = [
    "Scoliosis",
    "Hip dysplasia",
    "Short fingers",
    "Speech delay",
    "Delayed fontanelle closure",
    "Undescended testes",
    "Palatal abnormalities",
    "Not speaking as much as expected for age",
    "Difficulty making himself understood",
    "Noticeably different posture when standing or walking",
    "Seems behind other children in development",
    "Struggles with clarity when trying to talk",
    "Moves a bit differently compared to peers",
    "Takes longer to reach milestones than expected" ]

OTHER_FEMALE = [
    "Scoliosis",
    "Hip dysplasia",
    "Short fingers",
    "Speech delay",
    "Delayed fontanelle closure",
    "Palatal abnormalities",
    "Not speaking as much as expected for age",
    "Difficulty making himself understood",
    "Noticeably different posture when standing or walking",
    "Seems behind other children in development",
    "Struggles with clarity when trying to talk",
    "Moves a bit differently compared to peers",
    "Takes longer to reach milestones than expected" ]

PATIENT_PERSONAS = [
    "brief_reassured",  
    "detailed_observant",       
    "anxious_concerned",        
    "matter_of_fact",           
    "confused_unsure",          
    "overwhelmed_parent",       
    "overprotective_parent",        
    "highly_educated_parent",       
    "low_health_literacy",  
    "first_time_parent",       
    "experienced_parent",       
    "emotionally_flat",        
    "detail_dumping_parent"     
]

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
        }
        for s in chosen
    ]
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
        "patient_persona": random.choice(PATIENT_PERSONAS), 
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


profiles = []

for i in range(1, NUM_PROFILES + 1):
    profiles.append(generate_profile(i))


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=4, ensure_ascii=False)


print(f"\nGenerated {NUM_PROFILES} patient profiles successfully.")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSample profile:\n")
print(json.dumps(profiles[0], indent=4))





