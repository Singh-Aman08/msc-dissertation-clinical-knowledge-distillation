import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "qwen25_1.5b_weighted_sft_final"
INPUT_FILE = "testing_consultations_06.jsonl"
OUTPUT_FILE = "patient_report_[weighted_sft_qwen25_1.5b]_06.jsonl"
HF_TOKEN = "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"

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. This pipeline requires hardware acceleration."
    )

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("Base model loaded:", BASE_MODEL_ID)
print("LoRA adapter loaded:", ADAPTER_PATH)

KBG_CONTEXT = """ What is KBG syndrome?
KBG syndrome was first described in 1975, and its name is derived from the initials of the first three patients reported with the condition.  People with KBG syndrome have a characteristic (and sometimes subtle) facial appearance, very large permanent teeth, and variable degrees of developmental  delay, learning difficulties and behavioural differences. Because the facial features can be subtle and are not always present, the diagnosis may not be  made until the permanent teeth have come through. Other features seen in  some affected individuals include conductive hearing loss, undescended testes in boys, seizures, skeletal anomalies and short stature. KBG syndrome is caused by changes (variants) in,  or a deletion of, the ANKRD11 gene in chromosome  16 (band q24.3). Most affected people are the first person in their family to carry the gene change, but a small proportion have inherited it from a parent, who is likely to have features of KBG syndrome. The condition affects boys and girls, and there are both mildly and more significantly affected individuals of both sexes. However, there appear to be some reports of more affected males than females but the reason for this is unclear.
Most people with KBG syndrome have:
A degree of developmental delay and some element of behavioural differences. Large permanent upper middle teeth (macrodontia of upper central incisors). Characteristic facial appearance: a triangular-shaped face; wide-spaced eyes and thick eyebrows, which sometimes join in the centre (synophrys). Short fingers (brachydactyly) with curved 5th finger (clinodactyly)
How common is KBG syndrome?
KBG Syndrome is rare and thought to affect several hundred people worldwide. It is likely that many people are not diagnosed because many of the features can  be mild in those with a change (variant) in ANKRD11, including the degree of learning (intellectual) disability. KBG Syndrome is one of the most prevalent causes of syndromic developmental delay.  This guide is designed to help families and healthcare professionals looking  after people affected by KBG syndrome. It contains information about the cause,  the ways in which it can affect people and suggestions about the help and management that can benefit people with this condition. 
What causes KBG syndrome?
KBG syndrome is caused by one copy of the ANKRD11 gene not functioning  properly. This may be due to a change within the gene that disrupts its function,  or to the loss (deletion) of the whole gene or part of it. The other copy is  unaffected. 
Why did this happen?
When children are conceived, their parents’ genetic material (DNA) is copied in  the egg and sperm that makes a new child. The biological copying method is not  perfect and occasionally random, rare changes occur in the genetic code of children that are not seen in the DNA of their parents.  KBG syndrome occurs when one of these random, rare changes affects the ANKRD11 gene in chromosome 16. This happens naturally and is not due to the biological parents’ diet, environment or lifestyle. In most people with KBG syndrome, the genetic change was a random (or “de novo”) change, meaning the change occurred for the first time in that family in the affected individual. 
Occasionally, one parent may have the same change (variant) and pass it on to their child. No one should be blamed for variants in their DNA and no parent is at fault when a new DNA change occurs in their child.
Can it happen again?
The possibility of having another child affected by a rare gene disorder depends on the genetic code of the parents. In most families, the genetic change has happened for the first time in the child with KBG syndrome. We call this a ‘de novo’ change. In this situation, when the parents are unaffected, the chances of  having another child with the same condition are very low (usually less than 1%). 
One reason why there is some residual chance of recurrence is due to a rare phenomenon called germline mosaicism. This is when a parent carries a genetic change, but it is limited to some of their egg or sperm cells. The genetic change would not, therefore, be detected in the parents’ blood tests. Unique publishes a short general guide to mosaicism that covers this phenomenon. If a parent has KBG syndrome, the chances of passing the condition on to a child are much higher at 50%, or 1 chance in 2, as the parent could either pass on their altered copy of the gene or the unaffected copy. This inheritance pattern is called autosomal dominant (because the change is on an autosomal chromosome and an outcome can be seen if only one copy of the gene is altered). Unique publishes a separate guide to single gene disorders -autosomal dominant inheritance. Each family situation is different, and a clinical geneticist or genetic counsellor can give specific advice for your family.
Development:
Developmental delay has been reported in almost all children with KBG syndrome. The degree of delay ranges from mild (in most) to severe (in few). Developmental “milestones”, including rolling, sitting, walking, playing with toys, using cutlery, using zips and buttons, and toilet training, are often delayed, although there is a wide range of eventual ability, with some children acquiring mobility and other skills around the same age as “typical” children and others showing more obvious delay. 
#Learning:
Children with KBG syndrome typically need extra help in school, though most go to a mainstream primary school. The extra demands of mainstream secondary school may prove too challenging, and children may transfer to special schooling or remain in mainstream schools with Educational Health Care Plans (EHCP). Those with chromosome deletions may have more significant problems, which are probably related to other genes which are also deleted. There have been reports of children with small ANKRD11 deletions or variants who have no development delay. A very small number of individuals have been described as having no learning difficulties. 
#Behaviour: 
People with KBG syndrome often have behavioural differences such as autism spectrum disorder (ASD), ADHD or anxiety. Medication(s) may be appropriate in some circumstances.
#Speech:
The vast majority of people with KBG syndrome learn to speak but speech delay is very common. Hearing loss and subtle palate problems can worsen speech delay. Differences in the pitch and quality of the voice can also be observed.
#Using their hands:
The fine motor development necessary for skills such as playing with toys and using cutlery, zips and buttons is frequently delayed, but typically full function is attained.
Growth:
Babies with KBG are usually within the expected weight range at birth but grow more slowly in childhood. Typically, children are around the 2nd-25th centile for height on their age-appropriate growth chart but there are reports of people with KBG syndrome with average and above average height. A small group will have height below the 0.4th centile. For this group it would be important to monitor their growth. Currently there is a lack of good evidence around the use of growth hormone in this condition and it is not licensed for this condition in the UK. It is likely that more evidence will become available. 
Growing up: 
Most people with KBG syndrome will enter puberty within the typical age range. For a small number, signs of puberty start at an earlier age than expected (“precocious puberty”) and these children should see a paediatrician for review. There are varying levels of educational attainment. Some children stay in mainstream education and obtain qualifications, but others may need more support and some benefit from a specialist setting. A small number of people have gone on to further education. Adults with KBG Syndrome have varying levels of independence. Some have gone on to have families of their own and manage to run their own households and work. Others continue to live with their parents or in supported settings. Some may live independently but require some support from family or friends with certain tasks. Levels of employment and the nature of employment varies but many do undertake some form of paid employment. The features and daily life in adulthood for those with KBG syndrome are currently the subject of research. We hope to update this section with further information once these studies have been completed. 
Medical concerns:
The number of medical concerns a person with KBG syndrome will have is extremely variable and impossible to predict. Below are some of the more common features that could be observed in affected individuals:
#Seizures:
Around 20-40% of people are reported as having seizures, of varying types.  Seizures typically respond to the usual anti-epileptic treatments. In a small number of cases epilepsy can be difficult to treat. 
#Hearing:
A significant proportion of people have recurrent otitis media or glue ear (a build -up of fluid in the ear), which can cause conductive hearing loss, where sounds are unable to pass into the inner ear. Many children have required multiple aeration tube (grommet) insertions to relieve pressure inside the ear, and some have still required hearing aids. Hearing should be carefully and regularly checked during the first few years. Many children have speech delay which may be linked to their hearing concerns.
#Eyesight:
People with KBG syndrome are more likely to have vision problems such as astigmatism (the front of the eye is not perfectly rounded, causing blurry vision), and short- or long sightedness (myopia and hypermetropia). Squint (strabismus), where the eyes do not look in the same direction, can occasionally be a feature. Treatment can include patching, exercises, glasses, and surgery to bring the eyes into line.
#Undescended testes:
Many boys with KBG syndrome are born with testicles that have not completed their journey from the tummy (abdomen) to the sack (scrotum) (cryptorchidism).  In some boys, the testes descend in due course on their own, but if they do not, they can be brought down and fixed in the scrotum with a simple surgical operation (orchidopexy).
#Skeletal:
Some people with KBG syndrome have an unusual structure of their spinal bones which can give rise to an increased curvature of the spine (scoliosis). Babies may experience a delayed closure of the soft spot on their head (anterior fontanelle). Many people with KBG syndrome have short fingers (brachydactyly) with a curvature of the 5th fingers (clinodactyly). A small number of children have had hip dysplasia, in which the hip joints are easily dislocated.
#Teeth:
As well as large front teeth, there can be a variety of other dental concerns. Weak enamel is frequently seen, and careful brushing is very important. It is important for children with KBG syndrome to have regular dental check-ups. For those with significant sensory issues this should be with a specialist SEN dentist.
#Feeding:
Many babies have feeding difficulties and some require short-term nasogastric tube feeding to supplement oral feeds. It is Unique’s experience that lack of interest in feeding can be significant and long-lasting. A very small number of children have required longer-term tube feeding.
#Cardiac:
Most children with KBG syndrome do not have a very serious heart condition. However, a study of 40 European patients showed that 15/40 (38%) had heart conditions. These included changes to the heart valves, holes in the heart chambers and structural changes. A few required surgery but some just needed monitoring. 
#Tethered cord:
One study found that 11% of children with KBG syndrome have tethered cord, which means that the spinal cord cannot move freely. This can cause pain and weakness in the legs, urinary and bowel issues, sacral dimple and an unusual gait. Most children with KBG syndrome do not have these issues. 
#Palate: 
While most do not have very significant anomalies of the roof of the mouth (palate), a proportion have subtler anomalies. These can affect speech development and feeding, meaning referral for specialist review may be appropriate in some cases.
Management recommendations:
Regular dental check-ups. Regular hearing reviews to age 5 (even if earlier reviews give a clear response). Eyesight (ophthalmology) review. Check position of testes in boys. Consider a palate review (particularly if there are feeding difficulties or speech concerns). Referral for a cardiac review (including echo and ECG) following diagnosis. If nothing is found (or already done) this does not need to be repeated. Consider a skeletal review (X-ray of the wrist (to determine bone age), hip, spine and skull) in children following diagnosis. Any concerns around asymmetric hip creases in infancy and/or asymmetric or painful gait should prompt medical review. Consider review and investigation for tethered cord (MRI) where clinical concerns arise on an individual basis (especially if sacral dimple is present). Monitor growth velocity: if height is below the 2nd centile consider referral for endocrine investigations on an individual basis and within context of familial heights Consider physiotherapy, occupational therapy, speech therapy and behavioural therapy.
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pass

print(f"Reading interactions from {INPUT_FILE}...")
print(f"Streaming report outputs straight to: {OUTPUT_FILE}")

conversation_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    count = 0
    for line in infile:
        if not line.strip():
            continue
        count += 1   
        record = json.loads(line)
        conversation_count += 1
        patient_id = record.get("patient_id", conversation_count)
        transcript = record.get("consultation", "")
        
        messages = [
            {
                "role": "system",
                "content": ("""
You are an expert clinical patient report synthesis engine specialising in rare genetic syndromes.

Transform the doctor–parent consultation into a structured Clinical Patient Report using only:
1. The consultation transcript.
2. The provided clinical reference guidelines.

Before writing the report, internally:
1. Extract all patient findings.
2. Assign each finding to its most appropriate clinical section.
3. Generate the final report.
Do not output this internal analysis.

Rules:
- Generate a concise, accurate clinical report.
- Do not add unsupported information or assumptions.
- "How this affects the patient" must contain only patient-specific information from the consultation.
- "How this affects others with the syndrome" must contain only relevant syndrome-level information from the clinical reference guidelines.
- Assign each symptom exclusively to its most appropriate clinical category.
- Provide recommendations only for reported symptoms. List them as bullet points, with each recommendation directly addressing the patient’s identified symptoms.
- If any information is unavailable for a particular section, write exactly "N/A" and nothing else.
- Ensure the final report follows all instructions before responding.

Use clear professional clinical language.

Output MUST strictly follow this exact markdown structure:

# CLINICAL PATIENT REPORT

## 1) RESPIRATORY

a) How this affects the patient:
b) How this affects others with the syndrome:

## 2) CARDIOLOGY

a) How this affects the patient:
b) How this affects others with the syndrome:

## 3) GASTROENTEROLOGY

a) How this affects the patient:
b) How this affects others with the syndrome:

## 4) IMMUNOLOGY

a) How this affects the patient:
b) How this affects others with the syndrome:

## 5) NEUROLOGY

a) How this affects the patient:
b) How this affects others with the syndrome:

## 6) EAR NOSE THROAT

a) How this affects the patient:
b) How this affects others with the syndrome:

## 7) OPHTHALMOLOGY AND VISION

a) How this affects the patient:
b) How this affects others with the syndrome:

## 8) DERMATOLOGY

a) How this affects the patient:
b) How this affects others with the syndrome:

## 9) DENTAL

a) How this affects the patient:
b) How this affects others with the syndrome:

## 10) EDUCATION

a) How this affects the patient:
b) How this affects others with the syndrome:

## 11) BEHAVIOUR AND DEVELOPMENT

a) How this affects the patient:
b) How this affects others with the syndrome:

## 12) SKELETAL

a) How this affects the patient:
b) How this affects others with the syndrome:

## 13) RECOMMENDATIONS FOR SCREENING AND TREATMENTS
"""


              )
            },
            {
                "role": "user",
                "content": (
                    f"=== CLINICAL REFERENCE GUIDELINES ===\n{KBG_CONTEXT}\n\n"
                    f"=== VERBATIM DOCTOR-PARENT CONVERSATION ===\n{transcript}\n\n"
                    "Generate the complete clinical report following the strict 13-category schema layout."
                )
            }
        ]
        

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=2048,
                temperature=0.3, # Lower temperature forces higher adherence to facts and logic rules
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_report = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True)
        
        # Structure distillation instruction template mapping row
        distillation_record = {
            "instruction": "Analyze this clinical interview dialogue, deduce the primary genetic condition using reference guidelines, and compile a structured patient report with targeted management recommendations.",
            "input": transcript,
            "output": generated_report.strip()
        }
        
        # Stream result straight to file
        with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(distillation_record, ensure_ascii=False) + "\n")
            
        #if count ==4:
            #break
        

print(f"\nPipeline successfully complete! {conversation_count} reports compiled inside: {OUTPUT_FILE}")

