---
base_model: unsloth/Qwen2.5-1.5B-Instruct
library_name: peft
model_name: qwen25_1.5b_normal_sft_final
tags:
- base_model:adapter:unsloth/Qwen2.5-1.5B-Instruct
- lora
- sft
- transformers
- trl
- unsloth
licence: license
pipeline_tag: text-generation
---

# Model Card for qwen25_1.5b_normal_sft_final

This model is a fine-tuned version of [unsloth/Qwen2.5-1.5B-Instruct](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.20.0
- TRL: 0.23.0
- Transformers: 4.57.2
- Pytorch: 2.11.0+cu128
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```