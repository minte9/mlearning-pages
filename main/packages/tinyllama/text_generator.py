"""Example of using the TinyLLaMA model via Hugging Face Transformers in Python. 
This model is a small version of Meta's LLaMA, optimized for running efficiently 
on CPUs or smaller GPUs.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Processing ...")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)

chat = [
    {"role": "user", "content": "What is flask (python)?"}
]

inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")
input_ids = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=30,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

print(chat[0]['content'])
print(response)

"""
    Processing ...
    What is flask (python)?
    <|assistant|>
    Flask is a lightweight web application framework for Python that provides 
    a simple and flexible way to develop and deploy web.
"""