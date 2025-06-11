""" Deep learning models can `translate` text from one language to another.  

One popular library for machine translation is transformers by `Hugging` Face.  
It provides easy-to-use interfaces to various `pre-trained` models.

Install required libraries:
    - transformers: for accessing pre-trained translation models
    - sentencepiece: tokenizer dependency for some models
    - torch and related packages: required by transformers for deep learning inference
    - sacremoses: used for tokenization in certain models

    pip install transformers
    pip install sentencepiece
    pip install torch torchvision torchaudio -f \
        https://download.pytorch.org/whl/torch_stable.html
    pip install sacremoses
"""

from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer 
model_name = 'Helsinki-NLP/opus-mt-en-ro'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)

# Text to translate
text = "Everything should be made as simple as possible, but no simpler."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Perform translation
translated = model.generate(**inputs)

# Decode translated text
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# Output result
print(text)
print(translated_text)

"""
    Everything should be made as simple as possible, but no simpler.
    Totul ar trebui să fie cât mai simplu posibil, dar nu mai simplu.
"""