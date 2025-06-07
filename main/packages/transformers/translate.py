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
    pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
    pip install sacremoses
"""

from transformers import MarianMTModel, MarianTokenizer

# Specify the model name for English to Romanian translation
model_name = 'Helsinki-NLP/opus-mt-en-ro'

# Load the tokenizer associated with the pre-trained translation model
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load the pre-trained MarianMT model for translation
model = MarianMTModel.from_pretrained(model_name)

# Define the input English sentence to be translated
text = "Everything should be made as simple as possible, but no simpler."

# Tokenize the input text and convert it to tensor format suitable for the model
inputs = tokenizer(text, return_tensors="pt")

# Generate the translated text using the model
translated = model.generate(**inputs)

# Decode the generated tensor back into human-readable text (Romanian)
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# Print the original English text and its Romanian translation
print(text)
print(translated_text)

"""
    Everything should be made as simple as possible, but no simpler.
    Totul ar trebui să fie cât mai simplu posibil, dar nu mai simplu.
"""