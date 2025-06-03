"""
    pip install transformers
    pip install sentencepiece
    pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
    pip install sacremoses
"""

from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer 
model_name = 'BlackKakapo/opus-mt-ro-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Example sentence in Romanian
sentence = "Aceasta este o propoziție de test pentru traducere."

# Tokenize the input text
inputs = tokenizer(sentence, return_tensors="pt")

# Perform translation
translated = model.generate(**inputs)

# Decode translated text
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# Output result
print(sentence)
print(translated_text)

"""
    Aceasta este o propoziție de test pentru traducere.
    This is a test sentence for translation.
"""