from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer 
model_name = 'BlackKakapo/opus-mt-ro-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Example sentences in Romanian
sentences = [
    "Aceasta este o propoziție de test pentru traducere.",
    "Acesta este un alt exemplu de propoziție.",
    "Cât de rapid poate fi acest model?"
]

# Tokenize the input text
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Perform translation
translated = model.generate(**inputs)

# Decode translated texts
translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

# Output results
for i, sentence in enumerate(sentences):
    print(i, sentence, translated_texts[i])

"""
    0 Aceasta este o propoziție de test pentru traducere. This is a test sentence for translation.
    1 Acesta este un alt exemplu de propoziție. This is another example of a sentence.
    2 Cât de rapid poate fi acest model? How fast can this model be?
"""