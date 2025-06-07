"""For loading models and tokenizers an `internet` connection is typically required.  
The library `fetches` the necessary model files from the Hugging Face model hub.  

Once you've loaded the model and tokenizer at least once and `cached` them locally,  
subsequent uses of the same model or tokenizer can often work `offline`.  
"""

from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer 
model_name = 'BlackKakapo/opus-mt-ro-en'    # Look Here

# Load the tokenizer associated with the pre-trained translation model
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load the pre-trained MarianMT model for translation
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