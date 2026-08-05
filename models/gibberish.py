from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")
tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")

# Classify text to prevent the customer from typing random text.
def detect_gibberish(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label_id = probabilities.argmax().item()
    
    return model.config.id2label[predicted_label_id]

 
