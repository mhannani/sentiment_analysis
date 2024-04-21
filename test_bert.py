# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT-arabizi")
model = AutoModelForSequenceClassification.from_pretrained("SI2M-Lab/DarijaBERT-arabizi")

# Example input text with a masked token
input_text = "عبقرية المدرب وروعة اللاعبين"

# Tokenize input text
encoded_input = tokenizer(input_text, return_tensors='pt')

# Perform inference
with torch.no_grad():
    output = model(**encoded_input)

# Get predicted probabilities for each class
probabilities = torch.softmax(output.logits, dim=1).squeeze()

# Assuming your model is fine-tuned for binary classification (e.g., positive vs. negative sentiment)
positive_prob = probabilities[1].item()  # Probability of positive sentiment
negative_prob = probabilities[0].item()  # Probability of negative sentiment

print("Positive probability:", positive_prob)
print("Negative probability:", negative_prob)
