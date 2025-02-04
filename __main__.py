import torch
from transformers import DistilBertTokenizer
from fake_news_detection.ipynb import FakeNewsClassifier

# Load Model
model = FakeNewsClassifier()
model.load_state_dict(torch.load("model_weights/distilbert-fake-news.pth.pth"))
model.eval()  # Set to evaluation mode

# Load Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Sample Text
text = "Breaking News: Aliens Land on Earth!"

# Tokenize Input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Make Prediction
with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    prediction = torch.argmax(logits, dim=1).item()

print("Fake News" if prediction == 0 else "Real News")
