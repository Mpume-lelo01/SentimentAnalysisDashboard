from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load model + tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment_batch(texts):
    results = []

    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output[0][0].numpy()
        probs = softmax(scores)

        sentiment = labels[probs.argmax()]
        confidence = round(float(probs.max()), 4)

        results.append({
            "text": text,
            "sentiment": sentiment,
            "positive_score": round(float(probs[2]), 4),
            "negative_score": round(float(probs[0]), 4),
            "confidence": confidence
        })

    return results



