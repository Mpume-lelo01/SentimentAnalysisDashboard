from transformers import pipeline

# Load Hugging Face sentiment model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment_batch(texts):
    results = []
    for text in texts:
        # Get both POSITIVE and NEGATIVE predictions
        predictions = classifier(text, top_k=2)
        sentiment_scores = {pred["label"]: pred["score"] for pred in predictions}

        positive_score = sentiment_scores.get("POSITIVE", 0)
        negative_score = sentiment_scores.get("NEGATIVE", 0)

        # Decide final sentiment and confidence
        if positive_score > negative_score:
            sentiment = "Positive"
            confidence = positive_score
        elif negative_score > positive_score:
            sentiment = "Negative"
            confidence = negative_score
        else:
            sentiment = "Neutral"
            confidence = positive_score  # or negative_score—they’re equal

        results.append({
            "text": text,
            "sentiment": sentiment,
            "positive_score": round(positive_score, 4),
            "negative_score": round(negative_score, 4),
            "confidence": round(confidence, 4)
        })

    return results
