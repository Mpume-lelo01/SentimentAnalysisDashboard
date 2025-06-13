import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import io
import base64

sns.set(style="whitegrid")

def plot_sentiment_distribution(results):
    """
    results: list of dicts with 'sentiment' key
    Returns: base64 PNG image string for Streamlit display
    """
    sentiments = [r['sentiment'] for r in results]
    counts = Counter(sentiments)
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=values, palette="pastel")
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
