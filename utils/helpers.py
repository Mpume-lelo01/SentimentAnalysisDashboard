from sklearn.feature_extraction.text import CountVectorizer

def extract_keywords(text, n=5):
    vectorizer = CountVectorizer(stop_words='english', max_features=n)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out().tolist()
