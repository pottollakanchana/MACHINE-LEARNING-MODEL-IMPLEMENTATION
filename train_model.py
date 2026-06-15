import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple training data (demo)
texts = [
    "free money now",
    "win lottery",
    "hello friend",
    "how are you",
    "claim reward",
    "good morning"
]

labels = [1, 1, 0, 0, 1, 0]  # 1=spam, 0=not spam

# Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Model
model = MultinomialNB()
model.fit(X, labels)

# SAVE CORRECTLY
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model created successfully!")