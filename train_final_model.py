"""
train_final_model.py

Trains a sentiment classifier (Positive / Negative / Neutral) on dataset.csv.
Features: word TF-IDF + character TF-IDF, fed into Logistic Regression.
Saves the model and both vectorizers as .pkl files for app.py to use.
"""

import pickle

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Settings kept at the top so they are easy to change while experimenting
DATASET_FILE = "dataset.csv"
CHAR_WEIGHT = 1.2   # app.py must use the same value when predicting
C_VALUE = 0.5


def load_data(path):
    data = pd.read_csv(path)
    print(f"Loaded {len(data)} training samples")
    return data["text"], data["sentiment"]


def build_word_features(texts):
    # Single words and word pairs ("not good") capture meaning and negation
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )
    features = vectorizer.fit_transform(texts)
    print(f"Word features: {features.shape[1]}")
    return vectorizer, features


def build_char_features(texts):
    # Character n-grams help with spelling mistakes, slang and emojis
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
    )
    features = vectorizer.fit_transform(texts)
    print(f"Character features: {features.shape[1]}")
    return vectorizer, features


def train_model(features, labels):
    # class_weight="balanced" stops the model from favouring Neutral,
    # which has far more samples than Negative
    model = LogisticRegression(C=C_VALUE, max_iter=2000, class_weight="balanced")
    model.fit(features, labels)
    return model


def save_file(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def main():
    texts, labels = load_data(DATASET_FILE)

    word_vectorizer, word_features = build_word_features(texts)
    char_vectorizer, char_features = build_char_features(texts)

    # Join both feature sets side by side into one matrix
    combined = hstack([word_features, char_features * CHAR_WEIGHT])

    print("Training model...")
    model = train_model(combined, labels)

    save_file(model, "model.pkl")
    save_file(word_vectorizer, "word_vectorizer.pkl")
    save_file(char_vectorizer, "char_vectorizer.pkl")
    print("Done! Saved model.pkl, word_vectorizer.pkl, char_vectorizer.pkl")


if __name__ == "__main__":
    main()