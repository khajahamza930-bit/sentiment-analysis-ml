"""Prediction utilities for the Sentiment Analysis application."""

import pickle
from pathlib import Path

from scipy.sparse import hstack


CHAR_WEIGHT = 1.2
PROJECT_DIR = Path(__file__).resolve().parent


def load_model_files():
    """Load the trained model and both TF-IDF vectorizers."""
    with open(PROJECT_DIR / "model.pkl", "rb") as file:
        model = pickle.load(file)

    with open(PROJECT_DIR / "word_vectorizer.pkl", "rb") as file:
        word_vectorizer = pickle.load(file)

    with open(PROJECT_DIR / "char_vectorizer.pkl", "rb") as file:
        char_vectorizer = pickle.load(file)

    return model, word_vectorizer, char_vectorizer


def predict_sentiment(text, model, word_vectorizer, char_vectorizer):
    """Predict sentiment and return class probabilities and confidence."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must not be empty.")

    word_features = word_vectorizer.transform([text])
    char_features = char_vectorizer.transform([text])

    combined_features = hstack(
        [word_features, char_features * CHAR_WEIGHT]
    )

    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]

    probability_dict = {
        label: float(probability) * 100
        for label, probability in zip(model.classes_, probabilities)
    }

    confidence = max(probabilities) * 100

    return prediction, probability_dict, confidence
