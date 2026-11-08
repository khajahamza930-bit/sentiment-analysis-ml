
import pickle
from pathlib import Path

import pytest
from scipy.sparse import hstack


BASE_DIR = Path(__file__).resolve().parents[1]

VALID_LABELS = {"positive", "negative", "neutral"}


@pytest.fixture(scope="session")
def model_files():
    """Load the trained model and both vectorizers."""

    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "word_vectorizer.pkl", "rb") as f:
        word_vectorizer = pickle.load(f)

    with open(BASE_DIR / "char_vectorizer.pkl", "rb") as f:
        char_vectorizer = pickle.load(f)

    return model, word_vectorizer, char_vectorizer


@pytest.mark.parametrize(
    "text",
    [
        "I absolutely love this product! It is amazing.",
        "This product is terrible and disappointing.",
        "The meeting is scheduled for tomorrow.",
    ],
)
def test_sentiment_prediction(text, model_files):
    """Test prediction and confidence scores for sample text."""

    model, word_vectorizer, char_vectorizer = model_files

    word_features = word_vectorizer.transform([text])
    char_features = char_vectorizer.transform([text])

    features = hstack([word_features, char_features * 1.2])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    scores = {
        str(label): float(probability) * 100
        for label, probability in zip(model.classes_, probabilities)
    }

    # Check that the prediction is a valid sentiment.
    assert str(prediction).lower() in VALID_LABELS

    # Check that all three sentiment classes are available.
    assert set(scores.keys()) == VALID_LABELS

    # Check that confidence scores are valid.
    assert all(0 <= score <= 100 for score in scores.values())

    # Probability percentages should total approximately 100%.
    assert sum(scores.values()) == pytest.approx(100, abs=0.01)
