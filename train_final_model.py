"""
Sentiment Analysis - Model Training

Combines word-level and character-level TF-IDF features
and trains a Logistic Regression classifier.
"""

import pickle

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# Model settings
DATASET_FILE = "dataset.csv"

CHAR_WEIGHT = 1.2
C_VALUE = 0.5
TEST_SIZE = 0.20
RANDOM_STATE = 42

WORD_MAX_FEATURES = 30000
CHAR_MAX_FEATURES = 50000


def load_data(path):
    """Load text and sentiment labels from the dataset."""

    data = pd.read_csv(path)
    data = data.dropna(subset=["text", "sentiment"])

    texts = data["text"].astype(str)
    labels = data["sentiment"].astype(str).str.lower()

    print(f"Total records: {len(data)}")
    print("\nSentiment distribution:")
    print(labels.value_counts())

    return texts, labels


def build_word_vectorizer():
    """Create the word-level TF-IDF vectorizer."""

    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=WORD_MAX_FEATURES,
    )


def build_char_vectorizer():
    """Create the character-level TF-IDF vectorizer."""

    return TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=CHAR_MAX_FEATURES,
    )


def build_features(word_vectorizer, char_vectorizer, texts, fit=False):
    """
    Convert text into combined word and character TF-IDF features.

    Fit the vectorizers when training. Otherwise, use their
    existing vocabulary to transform new text.
    """

    if fit:
        word_features = word_vectorizer.fit_transform(texts)
        char_features = char_vectorizer.fit_transform(texts)
    else:
        word_features = word_vectorizer.transform(texts)
        char_features = char_vectorizer.transform(texts)

    return hstack([
        word_features,
        char_features * CHAR_WEIGHT,
    ])


def create_model():
    """Create the Logistic Regression classifier."""

    return LogisticRegression(
        C=C_VALUE,
        max_iter=2000,
        class_weight="balanced",
    )


def evaluate_model(model, test_features, y_test, class_labels):
    """Evaluate the model and save its performance report."""

    predictions = model.predict(test_features)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    )

    print("\n========== MODEL EVALUATION ==========")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(confusion_df)

    report_text = (
        "SENTIMENT ANALYSIS MODEL EVALUATION\n"
        "===================================\n\n"
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
        "Classification Report:\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{confusion_df}\n"
    )

    with open("model_evaluation.txt", "w", encoding="utf-8") as file:
        file.write(report_text)

    return accuracy


def save_file(obj, filename):
    """Save a trained model or vectorizer as a pickle file."""

    with open(filename, "wb") as file:
        pickle.dump(obj, file)

    print(f"Saved: {filename}")


def main():
    """Run model evaluation and train the final deployment model."""

    # Load the dataset
    texts, labels = load_data(DATASET_FILE)
    class_labels = sorted(labels.unique())

    # Keep a separate test set for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Train the evaluation model using only training data
    print("\nBuilding training features...")

    word_vectorizer = build_word_vectorizer()
    char_vectorizer = build_char_vectorizer()

    train_features = build_features(
        word_vectorizer,
        char_vectorizer,
        X_train,
        fit=True,
    )

    print("Training feature shape:", train_features.shape)

    print("\nTraining evaluation model...")

    evaluation_model = create_model()
    evaluation_model.fit(train_features, y_train)

    # Transform test data using the fitted vectorizers
    print("\nPreparing test features...")

    test_features = build_features(
        word_vectorizer,
        char_vectorizer,
        X_test,
    )

    evaluate_model(
        evaluation_model,
        test_features,
        y_test,
        class_labels,
    )

    # Train the final model using the complete dataset
    print("\n========== FINAL MODEL TRAINING ==========")
    print("Building features using all dataset records...")

    final_word_vectorizer = build_word_vectorizer()
    final_char_vectorizer = build_char_vectorizer()

    final_features = build_features(
        final_word_vectorizer,
        final_char_vectorizer,
        texts,
        fit=True,
    )

    print("Final feature shape:", final_features.shape)

    print("\nTraining final model on all records...")

    final_model = create_model()
    final_model.fit(final_features, labels)

    # Save the final model and vectorizers
    save_file(final_model, "model.pkl")
    save_file(final_word_vectorizer, "word_vectorizer.pkl")
    save_file(final_char_vectorizer, "char_vectorizer.pkl")

    print("\n========== TRAINING COMPLETED ==========")
    print("Evaluation report: model_evaluation.txt")
    print("Final model: model.pkl")
    print("Word vectorizer: word_vectorizer.pkl")
    print("Character vectorizer: char_vectorizer.pkl")


if __name__ == "__main__":
    main()
