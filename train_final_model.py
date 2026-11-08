
"""
train_final_model.py

Trains and evaluates a three-class sentiment classifier.
Features: Word TF-IDF + Character TF-IDF.
Model: Logistic Regression.

Outputs:
- model.pkl
- word_vectorizer.pkl
- char_vectorizer.pkl
- evaluation_report.txt
- confusion_matrix.csv
- class_metrics.csv
"""

import pickle

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Configuration
DATASET_FILE = "dataset.csv"
CHAR_WEIGHT = 1.2
C_VALUE = 0.5
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data(path):
    data = pd.read_csv(path)

    data = data.dropna(subset=["text", "sentiment"])
    data["text"] = data["text"].astype(str)

    print(f"Loaded {len(data)} samples")

    return data["text"], data["sentiment"]


def build_word_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )

    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    print(f"Word features: {train_features.shape[1]}")

    return vectorizer, train_features, test_features


def build_char_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
    )

    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    print(f"Character features: {train_features.shape[1]}")

    return vectorizer, train_features, test_features


def train_model(features, labels):
    model = LogisticRegression(
        C=C_VALUE,
        max_iter=2000,
        class_weight="balanced",
    )

    model.fit(features, labels)

    return model


def save_file(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    labels = sorted(model.classes_)

    report = classification_report(
        test_labels,
        predictions,
        labels=labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=labels,
    )

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    # Save classification report
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("SENTIMENT ANALYSIS - MODEL EVALUATION\n\n")
        f.write(f"Accuracy : {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall   : {recall * 100:.2f}%\n")
        f.write(f"F1-score : {f1 * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save confusion matrix
    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    matrix_df.index.name = "Actual"
    matrix_df.to_csv("confusion_matrix.csv")

    # Save individual class metrics
    class_report = classification_report(
        test_labels,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(class_report).transpose().to_csv(
        "class_metrics.csv"
    )

    print("\nEvaluation files saved successfully!")


def main():
    texts, labels = load_data(DATASET_FILE)

    # Split data before fitting vectorizers to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Build word features
    word_vectorizer, word_train, word_test = build_word_features(
        X_train, X_test
    )

    # Build character features
    char_vectorizer, char_train, char_test = build_char_features(
        X_train, X_test
    )

    # Combine features
    train_features = hstack([
        word_train,
        char_train * CHAR_WEIGHT,
    ])

    test_features = hstack([
        word_test,
        char_test * CHAR_WEIGHT,
    ])

    print("\nTraining Logistic Regression model...")

    model = train_model(train_features, y_train)

    # Evaluate model
    evaluate_model(model, test_features, y_test)

    # Save trained model and vectorizers
    save_file(model, "model.pkl")
    save_file(word_vectorizer, "word_vectorizer.pkl")
    save_file(char_vectorizer, "char_vectorizer.pkl")

    print("\nModel and vectorizers saved successfully!")
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
