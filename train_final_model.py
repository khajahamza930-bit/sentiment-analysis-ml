import pandas as pd
import pickle

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. LOAD TRAINING DATA
# ==============================

train_data = pd.read_csv("dataset.csv")

X_train = train_data["text"]
y_train = train_data["sentiment"]

print("Training dataset loaded successfully!")
print("Training samples:", len(train_data))


# ==============================
# 2. WORD TF-IDF
# ==============================

print("\nCreating Word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)

X_train_word = word_vectorizer.fit_transform(X_train)

print("Word TF-IDF completed!")
print("Word features:", len(word_vectorizer.get_feature_names_out()))


# ==============================
# 3. CHARACTER TF-IDF
# ==============================

print("\nCreating Character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000
)

X_train_char = char_vectorizer.fit_transform(X_train)

print("Character TF-IDF completed!")
print("Character features:", len(char_vectorizer.get_feature_names_out()))


# ==============================
# 4. COMBINE FEATURES
# ==============================

print("\nCombining Word and Character features...")

X_train_combined = hstack([
    X_train_word,
    X_train_char * 1.2
])

print("Feature combination completed!")


# ==============================
# 5. TRAIN LOGISTIC REGRESSION
# ==============================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    C=0.5,
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_combined, y_train)

print("Model training completed!")


# ==============================
# 6. SAVE MODEL
# ==============================

print("\nSaving final model files...")

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("word_vectorizer.pkl", "wb") as file:
    pickle.dump(word_vectorizer, file)

with open("char_vectorizer.pkl", "wb") as file:
    pickle.dump(char_vectorizer, file)


# ==============================
# 7. FINISHED
# ==============================

print("\n==========================================")
print("FINAL MODEL CREATED SUCCESSFULLY!")
print("==========================================")

print("\nFiles saved:")
print("1. model.pkl")
print("2. word_vectorizer.pkl")
print("3. char_vectorizer.pkl")

print("\nFinal configuration:")
print("Word TF-IDF: 1-2 grams")
print("Word features: 30,000")
print("Character TF-IDF: 3-5 grams")
print("Character features: 50,000")
print("Character weight: 1.2")
print("Logistic Regression C: 0.5")
print("Class weight: balanced")

print("\nTraining completed successfully! 🎉")