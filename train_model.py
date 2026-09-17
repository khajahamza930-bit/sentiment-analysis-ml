import csv
import pickle
from importlib import import_module

train_test_split = import_module("sklearn.model_selection").train_test_split
TfidfVectorizer = import_module("sklearn.feature_extraction.text").TfidfVectorizer
LogisticRegression = import_module("sklearn.linear_model").LogisticRegression
metrics = import_module("sklearn.metrics")
accuracy_score = metrics.accuracy_score
classification_report = metrics.classification_report


# 1. Load the dataset
with open("dataset.csv", "r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

if not rows or "text" not in rows[0] or "sentiment" not in rows[0]:
    raise ValueError("dataset.csv must contain 'text' and 'sentiment' columns")

data = rows

print("Dataset loaded successfully!")
print(data[:5])


# 2. Separate text and sentiment
X = [row["text"] for row in data]
y = [row["sentiment"] for row in data]


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5. Create the Machine Learning model
model = LogisticRegression(max_iter=1000)


# 6. Train the model
model.fit(X_train_tfidf, y_train)


# 7. Make predictions
y_pred = model.predict(X_test_tfidf)


# 8. Check model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 9. Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# 10. Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print("Training completed successfully!")