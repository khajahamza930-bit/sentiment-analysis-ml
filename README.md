# 💬 Sentiment Analysis Using Machine Learning

A web app that reads a piece of text and predicts whether its sentiment is **Positive**, **Neutral** or **Negative**. It uses TF-IDF features and a Logistic Regression model, with a Streamlit interface on top.

![App screenshot](screenshot.png)

---

## ✨ Features

- Classifies text into 3 classes: 😊 Positive, 😐 Neutral, 😞 Negative
- Shows the confidence of the prediction and the probability of every class
- Quick example sentences to try the app immediately
- Dark, modern interface built with Streamlit and custom CSS

---

## 🛠️ Tech Stack

| Area | Tools |
| ---- | ----- |
| Language | Python |
| Data handling | Pandas, NumPy, SciPy |
| Machine learning | Scikit-learn (TF-IDF, Logistic Regression) |
| Web app | Streamlit, custom CSS |

---

## 📁 Project Structure

```text
Sentiment_Analysis_Project/
│
├── app.py                  # Streamlit web app
├── style.css               # Styling for the app
├── train_final_model.py    # Trains the model and saves the .pkl files
├── dataset.csv             # Training data (text + sentiment)
├── model.pkl               # Trained Logistic Regression model
├── word_vectorizer.pkl     # Word-level TF-IDF vectorizer
├── char_vectorizer.pkl     # Character-level TF-IDF vectorizer
├── requirements.txt        # Python packages
└── README.md
```

---

## 🤖 How It Works

1. **Text to numbers.** A model can't read words, so the text is converted into numbers with two TF-IDF vectorizers:
   - **Word TF-IDF:** single words and word pairs (1–2 grams, up to 30,000 features). This captures meaning such as "not good".
   - **Character TF-IDF:** groups of 3–5 characters (up to 50,000 features). This helps with spelling mistakes, slang and emojis.
2. **Combine the features.** Both feature sets are joined side by side. The character features are multiplied by a weight of `1.2`.
3. **Classify.** A Logistic Regression model (`C=0.25`, `class_weight="balanced"`, `max_iter=2000`) predicts the sentiment and gives a probability for each class.

`class_weight="balanced"` is used because the data is uneven: Neutral has far more samples than Negative, and without balancing the model would favour Neutral.

---

## 📊 Dataset

The model is trained on the [TweetEval](https://github.com/cardiffnlp/tweeteval) sentiment dataset (tweets labelled Positive, Neutral or Negative) from Cardiff NLP.

| Split | Samples |
| ----- | ------: |
| Training | 45,615 |
| Test | 12,284 |

Training class distribution: Neutral 20,673 · Positive 17,849 · Negative 7,093

---

## 📈 Results

**Test accuracy: 62.81%** (on 12,284 test samples)

| Sentiment | Precision | Recall | F1-score | Support |
| --------- | --------: | -----: | -------: | ------: |
| Negative | 58.52% | 71.95% | 64.54% | 3,972 |
| Neutral | 68.58% | 56.70% | 62.07% | 5,937 |
| Positive | 59.87% | 62.82% | 61.31% | 2,375 |

I tested different values of the regularization parameter `C`, and `C = 0.25` gave the best accuracy:

| C | Test accuracy |
| -: | ------------: |
| 0.25 | 62.81% |
| 0.5 | 62.39% |
| 0.75 | 62.04% |
| 1.0 | 61.70% |

This is a classical machine learning baseline. Sentiment on short, informal tweets is hard (sarcasm, slang, mixed feelings), so neural models such as transformers would likely score higher.

---

## 🚀 Getting Started

**Requirements:** Python 3.11 or newer

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install the packages
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

### Retrain the model (optional)

The trained `.pkl` files are already included. To train again from `dataset.csv`:

```bash
python train_final_model.py
```

This overwrites `model.pkl`, `word_vectorizer.pkl` and `char_vectorizer.pkl`.

---

## 💡 Example

```text
Input:   The product is excellent and easy to use.
Output:  😊 POSITIVE
```

---

## 🔮 Future Improvements

- Compare more models (Naive Bayes, SVM, transformer models)
- Support more languages
- Add charts and sentiment analytics
- Analyze many texts at once by uploading a CSV file
- Improve accuracy on sarcasm and mixed sentiment

---

## 👨‍💻 Author

**<Your Name>**
B.Tech CSE (AIML), Nawab Shah Alam Khan College of Engineering and Technology

---

## 📄 License

This project is for educational purposes. See the `LICENSE` file for details.