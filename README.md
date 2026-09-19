# 💬 Sentiment Analysis Using Machine Learning

## 📌 Project Overview

This project is a **Sentiment Analysis application** that uses Natural Language Processing (NLP) and Machine Learning to classify text into three sentiment categories:

* 😊 Positive
* 😐 Neutral
* 😞 Negative

The application allows users to enter a sentence, review, comment, or feedback and automatically predicts its sentiment using a trained Machine Learning model.

The project is developed using **Python, Scikit-learn, TF-IDF, Logistic Regression, and Streamlit**.

---

## 🎯 Objectives

* Analyze the sentiment of text automatically.
* Classify text into Positive, Negative, or Neutral.
* Apply Natural Language Processing techniques.
* Use TF-IDF for text feature extraction.
* Train a Machine Learning classification model.
* Build an interactive web interface using Streamlit.
* Demonstrate a practical application of Machine Learning and NLP.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK / Text Processing
* TF-IDF
* Logistic Regression
* Streamlit
* SciPy

---

## 📁 Project Structure

```text
Sentiment_Analysis_Project/
│
├── app.py
├── dataset.csv
├── model.pkl
├── word_vectorizer.pkl
├── char_vectorizer.pkl
├── train_final_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🤖 Machine Learning Approach

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.

Two types of TF-IDF features are used:

### 1. Word TF-IDF

* N-gram range: `(1, 2)`
* Minimum document frequency: `2`
* Maximum features: `30,000`

### 2. Character TF-IDF

* Character analyzer
* N-gram range: `(3, 5)`
* Minimum document frequency: `2`
* Maximum features: `50,000`

The Word and Character features are combined before being passed to the Machine Learning model.

### Machine Learning Model

**Logistic Regression**

* C value: `0.25`
* Class weight: `balanced`
* Maximum iterations: `2000`

---

## 📊 Dataset

The project uses the **TweetEval Sentiment dataset**.

### Training Dataset

* Total training samples: **45,615**

### Test Dataset

* Total test samples: **12,284**

### Sentiment Classes

| Label | Sentiment |
| ----- | --------- |
| 0     | Negative  |
| 1     | Neutral   |
| 2     | Positive  |

---

## 📈 Model Performance

The final verified model achieved:

### **Test Accuracy: 62.81%**

Classification results:

| Sentiment | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Negative  |    58.52% | 71.95% |   64.54% |
| Neutral   |    68.58% | 56.70% |   62.07% |
| Positive  |    59.87% | 62.82% |   61.31% |

The model was evaluated on **12,284 test samples**.

---

## 🌐 Streamlit Application

The project includes an interactive **Streamlit web application**.

Users can:

1. Enter text.
2. Click **Analyze Sentiment**.
3. Receive the predicted sentiment.
4. View the prediction probability.

Example:

```text
Input:
The product is excellent and easy to use.

Output:
😊 POSITIVE
```

---

## 📦 Installation

Clone the repository and install the required libraries.

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 🧠 Training the Model

To train the Machine Learning model:

```bash
python train_final_model.py
```

The trained model and TF-IDF vectorizers are saved as `.pkl` files.

---

## 💡 Applications

Sentiment Analysis can be used for:

* Product reviews
* Movie reviews
* Customer feedback
* Social media analysis
* Online surveys
* Customer support
* Business intelligence
* Feedback analysis

---

## 🚀 Future Scope

The project can be improved in the future by:

* Using larger datasets.
* Supporting multiple languages.
* Using advanced NLP techniques.
* Adding real-time social media analysis.
* Comparing multiple Machine Learning models.
* Adding sentiment analytics and visualizations.
* Improving model accuracy.

---

## 👨‍💻 Project Information

**Project:** Sentiment Analysis Using Machine Learning

**Technology:** Python, NLP, Machine Learning, Streamlit

**Model:** Logistic Regression

**Feature Extraction:** Word + Character TF-IDF

**Test Accuracy:** **62.81%**

---

## 📄 License

This project is created for educational and academic purposes.
