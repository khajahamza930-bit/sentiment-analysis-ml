# 💬 Sentiment Analysis Using Machine Learning

A machine learning-based Sentiment Analysis project that classifies text into **Positive, Negative, or Neutral** sentiment.

## 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) technique used to identify the emotional tone of text.

This project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text feature extraction and **Logistic Regression** for sentiment classification.

A simple **Streamlit web application** allows users to enter a sentence, review, or comment and receive a predicted sentiment with a confidence score.

## 🎯 Objectives

- To preprocess and analyze text data.
- To convert text into numerical features using TF-IDF.
- To train a supervised machine learning model.
- To classify text into Positive, Negative, or Neutral sentiment.
- To build an interactive Streamlit web application.
- To display the predicted sentiment and confidence score.

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **NLTK**
- **Streamlit**
- **TF-IDF**
- **Logistic Regression**

## 📂 Project Structure

```text
sentiment-analysis-ml/
│
├── app.py
├── dataset.csv
├── model.pkl
├── vectorizer.pkl
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md