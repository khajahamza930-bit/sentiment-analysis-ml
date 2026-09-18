# 💬 Sentiment Analysis Using Machine Learning

A machine learning-based **Sentiment Analysis** project that classifies text into **Positive, Negative, or Neutral** sentiment.

The project uses **Natural Language Processing (NLP)**, **TF-IDF feature extraction**, and **Logistic Regression**. A Streamlit web application provides an interactive interface where users can enter text and receive a sentiment prediction with its prediction probability.

---

## 📌 Project Overview

Sentiment Analysis is an NLP technique used to identify the emotional tone or sentiment expressed in a piece of text.

This project analyzes a sentence, review, or comment and classifies it into one of three categories:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

The machine learning pipeline uses both **Word TF-IDF** and **Character TF-IDF** features. These features are combined and passed to a **Logistic Regression** classifier.

---

## 🎯 Objectives

- To understand Natural Language Processing and Sentiment Analysis.
- To preprocess and analyze text data.
- To convert text into numerical features using TF-IDF.
- To train a supervised machine learning classification model.
- To classify text into Positive, Negative, or Neutral sentiment.
- To evaluate the trained model using a separate test dataset.
- To develop an interactive web application using Streamlit.

---

## 🧠 Machine Learning Approach

The project follows this pipeline:

```text
User Input
    ↓
Text Processing
    ↓
Word TF-IDF + Character TF-IDF
    ↓
Feature Combination
    ↓
Logistic Regression
    ↓
Sentiment Prediction
    ↓
Positive / Neutral / Negative