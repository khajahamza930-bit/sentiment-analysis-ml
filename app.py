import streamlit as st
import pickle
from scipy.sparse import hstack


# ==============================
# LOAD MODEL
# ==============================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("word_vectorizer.pkl", "rb") as file:
    word_vectorizer = pickle.load(file)

with open("char_vectorizer.pkl", "rb") as file:
    char_vectorizer = pickle.load(file)


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.title("💬 Sentiment Analysis")

    st.markdown("---")

    st.write("### About the Project")

    st.write(
        "This application uses Natural Language Processing "
        "and Machine Learning to classify text into "
        "Positive, Negative, or Neutral sentiment."
    )

    st.markdown("---")

    st.write("### Technologies")

    st.write("🐍 Python")
    st.write("🧠 Machine Learning")
    st.write("📝 TF-IDF")
    st.write("📊 Logistic Regression")
    st.write("🌐 Streamlit")

    st.markdown("---")

    st.write("### Model")

    st.write("Logistic Regression")
    st.write("Accuracy: 62.47%")


# ==============================
# MAIN TITLE
# ==============================

st.title("💬 Sentiment Analysis")

st.subheader("Using Machine Learning")

st.write(
    "Analyze the sentiment of a sentence, review, "
    "or comment using our machine learning model."
)

st.markdown("---")


# ==============================
# TEXT INPUT
# ==============================

st.write("### 📝 Enter Your Text")

text = st.text_area(
    "Type your sentence below:",
    placeholder="Example: The product is excellent and easy to use.",
    height=150
)


# ==============================
# ANALYZE BUTTON
# ==============================

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text.")

    else:

        # Word TF-IDF
        word_features = word_vectorizer.transform([text])

        # Character TF-IDF
        char_features = char_vectorizer.transform([text])

        # Combine features
        combined_features = hstack([
            word_features,
            char_features * 1.2
        ])

        # Prediction
        prediction = model.predict(combined_features)[0]

        # Probability
        probabilities = model.predict_proba(combined_features)[0]
        probability = max(probabilities) * 100


        # ==============================
        # RESULT
        # ==============================

        st.markdown("---")

        st.write("### 🎯 Analysis Result")

        if prediction == "positive":

            st.success("😊 POSITIVE")

            st.write(
                "The text expresses a positive sentiment."
            )

        elif prediction == "negative":

            st.error("😞 NEGATIVE")

            st.write(
                "The text expresses a negative sentiment."
            )

        else:

            st.info("😐 NEUTRAL")

            st.write(
                "The text expresses a neutral sentiment."
            )


        st.metric(
            "Prediction Probability",
            f"{probability:.2f}%"
        )

# ==============================
# PROJECT STATISTICS
# ==============================

st.markdown("---")

st.write("### 📊 Project Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Samples", "45,615")

with col2:
    st.metric("Test Samples", "12,284")

with col3:
    st.metric("Model Accuracy", "62.47%")


st.write("### 🤖 Machine Learning Model")

st.write(
    "**Logistic Regression** with Word and Character TF-IDF features."
)

st.write("### 🏷️ Sentiment Classes")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("😊 Positive")

with col2:
    st.info("😐 Neutral")

with col3:
    st.error("😞 Negative")

# ==============================
# EXAMPLES
# ==============================

st.markdown("---")

st.write("### 💡 Try These Examples")

st.write("😊 **Positive:**")
st.code("I absolutely love this product. It is amazing!")

st.write("😞 **Negative:**")
st.code("This product is terrible. I hate it.")

st.write("😐 **Neutral:**")
st.code("The meeting is scheduled for tomorrow.")


# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.caption(
    "Sentiment Analysis Using Machine Learning | "
    "Mini Project"
)