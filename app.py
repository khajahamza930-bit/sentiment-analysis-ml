import importlib


try:
    st = importlib.import_module("streamlit")
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Streamlit is required. Install it with: python -m pip install streamlit"
    ) from error
import pickle


# Load trained model and TF-IDF vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# Title
st.title("💬 Sentiment Analysis")
st.subheader("Using Machine Learning")


# Description
st.write(
    "Enter a sentence, review, or comment below "
    "to predict its sentiment."
)


# Text input
text = st.text_area(
    "Enter your text:",
    placeholder="Example: The product is excellent and easy to use."
)


# Prediction button
if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Convert text into TF-IDF features
        text_tfidf = vectorizer.transform([text])

        # Predict sentiment
        prediction = model.predict(text_tfidf)[0]

        # Get probabilities
        probabilities = model.predict_proba(text_tfidf)[0]

        confidence = max(probabilities) * 100

        # Display result
        st.markdown("### Prediction")

        if prediction == "positive":
            st.success("😊 POSITIVE")

        elif prediction == "negative":
            st.error("😞 NEGATIVE")

        else:
            st.info("😐 NEUTRAL")

        st.write(f"**Confidence:** {confidence:.2f}%")