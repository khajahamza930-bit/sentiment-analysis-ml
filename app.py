"""
app.py

Streamlit web app for the Sentiment Analysis project.
The user types some text and the trained model (see train_final_model.py)
predicts whether it is Positive, Neutral or Negative.

Run with:  streamlit run app.py
"""

import pickle
import time
from pathlib import Path

import streamlit as st
from scipy.sparse import hstack

# Must match the value used in train_final_model.py
CHAR_WEIGHT = 1.2

BASE_DIR = Path(__file__).parent

# Icon, title and message shown for each prediction
RESULT_INFO = {
    "positive": ("😊", "POSITIVE", "The text expresses a positive sentiment."),
    "negative": ("😞", "NEGATIVE", "The text expresses a negative sentiment."),
    "neutral": ("😐", "NEUTRAL", "The text expresses a neutral sentiment."),
}

EXAMPLES = [
    "😊 “I absolutely love this product!”",
    "😞 “This product is terrible and disappointing.”",
    "😐 “The meeting is scheduled for tomorrow.”",
    "😊 “The service was excellent and very helpful.”",
    "😞 “I am unhappy with the quality.”",
]

TECHNOLOGIES = [
    "🐍 Python",
    "🤖 Machine Learning",
    "📝 TF-IDF",
    "📊 Logistic Regression",
    "🌐 Streamlit",
]

DEVELOPERS = [
    ("Khaja Ameen Uddin", "161024748302"),
    ("Mohammed Wali Uddin", "161024748305"),
]


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Read style.css and apply it to the page."""
    css = (BASE_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the saved model and both vectorizers (cached, so it only runs once)."""
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "word_vectorizer.pkl", "rb") as f:
        word_vectorizer = pickle.load(f)
    with open(BASE_DIR / "char_vectorizer.pkl", "rb") as f:
        char_vectorizer = pickle.load(f)
    return model, word_vectorizer, char_vectorizer


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_sentiment(text, model, word_vectorizer, char_vectorizer):
    """Return the predicted label and a {label: percentage} dictionary."""
    word_features = word_vectorizer.transform([text])
    char_features = char_vectorizer.transform([text])

    # Same feature combination as in training
    features = hstack([word_features, char_features * CHAR_WEIGHT])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    scores = {
        label: probability * 100
        for label, probability in zip(model.classes_, probabilities)
    }
    return prediction, scores


# ---------------------------------------------------------
# UI pieces
# ---------------------------------------------------------

def show_sidebar():
    tech_chips = "".join(f'<span class="tech-chip">{tech}</span>' for tech in TECHNOLOGIES)

    developers_html = '<div class="developer-divider"></div>'.join(
        f"""
        <div>
            <div class="developer-name">{name}</div>
            <div class="developer-roll">Roll No: {roll_no}</div>
        </div>
        """
        for name, roll_no in DEVELOPERS
    )

    with st.sidebar:
        st.html(f"""
            <div class="sidebar-brand">
                <div class="sidebar-logo">💬</div>
                <div>
                    <div class="sidebar-title">Sentiment AI</div>
                    <div class="sidebar-subtitle">AI • NLP • MACHINE LEARNING</div>
                </div>
            </div>

            <div class="side-section">
                <div class="side-heading">📌 About</div>
                <div class="side-description">
                    AI-powered sentiment analysis that uses Natural Language
                    Processing and Machine Learning to understand the emotion
                    behind text.
                </div>
            </div>

            <div class="side-section">
                <div class="side-heading">🛠️ Technology</div>
                <div class="tech-grid">{tech_chips}</div>
            </div>

            <div class="side-section">
                <div class="side-heading">🧠 Model</div>
                <div class="model-grid">
                    <div class="model-stat">
                        <div class="model-value">62.81%</div>
                        <div class="model-label">Accuracy</div>
                    </div>
                    <div class="model-stat">
                        <div class="model-value">3</div>
                        <div class="model-label">Classes</div>
                    </div>
                    <div class="model-stat wide">
                        <div class="model-value">45,615</div>
                        <div class="model-label">Training Samples</div>
                    </div>
                </div>
            </div>

            <div class="developer-card">
                <div class="developer-heading">👨‍💻 Developed By</div>
                {developers_html}
            </div>
        """)


def show_hero():
    st.html("""
        <div class="hero">
            <div class="hero-badge">🤖 AI POWERED SENTIMENT ANALYSIS</div>
            <h1>Understand the Emotion Behind Text</h1>
            <p>
                Analyze text and discover whether the sentiment
                is Positive, Neutral, or Negative.
            </p>
        </div>
    """)


def show_examples():
    chips = "".join(f'<span class="example-chip">{example}</span>' for example in EXAMPLES)
    st.html(f"""
        <div class="example-panel">
            <div class="example-label">⚡ Quick examples</div>
            {chips}
        </div>
    """)


def show_result(prediction, confidence):
    # Fall back to neutral if the model returns an unexpected label
    icon, title, description = RESULT_INFO.get(prediction, RESULT_INFO["neutral"])

    st.html(f"""
        <div class="result">
            <div class="result-icon">{icon}</div>
            <div class="result-title">{title}</div>
            <div class="confidence">
                {description}<br><br>
                Confidence: <b>{confidence:.2f}%</b>
            </div>
        </div>
    """)


def show_probabilities(scores):
    st.html('<div style="height:18px;"></div><h3>📊 Sentiment Probability</h3>')

    columns = st.columns(3)
    cards = [("positive", "😊 Positive"), ("neutral", "😐 Neutral"), ("negative", "😞 Negative")]

    for column, (label, display_name) in zip(columns, cards):
        with column:
            st.html(f"""
                <div class="metric-card">
                    <div class="metric-value">{scores.get(label, 0):.2f}%</div>
                    <div class="metric-label">{display_name}</div>
                </div>
            """)


def show_footer():
    st.html("""
        <div class="footer">
            Sentiment Analysis Using Machine Learning
            <div style="margin-top: 10px;">CSE / AIML • NSAKCET</div>
            <div style="margin-top: 10px; font-size: 12px;">
                Developed by Khaja Ameen Uddin
            </div>
        </div>
    """)


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

def main():
    load_css()
    model, word_vectorizer, char_vectorizer = load_model()

    show_sidebar()
    show_hero()

    st.markdown("### ✍️ Enter your text")
    show_examples()

    text = st.text_area(
        "Enter text for sentiment analysis",
        placeholder="Example: I absolutely love this product. It is amazing!",
        height=160,
        label_visibility="collapsed",
    )

    if not st.button("🔍 Analyze Sentiment", use_container_width=True):
        show_footer()
        return

    if text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
        show_footer()
        return

    with st.spinner("🤖 AI is analyzing your text..."):
        time.sleep(0.8)  # short pause so the spinner is visible
        prediction, scores = predict_sentiment(text, model, word_vectorizer, char_vectorizer)

    confidence = max(scores.values())
    show_result(prediction, confidence)
    show_probabilities(scores)
    show_footer()


main()