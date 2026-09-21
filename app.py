import streamlit as st
import pickle
from scipy.sparse import hstack
import time


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — MODERN AI UI + ANIMATIONS
# =========================================================

st.markdown("""
<style>
.project-info{margin:8px auto 24px;max-width:1000px;text-align:center;padding:24px;border:1px solid rgba(129,140,248,.22);border-radius:22px;background:linear-gradient(145deg,rgba(15,23,42,.82),rgba(17,24,39,.58));backdrop-filter:blur(18px);box-shadow:0 20px 55px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.04);animation:fadeIn .8s ease}.project-title{font-size:25px;font-weight:800;color:#fff;letter-spacing:.02em}.project-meta{margin-top:8px;color:#c7d2fe;font-weight:700}.project-college{margin-top:6px;color:#94a3b8;font-size:14px}.project-team-title{margin-top:18px;color:#a5b4fc;font-size:12px;text-transform:uppercase;letter-spacing:.12em;font-weight:800}.project-team{display:flex;justify-content:center;gap:18px;margin-top:10px;flex-wrap:wrap}.project-team span{min-width:250px;padding:12px 18px;border-radius:14px;background:rgba(99,102,241,.07);border:1px solid rgba(129,140,248,.14);color:#e2e8f0;line-height:1.5}.project-team small{color:#94a3b8}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}


/* Main background */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.16), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.14), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(14,165,233,0.10), transparent 35%),
        #0b1020;
}


/* Animated glow */

.stApp::before {
    content: "";
    position: fixed;
    width: 450px;
    height: 450px;
    border-radius: 50%;
    background: rgba(99,102,241,0.10);
    filter: blur(90px);
    top: 5%;
    left: -150px;
    animation: floatGlow 7s ease-in-out infinite;
    pointer-events: none;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: rgba(168,85,247,0.10);
    filter: blur(90px);
    bottom: 0;
    right: -120px;
    animation: floatGlow2 8s ease-in-out infinite;
    pointer-events: none;
}


@keyframes floatGlow {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(35px);
    }
}

@keyframes floatGlow2 {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-35px);
    }
}


/* Hero */

.hero {
    text-align: center;
    padding: 35px 20px 20px 20px;
    animation: fadeIn 1s ease;
}

.hero-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(129,140,248,0.35);
    color: #c7d2fe;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #ffffff, #a5b4fc, #e9d5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #aab3cf;
    font-size: 18px;
    margin-top: 12px;
}


@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* Cards */

.card {
    background: rgba(17,24,39,0.72);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(15px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.22);
    margin-bottom: 20px;
}


/* Text area */

textarea {
    background: rgba(15,23,42,0.85) !important;
    color: white !important;
    border: 1px solid rgba(129,140,248,0.35) !important;
    border-radius: 15px !important;
}


/* Analyze button */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    box-shadow: 0 8px 25px rgba(99,102,241,0.30);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(139,92,246,0.45);
}


/* Result */

.result {
    text-align: center;
    padding: 30px;
    border-radius: 22px;
    background: rgba(17,24,39,0.78);
    border: 1px solid rgba(255,255,255,0.10);
    animation: resultAppear 0.7s ease;
}

@keyframes resultAppear {
    from {
        opacity: 0;
        transform: scale(0.94);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.result-icon {
    font-size: 65px;
    margin-bottom: 8px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.confidence {
    color: #aab3cf;
    font-size: 17px;
    margin-top: 8px;
}


/* Probability cards */

.metric-card {
    text-align: center;
    padding: 20px;
    border-radius: 18px;
    background: rgba(17,24,39,0.70);
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: white;
}

.metric-label {
    color: #aab3cf;
    font-size: 14px;
    margin-top: 5px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(8,13,28,0.96);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    color: white;
}

.sidebar-text {
    color: #aab3cf;
    line-height: 1.6;
}


/* Footer */

.footer {
    text-align: center;
    color: #717b98;
    font-size: 13px;
    padding: 35px 0 10px 0;
}


/* Hide Streamlit menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ===== PROFESSIONAL AI UI ENHANCEMENTS ===== */

:root {
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --surface: rgba(15, 23, 42, 0.76);
    --border: rgba(148, 163, 184, 0.16);
}

/* subtle animated grid */
.stApp {
    background-image:
        linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px),
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.16), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.14), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(14,165,233,0.10), transparent 35%),
        #0b1020;
    background-size: 42px 42px, 42px 42px, auto, auto, auto, auto;
    animation: gridDrift 24s linear infinite;
}

@keyframes gridDrift {
    from { background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }
    to { background-position: 42px 42px, 42px 42px, 0 0, 0 0, 0 0; }
}

/* glass panels */
.card, .result, .metric-card {
    background: linear-gradient(145deg, rgba(15,23,42,.82), rgba(17,24,39,.62));
    border: 1px solid var(--border);
    box-shadow:
        0 20px 55px rgba(0,0,0,.24),
        inset 0 1px 0 rgba(255,255,255,.035);
    backdrop-filter: blur(18px);
}

.card {
    position: relative;
    overflow: hidden;
}

.card::before {
    content: "";
    position: absolute;
    left: -20%;
    top: 0;
    width: 140%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,.75), rgba(6,182,212,.55), transparent);
    animation: scanLine 5s ease-in-out infinite;
}

@keyframes scanLine {
    0%, 100% { transform: translateX(-25%); opacity: .35; }
    50% { transform: translateX(25%); opacity: 1; }
}

/* hero polish */
.hero {
    position: relative;
    padding-top: 46px;
}

.hero-badge {
    letter-spacing: .12em;
    text-transform: uppercase;
    box-shadow: 0 0 30px rgba(99,102,241,.12);
    animation: badgePulse 3.5s ease-in-out infinite;
}

@keyframes badgePulse {
    0%,100% { box-shadow: 0 0 0 rgba(99,102,241,0); }
    50% { box-shadow: 0 0 32px rgba(99,102,241,.22); }
}

.hero h1 {
    text-shadow: 0 0 35px rgba(129,140,248,.12);
}

/* professional input */
textarea {
    transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease !important;
}

textarea:focus {
    border-color: rgba(129,140,248,.8) !important;
    box-shadow: 0 0 0 1px rgba(129,140,248,.28), 0 0 28px rgba(99,102,241,.12) !important;
}

/* button shimmer */
.stButton > button {
    position: relative;
    overflow: hidden;
    background: linear-gradient(100deg, #6366f1, #7c3aed, #06b6d4, #7c3aed, #6366f1);
    background-size: 300% 100%;
    animation: buttonFlow 7s ease infinite;
}

.stButton > button::after {
    content: "";
    position: absolute;
    top: 0;
    left: -120%;
    width: 70%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent);
    transform: skewX(-18deg);
    animation: buttonShine 4.5s ease-in-out infinite;
}

@keyframes buttonFlow {
    0%,100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes buttonShine {
    0%,55% { left: -120%; }
    75%,100% { left: 150%; }
}

/* result reveal */
.result {
    position: relative;
    overflow: hidden;
}

.result::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(115deg, transparent 35%, rgba(255,255,255,.045) 50%, transparent 65%);
    transform: translateX(-120%);
    animation: resultSweep 3.8s ease-in-out infinite;
    pointer-events: none;
}

@keyframes resultSweep {
    0%,45% { transform: translateX(-120%); }
    70%,100% { transform: translateX(120%); }
}

.result-icon {
    filter: drop-shadow(0 0 18px rgba(255,255,255,.12));
    animation: iconFloat 3s ease-in-out infinite;
}

@keyframes iconFloat {
    0%,100% { transform: translateY(0) scale(1); }
    50% { transform: translateY(-5px) scale(1.03); }
}

.metric-card {
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.015);
    border-color: rgba(129,140,248,.30);
    box-shadow: 0 18px 40px rgba(0,0,0,.28), 0 0 25px rgba(99,102,241,.08);
}

.metric-value {
    text-shadow: 0 0 18px rgba(165,180,252,.18);
}

/* examples */
.example-panel {
    margin-top: 6px;
    margin-bottom: 18px;
    padding: 16px 18px;
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 16px;
    background: rgba(2,6,23,.35);
}

.example-label {
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 10px;
}

.example-chip {
    display: inline-block;
    margin: 4px 6px 4px 0;
    padding: 7px 11px;
    border-radius: 999px;
    border: 1px solid rgba(129,140,248,.18);
    background: rgba(99,102,241,.07);
    color: #cbd5e1;
    font-size: 12px;
}

/* footer */
.footer {
    opacity: .72;
    letter-spacing: .03em;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("word_vectorizer.pkl", "rb") as file:
        word_vectorizer = pickle.load(file)

    with open("char_vectorizer.pkl", "rb") as file:
        char_vectorizer = pickle.load(file)

    return model, word_vectorizer, char_vectorizer


model, word_vectorizer, char_vectorizer = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">💬 Sentiment AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🧠 About")

    st.markdown(
        '<div class="sidebar-text">'
        'This application uses Natural Language Processing '
        'and Machine Learning to analyze text sentiment.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ Technology")

    st.markdown(
        """
        <div class="sidebar-text">
        🐍 Python<br>
        🤖 Machine Learning<br>
        📝 TF-IDF<br>
        📊 Logistic Regression<br>
        🌐 Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📊 Model")

    st.markdown(
        '<div class="sidebar-text">'
        '<b>Accuracy:</b> 62.81%<br>'
        '<b>Classes:</b> 3<br>'
        '<b>Training samples:</b> 45,615'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
<div class="project-info">
<div class="project-title">🎓 SENTIMENT ANALYSIS USING MACHINE LEARNING</div>
<div class="project-meta">Mini Project &nbsp;•&nbsp; CSE / AIML &nbsp;•&nbsp; NSAKCET</div>
<div class="project-college">Nawab Shah Alam Khan College of Engineering and Technology (NSAKCET)</div>
<div class="project-team-title">Developed By</div>
<div class="project-team"><span>👨‍💻 <b>Khaja Ameen Uddin</b><br><small>Roll No: 161024748302</small></span><span>👨‍💻 <b>Wali Uddin</b><br><small>Roll No: 161024748305</small></span></div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="hero">
<div class="hero-badge">🤖 AI POWERED SENTIMENT ANALYSIS</div>
<h1>Understand the Emotion Behind Text</h1>
<p>Analyze text and discover whether the sentiment is Positive, Neutral, or Negative.</p>
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("### ✍️ Enter your text")

st.html("""
<div class="example-panel">
    <div class="example-label">⚡ Quick examples</div>
    <span class="example-chip">😊 “I absolutely love this product!”</span>
    <span class="example-chip">😞 “This product is terrible and disappointing.”</span>
    <span class="example-chip">😐 “The meeting is scheduled for tomorrow.”</span>
    <span class="example-chip">😊 “The service was excellent and very helpful.”</span>
    <span class="example-chip">😞 “I am unhappy with the quality.”</span>
</div>
""")

text = st.text_area(
    "",
    placeholder="Example: I absolutely love this product. It is amazing!",
    height=160,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")

    else:

        with st.spinner("🤖 AI is analyzing your text..."):
            time.sleep(0.8)

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

            # Probabilities
            probabilities = model.predict_proba(combined_features)[0]

            classes = model.classes_

            probability_dict = {
                cls: prob * 100
                for cls, prob in zip(classes, probabilities)
            }

            confidence = max(probabilities) * 100


        # =================================================
        # RESULT
        # =================================================

        if prediction == "positive":

            icon = "😊"
            title = "POSITIVE"
            description = "The text expresses a positive sentiment."

        elif prediction == "negative":

            icon = "😞"
            title = "NEGATIVE"
            description = "The text expresses a negative sentiment."

        else:

            icon = "😐"
            title = "NEUTRAL"
            description = "The text expresses a neutral sentiment."


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

        st.html("""
<div style="height:18px"></div>
<h3>📊 Sentiment Probability</h3>
""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.html(f"""
<div class="metric-card">
    <div class="metric-value">{probability_dict.get("positive", 0):.2f}%</div>
    <div class="metric-label">😊 Positive</div>
</div>
""")

        with col2:
            st.html(f"""
<div class="metric-card">
    <div class="metric-value">{probability_dict.get("neutral", 0):.2f}%</div>
    <div class="metric-label">😐 Neutral</div>
</div>
""")

        with col3:
            st.html(f"""
<div class="metric-card">
    <div class="metric-value">{probability_dict.get("negative", 0):.2f}%</div>
    <div class="metric-label">😞 Negative</div>
</div>
""")



# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Sentiment Analysis Using Machine Learning
        <br>
        CSE / AIML • NSAKCET<br>Khaja Ameen Uddin • 161024748302 &nbsp;|&nbsp; Wali Uddin • 161024748305
    </div>
    """,
    unsafe_allow_html=True
)
