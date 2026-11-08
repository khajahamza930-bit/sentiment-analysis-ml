import streamlit as st
import time

from predictor import load_model_files, predict_sentiment


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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html,
    body,
    [class*="css"] {
        font-family: 'Inter', sans-serif;
    }


    /* =====================================================
       MAIN BACKGROUND
       ===================================================== */

    .stApp {
        background-image:
            linear-gradient(
                rgba(148,163,184,0.035) 1px,
                transparent 1px
            ),
            linear-gradient(
                90deg,
                rgba(148,163,184,0.035) 1px,
                transparent 1px
            ),
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(168,85,247,0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(14,165,233,0.10),
                transparent 35%
            ),
            #0b1020;

        background-size:
            42px 42px,
            42px 42px,
            auto,
            auto,
            auto,
            auto;

        animation: gridDrift 24s linear infinite;
    }

    @keyframes gridDrift {
        from {
            background-position:
                0 0,
                0 0,
                0 0,
                0 0,
                0 0;
        }

        to {
            background-position:
                42px 42px,
                42px 42px,
                0 0,
                0 0,
                0 0;
        }
    }


    /* =====================================================
       ANIMATED GLOW
       ===================================================== */

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

        0%,
        100% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(35px);
        }
    }

    @keyframes floatGlow2 {

        0%,
        100% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-35px);
        }
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        text-align: center;

        padding: 46px 20px 20px 20px;

        animation: fadeIn 1s ease;

        position: relative;
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

        letter-spacing: .12em;

        text-transform: uppercase;

        box-shadow:
            0 0 30px rgba(99,102,241,.12);

        animation:
            badgePulse 3.5s ease-in-out infinite;
    }

    .hero h1 {
        font-size: 48px;

        font-weight: 800;

        margin: 0;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #a5b4fc,
                #e9d5ff
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        text-shadow:
            0 0 35px rgba(129,140,248,.12);
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

    @keyframes badgePulse {

        0%,
        100% {
            box-shadow:
                0 0 0 rgba(99,102,241,0);
        }

        50% {
            box-shadow:
                0 0 32px rgba(99,102,241,.22);
        }
    }


    /* =====================================================
       GLASS CARDS
       ===================================================== */

    .card,
    .result,
    .metric-card {

        background:
            linear-gradient(
                145deg,
                rgba(15,23,42,.82),
                rgba(17,24,39,.62)
            );

        border:
            1px solid rgba(148,163,184,.16);

        box-shadow:
            0 20px 55px rgba(0,0,0,.24),
            inset 0 1px 0 rgba(255,255,255,.035);

        backdrop-filter: blur(18px);
    }

    .card {

        position: relative;

        overflow: hidden;

        border-radius: 20px;

        padding: 25px;

        margin-bottom: 20px;
    }

    .card::before {

        content: "";

        position: absolute;

        left: -20%;

        top: 0;

        width: 140%;

        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(124,58,237,.75),
                rgba(6,182,212,.55),
                transparent
            );

        animation:
            scanLine 5s ease-in-out infinite;
    }

    @keyframes scanLine {

        0%,
        100% {
            transform: translateX(-25%);
            opacity: .35;
        }

        50% {
            transform: translateX(25%);
            opacity: 1;
        }
    }


    /* =====================================================
       TEXT AREA
       ===================================================== */

    textarea {

        background:
            rgba(15,23,42,0.85) !important;

        color: white !important;

        border:
            1px solid rgba(129,140,248,0.35) !important;

        border-radius:
            15px !important;

        transition:
            border-color .25s ease,
            box-shadow .25s ease,
            transform .25s ease !important;
    }

    textarea:focus {

        border-color:
            rgba(129,140,248,.8) !important;

        box-shadow:
            0 0 0 1px rgba(129,140,248,.28),
            0 0 28px rgba(99,102,241,.12) !important;
    }


    /* =====================================================
       ANALYZE BUTTON
       ===================================================== */

    .stButton > button {

        width: 100%;

        border-radius: 14px;

        border: none;

        padding: 14px;

        font-size: 17px;

        font-weight: 700;

        color: white;

        background:
            linear-gradient(
                100deg,
                #6366f1,
                #7c3aed,
                #06b6d4,
                #7c3aed,
                #6366f1
            );

        background-size: 300% 100%;

        box-shadow:
            0 8px 25px rgba(99,102,241,0.30);

        transition: all 0.25s ease;

        animation:
            buttonFlow 7s ease infinite;

        position: relative;

        overflow: hidden;
    }

    .stButton > button:hover {

        transform: translateY(-3px);

        box-shadow:
            0 12px 35px rgba(139,92,246,0.45);
    }

    .stButton > button::after {

        content: "";

        position: absolute;

        top: 0;

        left: -120%;

        width: 70%;

        height: 100%;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(255,255,255,.18),
                transparent
            );

        transform: skewX(-18deg);

        animation:
            buttonShine 4.5s ease-in-out infinite;
    }

    @keyframes buttonFlow {

        0%,
        100% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }
    }

    @keyframes buttonShine {

        0%,
        55% {
            left: -120%;
        }

        75%,
        100% {
            left: 150%;
        }
    }


    /* =====================================================
       RESULT
       ===================================================== */

    .result {

        text-align: center;

        padding: 30px;

        border-radius: 22px;

        animation:
            resultAppear 0.7s ease;

        position: relative;

        overflow: hidden;
    }

    .result::after {

        content: "";

        position: absolute;

        inset: 0;

        background:
            linear-gradient(
                115deg,
                transparent 35%,
                rgba(255,255,255,.045) 50%,
                transparent 65%
            );

        transform:
            translateX(-120%);

        animation:
            resultSweep 3.8s ease-in-out infinite;

        pointer-events: none;
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

    @keyframes resultSweep {

        0%,
        45% {
            transform:
                translateX(-120%);
        }

        70%,
        100% {
            transform:
                translateX(120%);
        }
    }

    .result-icon {

        font-size: 65px;

        margin-bottom: 8px;

        filter:
            drop-shadow(
                0 0 18px rgba(255,255,255,.12)
            );

        animation:
            iconFloat 3s ease-in-out infinite;
    }

    @keyframes iconFloat {

        0%,
        100% {
            transform:
                translateY(0)
                scale(1);
        }

        50% {
            transform:
                translateY(-5px)
                scale(1.03);
        }
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


    /* =====================================================
       PROBABILITY CARDS
       ===================================================== */

    .metric-card {

        text-align: center;

        padding: 20px;

        border-radius: 18px;

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease,
            border-color 0.25s ease;

        position: relative;

        overflow: hidden;
    }

    .metric-card:hover {

        transform:
            translateY(-6px)
            scale(1.015);

        border-color:
            rgba(129,140,248,.30);

        box-shadow:
            0 18px 40px rgba(0,0,0,.28),
            0 0 25px rgba(99,102,241,.08);
    }

    .metric-value {

        font-size: 28px;

        font-weight: 800;

        color: white;

        text-shadow:
            0 0 18px rgba(165,180,252,.18);
    }

    .metric-label {

        color: #aab3cf;

        font-size: 14px;

        margin-top: 5px;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {

        background:
            radial-gradient(
                circle at 15% 8%,
                rgba(99,102,241,.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 72%,
                rgba(6,182,212,.10),
                transparent 30%
            ),
            linear-gradient(
                160deg,
                rgba(7,12,28,.99),
                rgba(11,17,36,.95)
            ) !important;

        border-right:
            1px solid rgba(129,140,248,.16);

        box-shadow:
            18px 0 50px rgba(0,0,0,.20);

        position: relative;

        overflow: hidden;
    }

    section[data-testid="stSidebar"]::before {

        content: "";

        position: absolute;

        width: 280px;
        height: 280px;

        left: -165px;
        top: 30%;

        border-radius: 50%;

        background:
            rgba(99,102,241,.13);

        filter: blur(70px);

        animation:
            sideGlow 8s ease-in-out infinite;

        pointer-events: none;
    }

    section[data-testid="stSidebar"]::after {

        content: "";

        position: absolute;

        inset: 0;

        background:
            linear-gradient(
                115deg,
                transparent 0%,
                rgba(255,255,255,.035) 48%,
                transparent 56%
            );

        transform:
            translateX(-120%);

        animation:
            sideSweep 9s ease-in-out infinite;

        pointer-events: none;
    }

    @keyframes sideGlow {

        0%,
        100% {
            transform:
                translate(0,0)
                scale(1);

            opacity: .55;
        }

        50% {

            transform:
                translate(85px,-30px)
                scale(1.18);

            opacity: 1;
        }
    }

    @keyframes sideSweep {

        0%,
        64% {
            transform:
                translateX(-120%);
        }

        82%,
        100% {
            transform:
                translateX(120%);
        }
    }

    section[data-testid="stSidebar"] > div:first-child {

        position: relative;

        z-index: 2;
    }


    /* =====================================================
       SIDEBAR BRAND
       ===================================================== */

    .sidebar-brand {

        display: flex;

        align-items: center;

        gap: 12px;

        padding:
            8px 2px 24px 2px;
    }

    .sidebar-logo {

        width: 44px;
        height: 44px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,.25),
                rgba(6,182,212,.18)
            );

        border:
            1px solid rgba(129,140,248,.25);

        font-size: 22px;

        box-shadow:
            0 8px 25px rgba(99,102,241,.15);
    }

    .sidebar-title {

        font-size: 21px;

        font-weight: 800;

        color: #ffffff;

        letter-spacing: -.02em;
    }

    .sidebar-subtitle {

        margin-top: 3px;

        color: #7dd3fc;

        font-size: 8px;

        font-weight: 800;

        letter-spacing: .16em;
    }


    /* =====================================================
       SIDEBAR TEXT
       ===================================================== */

    .sidebar-text {

        color: #aeb9d2;

        line-height: 1.7;

        font-size: 13.5px;
    }

    section[data-testid="stSidebar"] hr {

        border: 0;

        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(129,140,248,.32),
                rgba(6,182,212,.20),
                transparent
            );

        margin: 18px 0;
    }

    section[data-testid="stSidebar"] h3 {

        color: #e2e8f0 !important;

        font-size: 14px !important;

        letter-spacing: .04em;
    }


    /* =====================================================
       SIDEBAR SECTIONS
       ===================================================== */

    .side-section {

        padding: 17px 0;

        border-top:
            1px solid rgba(148,163,184,.10);
    }

    .side-heading {

        color: #e2e8f0;

        font-size: 13px;

        font-weight: 700;

        letter-spacing: .04em;

        margin-bottom: 10px;
    }

    .side-description {

        color: #94a3b8;

        font-size: 12px;

        line-height: 1.7;
    }


    /* =====================================================
       TECHNOLOGY CHIPS
       ===================================================== */

    .tech-grid {

        display: flex;

        flex-wrap: wrap;

        gap: 7px;
    }

    .tech-chip {

        display: inline-block;

        padding: 6px 9px;

        border-radius: 8px;

        background:
            rgba(99,102,241,.08);

        border:
            1px solid rgba(129,140,248,.14);

        color: #cbd5e1;

        font-size: 10px;

        font-weight: 600;

        transition:
            all .25s ease;
    }

    .tech-chip:hover {

        background:
            rgba(99,102,241,.16);

        border-color:
            rgba(129,140,248,.30);

        transform:
            translateY(-2px);
    }


    /* =====================================================
       MODEL STATS
       ===================================================== */

    .model-grid {

        display: grid;

        grid-template-columns:
            1fr 1fr;

        gap: 8px;
    }

    .model-stat {

        padding: 11px 9px;

        text-align: center;

        border-radius: 11px;

        background:
            rgba(15,23,42,.55);

        border:
            1px solid rgba(148,163,184,.10);

        transition:
            all .25s ease;
    }

    .model-stat.wide {

        grid-column:
            span 2;
    }

    .model-stat:hover {

        transform:
            translateY(-2px);

        border-color:
            rgba(129,140,248,.25);
    }

    .model-value {

        color: #e0e7ff;

        font-size: 17px;

        font-weight: 800;
    }

    .model-label {

        margin-top: 3px;

        color: #64748b;

        font-size: 9px;

        text-transform:
            uppercase;

        letter-spacing: .08em;
    }


    /* =====================================================
       DEVELOPER CARD
       ===================================================== */

    .developer-card {

        margin-top: 17px;

        padding: 15px;

        border-radius: 15px;

        background:
            linear-gradient(
                145deg,
                rgba(99,102,241,.11),
                rgba(6,182,212,.06)
            );

        border:
            1px solid rgba(129,140,248,.17);

        box-shadow:
            0 10px 30px rgba(0,0,0,.18);

        transition:
            all .3s ease;
    }

    .developer-card:hover {

        transform:
            translateY(-3px);

        border-color:
            rgba(129,140,248,.32);

        box-shadow:
            0 15px 35px rgba(99,102,241,.12);
    }

    .developer-heading {

        color: #c7d2fe;

        font-size: 11px;

        font-weight: 800;

        letter-spacing: .08em;

        text-transform:
            uppercase;

        margin-bottom: 14px;
    }

    .developer-name {

        color: #f1f5f9;

        font-size: 12px;

        font-weight: 700;
    }

    .developer-roll {

        color: #64748b;

        font-size: 10px;

        margin-top: 3px;
    }

    .developer-divider {

        height: 1px;

        margin: 11px 0;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(129,140,248,.25),
                transparent
            );
    }


    /* =====================================================
       EXAMPLE PANEL
       ===================================================== */

    .example-panel {

        margin-top: 6px;

        margin-bottom: 18px;

        padding: 16px 18px;

        border:
            1px solid rgba(148,163,184,.13);

        border-radius: 16px;

        background:
            rgba(2,6,23,.35);
    }

    .example-label {

        color: #94a3b8;

        font-size: 12px;

        letter-spacing: .08em;

        text-transform:
            uppercase;

        font-weight: 700;

        margin-bottom: 10px;
    }

    .example-chip {

        display: inline-block;

        margin: 4px 6px 4px 0;

        padding: 7px 11px;

        border-radius: 999px;

        border:
            1px solid rgba(129,140,248,.18);

        background:
            rgba(99,102,241,.07);

        color: #cbd5e1;

        font-size: 12px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {

        text-align: center;

        color: #717b98;

        font-size: 13px;

        padding: 35px 0 10px 0;

        opacity: .72;

        letter-spacing: .03em;
    }


    /* =====================================================
       HIDE STREAMLIT DEFAULT UI
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return load_model_files()


model, word_vectorizer, char_vectorizer = load_model()


# =========================================================
# SIDEBAR
# =========================================================

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                💬
            </div>

            <div>

                <div class="sidebar-title">
                    Sentiment AI
                </div>

                <div class="sidebar-subtitle">
                    AI • NLP • MACHINE LEARNING
                </div>

            </div>

        </div>


        <div class="side-section">

            <div class="side-heading">
                📌 About
            </div>

            <div class="side-description">
                AI-powered sentiment analysis that uses
                Natural Language Processing and Machine
                Learning to understand the emotion behind text.
            </div>

        </div>


        <div class="side-section">

            <div class="side-heading">
                🛠️ Technology
            </div>

            <div class="tech-grid">

                <span class="tech-chip">
                    🐍 Python
                </span>

                <span class="tech-chip">
                    🤖 Machine Learning
                </span>

                <span class="tech-chip">
                    📝 TF-IDF
                </span>

                <span class="tech-chip">
                    📊 Logistic Regression
                </span>

                <span class="tech-chip">
                    🌐 Streamlit
                </span>

            </div>

        </div>


        <div class="side-section">

            <div class="side-heading">
                🧠 Model
            </div>

            <div class="model-grid">

                <div class="model-stat">

                    <div class="model-value">

                    </div>

                    <div class="model-label">
                        Accuracy
                    </div>

                </div>


                <div class="model-stat">

                    <div class="model-value">
                        3
                    </div>

                    <div class="model-label">
                        Classes
                    </div>

                </div>


                <div class="model-stat wide">

                    <div class="model-value">
                        45,615
                    </div>

                    <div class="model-label">
                        Training Samples
                    </div>

                </div>

            </div>

        </div>


        <div class="developer-card">

            <div class="developer-heading">
                👨‍💻 Developed By
            </div>


            <div class="developer-person">

                <div class="developer-name">
                    Khaja Ameen Uddin
                </div>

                <div class="developer-roll">
                    Roll No: 161024748302
                </div>

            </div>

        </div>
        """
    )

# =========================================================
# HERO SECTION
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="hero-badge">
            🤖 AI POWERED SENTIMENT ANALYSIS
        </div>

        <h1>
            Understand the Emotion Behind Text
        </h1>

        <p>
            Analyze text and discover whether the sentiment
            is Positive, Neutral, or Negative.
        </p>

    </div>
    """
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("### ✍️ Enter your text")


# =========================================================
# QUICK EXAMPLES
# =========================================================

st.html(
    """
    <div class="example-panel">

        <div class="example-label">
            ⚡ Quick examples
        </div>

        <span class="example-chip">
            😊 “I absolutely love this product!”
        </span>

        <span class="example-chip">
            😞 “This product is terrible and disappointing.”
        </span>

        <span class="example-chip">
            😐 “The meeting is scheduled for tomorrow.”
        </span>

        <span class="example-chip">
            😊 “The service was excellent and very helpful.”
        </span>

        <span class="example-chip">
            😞 “I am unhappy with the quality.”
        </span>

    </div>
    """
)


# =========================================================
# TEXT INPUT
# =========================================================

text = st.text_area(
    "Enter text for sentiment analysis",
    placeholder=(
        "Example: I absolutely love this product. "
        "It is amazing!"
    ),
    height=160,
    label_visibility="collapsed"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    # -----------------------------------------------------
    # EMPTY INPUT CHECK
    # -----------------------------------------------------

    if text.strip() == "":

        st.warning(
            "⚠️ Please enter some text before analyzing."
        )

    else:

        # -------------------------------------------------
        # LOADING
        # -------------------------------------------------

        with st.spinner(
            "🤖 AI is analyzing your text..."
        ):

            time.sleep(0.8)


            prediction, probability_dict, confidence = predict_sentiment(
                text,
                model,
                word_vectorizer,
                char_vectorizer,
            )


        # =================================================
        # RESULT INFORMATION
        # =================================================

        if prediction == "positive":

            icon = "😊"

            title = "POSITIVE"

            description = (
                "The text expresses a positive sentiment."
            )

        elif prediction == "negative":

            icon = "😞"

            title = "NEGATIVE"

            description = (
                "The text expresses a negative sentiment."
            )

        else:

            icon = "😐"

            title = "NEUTRAL"

            description = (
                "The text expresses a neutral sentiment."
            )


        # =================================================
        # RESULT CARD
        # =================================================

        st.html(
            f"""
            <div class="result">

                <div class="result-icon">
                    {icon}
                </div>

                <div class="result-title">
                    {title}
                </div>

                <div class="confidence">

                    {description}

                    <br>
                    <br>

                    Confidence:
                    <b>{confidence:.2f}%</b>

                </div>

            </div>
            """
        )


        # =================================================
        # PROBABILITY TITLE
        # =================================================

        st.html(
            """
            <div style="height:18px;"></div>

            <h3>
                📊 Sentiment Probability
            </h3>
            """
        )


        # =================================================
        # PROBABILITY CARDS
        # =================================================

        col1, col2, col3 = st.columns(3)


        # -------------------------------------------------
        # POSITIVE
        # -------------------------------------------------

        with col1:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {probability_dict.get("positive", 0):.2f}%
                    </div>

                    <div class="metric-label">
                        😊 Positive
                    </div>

                </div>
                """
            )


        # -------------------------------------------------
        # NEUTRAL
        # -------------------------------------------------

        with col2:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {probability_dict.get("neutral", 0):.2f}%
                    </div>

                    <div class="metric-label">
                        😐 Neutral
                    </div>

                </div>
                """
            )


        # -------------------------------------------------
        # NEGATIVE
        # -------------------------------------------------

        with col3:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {probability_dict.get("negative", 0):.2f}%
                    </div>

                    <div class="metric-label">
                        😞 Negative
                    </div>

                </div>
                """
            )


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="footer">

        Sentiment Analysis Using Machine Learning

        <div style="margin-top: 10px;">
            CSE / AIML • NSAKCET
        </div>

        <div style="margin-top: 10px; font-size: 12px;">
            Developed by Khaja Ameen Uddin
        </div>

    </div>
    """
)