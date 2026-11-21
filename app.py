from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from predictor import load_model_files, predict_sentiment

st.set_page_config(
    page_title="Sentiment AI | Intelligence Workspace",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = "style.css"
with open(CSS_PATH, encoding="utf-8") as css_file:
    st.html(f"<style>{css_file.read()}</style>")


@st.cache_resource
def load_model():
    return load_model_files()


model, word_vectorizer, char_vectorizer = load_model()


PAGES = [
    "Overview",
    "Sentiment Trends",
    "Sentiment Distribution",
    "Model Confidence",
    "Analyze Text",
    "Analysis History",
    "Model Information",
]

EXAMPLES = [
    ("Positive", "I absolutely love this product. The quality is outstanding!"),
    ("Negative", "This experience was terrible and extremely disappointing."),
    ("Neutral", "The meeting is scheduled for tomorrow afternoon."),
]

EVALUATION = {
    "accuracy": 66.47,
    "test_samples": 9123,
    "classes": {
        "negative": {"precision": 0.48, "recall": 0.64, "f1": 0.55, "support": 1419},
        "neutral": {"precision": 0.70, "recall": 0.65, "f1": 0.67, "support": 4134},
        "positive": {"precision": 0.73, "recall": 0.69, "f1": 0.71, "support": 3570},
    },
    "confusion_matrix": [
        [908, 361, 150],
        [696, 2677, 761],
        [300, 791, 2479],
    ],
}

if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = ""

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"


history = st.session_state.analysis_history


def page_header(section: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="sa-page-head">
            <div class="sa-eyebrow">{html.escape(section)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(
        f'<div class="sa-section-label">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="sa-metric-card">
            <div class="sa-metric-label">{html.escape(label)}</div>
            <div class="sa-metric-value">{html.escape(value)}</div>
            <div class="sa-metric-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_counts(records: list) -> dict:
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for record in records:
        label = str(record["prediction"]).lower()
        if label in counts:
            counts[label] += 1
    return counts


def average_confidence(records: list) -> float:
    if not records:
        return 0.0
    return sum(float(item["confidence"]) for item in records) / len(records)


def show_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="sa-empty"><span>✳</span><p>{html.escape(message)}</p></div>',
        unsafe_allow_html=True,
    )


def render_result(result: dict) -> None:
    prediction = str(result["prediction"]).lower()
    confidence = max(0.0, min(100.0, float(result["confidence"])))
    probabilities = result["probabilities"]

    symbols = {"positive": "↗", "negative": "↘", "neutral": "—"}
    descriptions = {
        "positive": "The model identified positive sentiment in this text.",
        "negative": "The model identified negative sentiment in this text.",
        "neutral": "The model identified neutral sentiment in this text.",
    }

    label = prediction.upper()
    style = prediction if prediction in symbols else "neutral"

    st.markdown(
        f"""
        <div class="sa-result-card {style}">
            <div class="sa-result-top">
                <span>DETECTED SENTIMENT</span>
                <span>ANALYSIS COMPLETE</span>
            </div>
            <div class="sa-result-main">
                <div class="sa-result-symbol">{symbols.get(style, "?")}</div>
                <div class="sa-result-copy">
                    <div class="sa-result-kicker">YOUR TEXT EXPRESSES</div>
                    <div class="sa-result-name">{html.escape(label)}</div>
                    <div class="sa-result-description">{descriptions.get(style, "Prediction completed.")}</div>
                </div>
                <div class="sa-result-confidence">
                    <strong>{confidence:.2f}%</strong>
                    <span>MODEL CONFIDENCE</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(confidence / 100)

    section_label("CLASS PROBABILITIES")
    for sentiment in ("positive", "neutral", "negative"):
        value = max(0.0, min(100.0, float(probabilities.get(sentiment, 0))))
        st.markdown(
            f"""
            <div class="sa-probability">
                <div><span>{sentiment.capitalize()}</span><strong>{value:.2f}%</strong></div>
                <div class="sa-probability-track">
                    <div class="{sentiment}" style="width:{value:.2f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View analyzed text"):
        st.write(result["text"])


# Sidebar navigation
with st.sidebar:
    st.markdown(
        """
        <div class="sa-brand">
            <div class="sa-brand-mark">✳</div>
            <div>
                <div class="sa-brand-name">sentiment<span>AI</span></div>
                <div class="sa-brand-caption">INTELLIGENCE WORKSPACE</div>
            </div>
        </div>
        <div class="sa-sidebar-heading">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Dashboard navigation",
        PAGES,
        index=PAGES.index(st.session_state.active_page),
        label_visibility="collapsed",
        key="dashboard_navigation",
    )
    st.session_state.active_page = selected_page

    counts = sentiment_counts(history)
    st.markdown(
        """
        <div class="sa-sidebar-heading">LIVE SESSION</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sa-sidebar-stats">
            <div><strong>{len(history)}</strong><span>Total analyses</span></div>
            <div><strong>{counts["positive"]}</strong><span>Positive results</span></div>
        </div>
        <div class="sa-sidebar-model">
            <div class="sa-sidebar-heading">MODEL ENGINE</div>
            <strong>{html.escape(type(model).__name__)}</strong>
            <p>Word TF-IDF + Character TF-IDF</p>
            <span>CHARACTER WEIGHT · 1.2</span>
        </div>
        <div class="sa-sidebar-status"><i></i> MODEL READY</div>
        """,
        unsafe_allow_html=True,
    )


# Overview
if selected_page == "Overview":
    page_header(
        "01 / WORKSPACE OVERVIEW",
        "Understand the bigger picture.",
        "A live view of your sentiment analysis activity and model performance.",
    )

    counts = sentiment_counts(history)
    total = len(history)
    average = average_confidence(history)

    columns = st.columns(4)
    with columns[0]:
        metric_card("TOTAL ANALYSES", str(total), "Current session")
    with columns[1]:
        metric_card("POSITIVE", str(counts["positive"]), "Session predictions")
    with columns[2]:
        metric_card("NEGATIVE", str(counts["negative"]), "Session predictions")
    with columns[3]:
        metric_card("AVERAGE CONFIDENCE", f"{average:.1f}%" if total else "—", "Current session")

    if st.button("Analyze New Text", key="overview_analyze", type="primary", use_container_width=True):
        st.session_state.active_page = "Analyze Text"
        st.rerun()
    st.write("")
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        section_label("SESSION ACTIVITY")
        st.markdown("### Your analysis activity")
        if history:
            activity = pd.DataFrame(history)
            activity["timestamp"] = pd.to_datetime(activity["timestamp"])
            chart_data = activity.set_index("timestamp")[["confidence"]]
            st.line_chart(chart_data, height=270)
        else:
            show_empty_state("Your activity chart will appear after your first analysis.")

    with right:
        section_label("SENTIMENT SNAPSHOT")
        st.markdown("### Current session distribution")
        if total:
            st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Analyses"]))
        else:
            show_empty_state("No sentiment results yet. Start by analyzing a piece of text.")

    section_label("MODEL BENCHMARK")
    st.markdown("### Evaluation on the held-out test set")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("TEST ACCURACY", "66.47%", "Recorded evaluation")
    with c2:
        metric_card("TEST SAMPLES", "9,123", "Held-out examples")
    with c3:
        metric_card("SENTIMENT CLASSES", "03", "Positive · Neutral · Negative")


# Sentiment Trends
elif selected_page == "Sentiment Trends":
    page_header(
        "02 / TEMPORAL ANALYTICS",
        "Follow the sentiment.",
        "Explore how your predictions evolve throughout this active session.",
    )

    if not history:
        show_empty_state("No session data yet. Analyze some text to generate your first trend.")
    else:
        frame = pd.DataFrame(history)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values("timestamp")

        section_label("CONFIDENCE OVER TIME")
        st.line_chart(frame.set_index("timestamp")[["confidence"]], height=320)

        section_label("PREDICTION COUNTS")
        counts = sentiment_counts(history)
        st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Predictions"]))

        st.caption("Charts use actual predictions from this session only. They reset when the session ends.")


# Sentiment Distribution
elif selected_page == "Sentiment Distribution":
    page_header(
        "03 / SENTIMENT COMPOSITION",
        "The balance of emotion.",
        "See how positive, neutral and negative predictions are distributed in this session.",
    )

    counts = sentiment_counts(history)
    total = len(history)

    if not total:
        show_empty_state("Your distribution will appear after your first sentiment analysis.")
    else:
        columns = st.columns(3)
        for column, sentiment in zip(columns, ("positive", "neutral", "negative")):
            with column:
                percentage = counts[sentiment] / total * 100
                metric_card(sentiment.upper(), str(counts[sentiment]), f"{percentage:.1f}% of session")

        section_label("SESSION DISTRIBUTION")
        st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["Predictions"]))

        st.markdown("### Distribution details")
        for sentiment in ("positive", "neutral", "negative"):
            value = counts[sentiment]
            percentage = value / total * 100
            st.markdown(
                f"""
                <div class="sa-distribution-row">
                    <div><span>{sentiment.capitalize()}</span><strong>{value} · {percentage:.1f}%</strong></div>
                    <div class="sa-distribution-track"><div class="{sentiment}" style="width:{percentage:.2f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Model Confidence
elif selected_page == "Model Confidence":
    page_header(
        "04 / PREDICTION CERTAINTY",
        "Look closer at confidence.",
        "Understand the confidence scores returned by your trained classifier.",
    )

    if not history:
        show_empty_state("Confidence analytics will appear after you analyze some text.")
    else:
        scores = [float(item["confidence"]) for item in history]
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("AVERAGE", f"{sum(scores) / len(scores):.2f}%", "Session confidence")
        with c2:
            metric_card("HIGHEST", f"{max(scores):.2f}%", "Session maximum")
        with c3:
            metric_card("LOWEST", f"{min(scores):.2f}%", "Session minimum")

        section_label("CONFIDENCE BY ANALYSIS")
        confidence_frame = pd.DataFrame({
            "Analysis": [f"#{i + 1}" for i in range(len(scores))],
            "Confidence": scores,
        }).set_index("Analysis")
        st.bar_chart(confidence_frame, height=320)

        st.caption("Confidence is the highest class probability returned by the model. It is not a guarantee of correctness.")


# Analyze Text
elif selected_page == "Analyze Text":
    page_header(
        "05 / SENTIMENT ANALYSIS",
        "Read between the emotions.",
        "Enter a review, comment or sentence and let your trained model analyze its sentiment.",
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left, st.container(border=True):
        section_label("INPUT TEXT")
        st.markdown("### What would you like to understand?")
        text = st.text_area(
            "Text for sentiment analysis",
            placeholder="Start typing or paste your text here...",
            height=210,
            key="analysis_text",
            label_visibility="collapsed",
        )
        st.caption(f"{len(text)} characters · English text works best with this model.")

        if st.button("Analyze sentiment  →", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                try:
                    with st.spinner("Analyzing language patterns..."):
                        prediction, probabilities, confidence = predict_sentiment(
                            text, model, word_vectorizer, char_vectorizer
                        )

                    result = {
                        "text": text,
                        "prediction": str(prediction).lower(),
                        "probabilities": probabilities,
                        "confidence": float(confidence),
                    }
                    st.session_state.analysis_result = result
                    history.append({
                        **result,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    })
                    st.success("Analysis completed and added to session history.")
                except Exception as error:  # noqa: BLE001
                    st.error(f"Analysis could not be completed: {error}")

    with right, st.container(border=True):
            section_label("QUICK START")
            st.markdown("### Explore with examples")
            st.caption("Choose an example to load it into the analysis field.")
            for label, example in EXAMPLES:
                if st.button(f"Load {label.lower()} example", use_container_width=True, key=f"load_{label}"):
                    st.session_state.analysis_text = example
                    st.rerun()

            st.markdown("---")
            st.markdown("**Three sentiment dimensions**")
            st.markdown(
                '<div class="sa-class-tags"><span class="positive">Positive</span>'
                '<span class="neutral">Neutral</span><span class="negative">Negative</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Predictions are generated by your trained word and character TF-IDF model.")

    result = st.session_state.analysis_result
    if result:
        st.write("")
        section_label("LATEST RESULT")
        render_result(result)


# Analysis History
elif selected_page == "Analysis History":
    page_header(
        "06 / SESSION RECORDS",
        "Your analysis history.",
        "Review, inspect and export predictions made during this active session.",
    )

    if not history:
        show_empty_state("Your analysis history is empty. Your completed analyses will appear here.")
    else:
        counts = sentiment_counts(history)
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("TOTAL RECORDS", str(len(history)), "This session")
        with c2:
            metric_card("POSITIVE", str(counts["positive"]), "Recorded predictions")
        with c3:
            metric_card("NEGATIVE", str(counts["negative"]), "Recorded predictions")

        frame = pd.DataFrame(history).copy()
        frame["text_preview"] = frame["text"].astype(str).str.slice(0, 100)
        display = frame[["timestamp", "text_preview", "prediction", "confidence"]].rename(
            columns={
                "timestamp": "Timestamp",
                "text_preview": "Analyzed text",
                "prediction": "Sentiment",
                "confidence": "Confidence (%)",
            }
        )
        st.dataframe(display.iloc[::-1], use_container_width=True, hide_index=True)

        csv_data = frame[["timestamp", "text", "prediction", "confidence"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export session history as CSV",
            data=csv_data,
            file_name="sentiment_ai_session_history.csv",
            mime="text/csv",
            type="primary",
        )

        if st.button("Clear session history", type="secondary"):
            st.session_state.analysis_history = []
            st.session_state.analysis_result = None
            st.rerun()


# Model Information
elif selected_page == "Model Information":
    page_header(
        "07 / MODEL INTELLIGENCE",
        "Know the engine.",
        "Technical details and recorded evaluation results for your sentiment classifier.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("TEST ACCURACY", "66.47%", "Held-out evaluation")
    with c2:
        metric_card("TEST SAMPLES", "9,123", "Evaluation support")
    with c3:
        metric_card("CLASS COUNT", "3", "Model output classes")

    section_label("MODEL ARCHITECTURE")
    st.markdown("### Hybrid TF-IDF + Logistic Regression")
    details = [
        ("Classifier", type(model).__name__),
        ("Word vectorizer", type(word_vectorizer).__name__),
        ("Character vectorizer", type(char_vectorizer).__name__),
        ("Character feature weight", "1.2"),
        ("Prediction interface", "predict + predict_proba"),
    ]
    for label, value in details:
        st.markdown(
            f'<div class="sa-detail-row"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>',
            unsafe_allow_html=True,
        )

    word_features = len(getattr(word_vectorizer, "vocabulary_", {}))
    char_features = len(getattr(char_vectorizer, "vocabulary_", {}))
    c1, c2 = st.columns(2)
    with c1:
        metric_card("WORD FEATURES", f"{word_features:,}", "Learned vocabulary size")
    with c2:
        metric_card("CHARACTER FEATURES", f"{char_features:,}", "Learned vocabulary size")

    section_label("CLASSIFICATION REPORT")
    report_rows = []
    for label, values in EVALUATION["classes"].items():
        report_rows.append({
            "Class": label.capitalize(),
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1 score": values["f1"],
            "Support": values["support"],
        })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    section_label("CONFUSION MATRIX")
    matrix = pd.DataFrame(
        EVALUATION["confusion_matrix"],
        index=["Actual negative", "Actual neutral", "Actual positive"],
        columns=["Predicted negative", "Predicted neutral", "Predicted positive"],
    )
    st.dataframe(matrix, use_container_width=True)

    st.caption("Evaluation values are taken from model_evaluation.txt. Session analytics are separate from these test-set metrics.")


st.markdown(
    """
    <div class="sa-footer">
        <span>SENTIMENT AI</span>
        <span>Machine Learning · Natural Language Processing</span>
        <span>NSAKCET</span>
    </div>
    """,
    unsafe_allow_html=True,
)