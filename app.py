
import streamlit as st
import joblib
import re
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Product Review Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS - CLEAN PROFESSIONAL DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }

    .main-title {
        font-size: 32px;
        font-weight: 750;
        color: #1e293b;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        min-height: 115px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 750;
        color: #1e293b;
        margin-top: 8px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: 600;
        padding: 10px 20px;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        border: none;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/fake_review_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(?<=[a-z])['’](?=[a-z])", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# SESSION HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.title("🔍 Review Detector")
    st.caption("AI-Based Detection System")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Model Evaluation", "Analysis History"]
    )

    st.divider()

    st.markdown("### Project Details")
    st.write("**Project:** Fake Product Review Detection")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Feature Extraction:** Character TF-IDF")
    st.write("**Dataset:** DiFraud Product Reviews")

    st.divider()

    st.caption("BCA AI Mini Project")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Fake Product Review Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">An AI-powered application that identifies potentially fake or deceptive product reviews.</div>',
        unsafe_allow_html=True
    )

    # Model performance cards

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">65.06%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro F1 Score</div>
            <div class="metric-value">0.6506</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Training Reviews</div>
            <div class="metric-value">16,776</div>
        </div>
        """, unsafe_allow_html=True)

    # Review input

    st.markdown(
        '<div class="section-title">Analyze a Product Review</div>',
        unsafe_allow_html=True
    )

    review = st.text_area(
        "Enter your product review below:",
        placeholder="Example: The product quality is excellent. I am very happy with this purchase...",
        height=160
    )

    if st.button("Analyze Review", type="primary"):

        if not review.strip():
            st.warning("Please enter a product review first.")

        else:
            cleaned_review = clean_text(review)

            if not cleaned_review:
                st.warning("Please enter a review containing readable text.")

            else:
                prediction = model.predict([cleaned_review])[0]

                probabilities = model.predict_proba([cleaned_review])[0]
                classes = list(model.classes_)

                fake_index = classes.index(1)
                fake_probability = float(probabilities[fake_index]) * 100

                genuine_probability = 100 - fake_probability

                st.markdown(
                    '<div class="section-title">Analysis Result</div>',
                    unsafe_allow_html=True
                )

                if prediction == 1:
                    st.error("Prediction: Fake / Deceptive Review")
                    result_text = "Fake / Deceptive"
                else:
                    st.success("Prediction: Genuine / Non-deceptive Review")
                    result_text = "Genuine / Non-deceptive"

                # Display scores

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        "Estimated Fake-Review Score",
                        f"{fake_probability:.2f}%"
                    )

                with result_col2:
                    st.metric(
                        "Estimated Genuine-Review Score",
                        f"{genuine_probability:.2f}%"
                    )

                st.write("Fake-review score")
                st.progress(fake_probability / 100)

                st.caption(
                    "The result is an ML model prediction, not proof that a review is fake or genuine."
                )

                # Save analysis in session history

                st.session_state.history.insert(0, {
                    "review": review,
                    "prediction": result_text,
                    "fake_score": round(fake_probability, 2)
                })

                st.session_state.history = st.session_state.history[:50]


# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------

elif page == "Model Evaluation":

    st.markdown(
        '<div class="main-title">Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Performance evaluation of the trained machine learning model.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Accuracy", "65.06%")
    col2.metric("Macro F1 Score", "0.6506")
    col3.metric("Test Samples", "2,098")

    # Model comparison chart

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    if os.path.exists("results/model_comparison.png"):

        st.image(
            "results/model_comparison.png",
            caption="Comparison of different machine learning models using validation data",
            use_container_width=True
        )

    else:
        st.warning("Model comparison chart not found.")

    # Confusion matrix

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    if os.path.exists("results/confusion_matrix.png"):

        st.image(
            "results/confusion_matrix.png",
            caption="Confusion matrix showing test dataset predictions",
            use_container_width=True
        )

    else:
        st.warning("Confusion matrix chart not found.")

    # Classification report

    st.markdown(
        '<div class="section-title">Classification Report</div>',
        unsafe_allow_html=True
    )

    st.table({
        "Metric": ["Precision", "Recall", "F1 Score", "Support"],
        "Genuine (0)": ["0.65", "0.66", "0.65", "1048"],
        "Fake (1)": ["0.65", "0.64", "0.65", "1050"]
    })

    st.info(
        "The model was selected using validation macro F1 and evaluated on a separate held-out test dataset. "
        "Performance may differ on real-world reviews."
    )


# --------------------------------------------------
# ANALYSIS HISTORY PAGE
# --------------------------------------------------

elif page == "Analysis History":

    st.markdown(
        '<div class="main-title">Analysis History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Previously analyzed reviews in the current app session.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:

        st.info(
            "No analysis history available. Analyze a review from the Dashboard first."
        )

    else:

        st.metric(
            "Total Reviews Analyzed",
            len(st.session_state.history)
        )

        for index, item in enumerate(st.session_state.history, start=1):

            with st.expander(
                f"Review {index} - {item['prediction']}"
            ):

                st.write("**Review Text:**")
                st.write(item["review"])

                st.write("**Prediction:**", item["prediction"])

                st.write(
                    "**Estimated Fake Score:**",
                    f"{item['fake_score']:.2f}%"
                )

                st.progress(item["fake_score"] / 100)

        if st.button("Clear Analysis History"):

            st.session_state.history = []
            st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer">AI-Based Fake Product Review Detection | BCA Mini Project | Developed using Python, Scikit-learn and Streamlit</div>',
    unsafe_allow_html=True
)
