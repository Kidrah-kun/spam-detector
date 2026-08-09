"""
app.py — Streamlit web interface for the AI-Based Spam Email Detection System.

This module loads four pre-trained classifiers (Naive Bayes, Logistic Regression,
Random Forest, SVM Linear) along with a TF-IDF vectorizer and feature scaler,
and provides a real-time prediction dashboard. The system combines TF-IDF
text features with 9 structural/behavioural features (message length, link count,
uppercase ratio, etc.) to classify a user-submitted message as Spam or Ham.

Project:  AI-Based Spam Email Detection System
Author:   Hardik Hathwal — B.Tech AI, Newton School of Technology
Dataset:  Synthetic dataset of common spam/ham message templates (see notebooks/)
"""

import os
import re

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")

# Spam keywords commonly associated with unsolicited messages
SPAM_TRIGGER_WORDS = [
    "free", "win", "winner", "cash", "prize", "urgent", "claim",
    "congratulations", "click", "offer", "guarantee", "limited",
    "act now", "call now", "credit", "loan", "discount",
]


@st.cache_resource
def load_models():
    """
    Load and cache all ML model artifacts from the models/ directory.

    Loads the TF-IDF vectorizer, structural feature scaler, and four
    classifiers (Naive Bayes, Logistic Regression, Random Forest, SVM).
    Results are cached by Streamlit so that disk I/O happens only once.

    Returns:
        tuple: (tfidf_vectorizer, feature_scaler, dict_of_models)
            - tfidf_vectorizer: fitted sklearn TfidfVectorizer
            - feature_scaler: fitted sklearn StandardScaler
            - dict_of_models: mapping of model name → fitted classifier
    """
    tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "feature_scaler.pkl"))
    models = {
        "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "models", "naive_bayes.pkl")),
        "Logistic Regression": joblib.load(
            os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
        ),
        "Random Forest": joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl")),
        "SVM Linear": joblib.load(os.path.join(BASE_DIR, "models", "svm.pkl")),
    }
    return tfidf, scaler, models


tfidf, scaler, models = load_models()
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text):
    """
    Normalize raw text for downstream NLP processing.

    Converts to lowercase, strips URLs, email addresses, non-alphabetic
    characters, and excess whitespace.

    Args:
        text (str): Raw message string (email body or SMS).

    Returns:
        str: Cleaned, lowercased text with URLs and emails removed.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text):
    """
    Tokenize, remove stop words, and apply Porter stemming.

    Builds on `clean_text` to produce a space-joined string of stemmed
    tokens suitable for TF-IDF vectorization.

    Args:
        text (str): Raw message string.

    Returns:
        str: Space-separated stemmed tokens, stop words excluded.
    """
    tokens = word_tokenize(clean_text(text))
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def extract_features(text):
    """
    Compute 9 structural and behavioural features from a raw message.

    Features capture statistical properties of the message text that
    complement TF-IDF bag-of-words features:
        - msg_length       : total character count
        - word_count       : number of whitespace-separated tokens
        - num_links        : count of http/https/www occurrences
        - num_exclaim      : count of exclamation marks
        - num_special      : count of non-alphanumeric characters
        - num_digits       : count of digit characters
        - upper_ratio      : fraction of characters that are uppercase
        - has_html         : 1 if HTML tags detected, else 0
        - trigger_count    : count of spam trigger keywords found

    Args:
        text (str): Raw message string (before any cleaning).

    Returns:
        tuple:
            - structural_cols (list[str]): Ordered feature column names.
            - values (list[float|int]): Corresponding feature values.
            - display (dict): Human-readable labels → values for the UI.
    """
    msg_length = len(text)
    word_count = len(text.split())
    num_links = len(re.findall(r"http[s]?://|www\.", text))
    num_exclaim = text.count("!")
    num_special = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    num_digits = len(re.findall(r"\d", text))
    num_upper = len(re.findall(r"[A-Z]", text))
    upper_ratio = num_upper / (msg_length + 1)
    has_html = int(bool(re.search(r"<[^>]+>", text)))
    trigger_count = sum(1 for w in SPAM_TRIGGER_WORDS if w in text.lower())

    # Column order must match the scaler fitted in Phase 2
    structural_cols = [
        "msg_length", "word_count", "num_links", "num_exclaim",
        "num_special", "num_digits", "upper_ratio", "has_html", "trigger_count",
    ]
    values = [
        msg_length, word_count, num_links, num_exclaim,
        num_special, num_digits, upper_ratio, has_html, trigger_count,
    ]
    display = {
        "Message Length": msg_length,
        "Word Count": word_count,
        "Links": num_links,
        "Exclamation Marks": num_exclaim,
        "Special Characters": num_special,
        "Digits": num_digits,
        "Uppercase Ratio": round(upper_ratio, 3),
        "Has HTML": has_html,
        "Spam Trigger Words": trigger_count,
    }
    return structural_cols, values, display


def predict(text, model_name):
    """
    Classify a message as Spam (1) or Ham (0) using the selected model.

    Pipeline:
        1. Preprocess text → TF-IDF vector.
        2. Extract 9 structural features → scale → sparse matrix.
        3. Horizontally stack both feature sets.
        4. For Naive Bayes, apply MinMaxScaler to ensure non-negative input.
        5. Run model.predict() and estimate confidence via predict_proba
           or decision_function where available.

    Args:
        text (str): Raw message string to classify.
        model_name (str): One of "Naive Bayes", "Logistic Regression",
                          "Random Forest", "SVM Linear".

    Returns:
        tuple:
            - pred (int): 1 = Spam, 0 = Ham.
            - confidence (float): Probability/score in [0, 1].
            - display (dict): Feature breakdown for UI rendering.
    """
    processed = preprocess(text)
    tfidf_vec = tfidf.transform([processed])
    cols, vals, display = extract_features(text)
    feat_df = pd.DataFrame([vals], columns=cols)
    structural = scaler.transform(feat_df)
    struct_sparse = csr_matrix(structural)
    X_combined = hstack([tfidf_vec, struct_sparse])

    model = models[model_name]

    if model_name == "Naive Bayes":
        mms = MinMaxScaler()
        X_combined = csr_matrix(mms.fit_transform(X_combined.toarray()))

    pred = model.predict(X_combined)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_combined)[0]
        confidence = float(prob[1] if pred == 1 else prob[0])
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_combined)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
    else:
        confidence = 1.0

    return pred, confidence, display


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🛡️ AI-Based Spam Email Detection System")
st.markdown("**Project 03 | NLP / Machine Learning | Newton School of Technology**")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📧 Enter Email / SMS Message")
    user_input = st.text_area(
        "Paste your message here:",
        height=200,
        placeholder=(
            "e.g. Congratulations! You have won a free iPhone. Click here to claim now!"
        ),
    )
    model_name = st.selectbox("🤖 Select Model", list(models.keys()))

    if st.button("🔍 Classify Message", use_container_width=True):
        if user_input.strip():
            pred, confidence, display = predict(user_input, model_name)
            st.markdown("---")
            if pred == 1:
                st.error(f"🚨 SPAM DETECTED — Confidence: {confidence:.1%}")
            else:
                st.success(f"✅ LEGITIMATE (HAM) — Confidence: {confidence:.1%}")
            st.progress(float(confidence))
            st.markdown("### 🔍 Feature Analysis")
            feat_df = pd.DataFrame(display.items(), columns=["Feature", "Value"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Please enter a message first!")

with col2:
    st.subheader("📊 Model Info")
    st.info(f"**Selected:** {model_name}")
    st.markdown("### 🧪 Example Messages")
    examples = {
        "🚨 Spam 1": "URGENT! You have won 1000 prize! Call now FREE reward claim!",
        "🚨 Spam 2": "Congratulations! Click here FREE iPhone. Limited offer expires today!",
        "✅ Ham 1": "Hey, are you coming to the meeting tomorrow at 10am?",
        "✅ Ham 2": "Can you pick up some groceries on the way home please?",
    }
    for label, example in examples.items():
        st.code(example, language=None)

    st.markdown("### 📈 Model Comparison")
    try:
        df_results = pd.read_csv(
            os.path.join(BASE_DIR, "reports", "final_evaluation_report.csv"), index_col=0
        )
        st.dataframe(df_results, use_container_width=True)
    except Exception:
        st.info("Run Phase 3 first")

st.markdown("---")
st.markdown("*Built by Hardik Hathwal | B.Tech AI | Newton School of Technology*")