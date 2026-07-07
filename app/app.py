import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf  = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    models = {
        "Naive Bayes":         joblib.load("models/naive_bayes.pkl"),
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest":       joblib.load("models/random_forest.pkl"),
        "SVM Linear":          joblib.load("models/svm.pkl"),
    }
    return tfidf, scaler, models

tfidf, scaler, models = load_models()

# ── Preprocessing ─────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
stemmer    = PorterStemmer()

spam_trigger_words = ["free","win","winner","cash","prize","urgent","claim",
                      "congratulations","click","offer","guarantee","limited",
                      "act now","call now","credit","loan","discount"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    tokens = word_tokenize(clean_text(text))
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def extract_features(text):
    return {
        "msg_length":    len(text),
        "word_count":    len(text.split()),
        "num_links":     len(re.findall(r"http[s]?://|www\.", text)),
        "num_exclaim":   text.count("!"),
        "num_special":   len(re.findall(r"[^a-zA-Z0-9\s]", text)),
        "num_digits":    len(re.findall(r"\d", text)),
        "num_upper":     len(re.findall(r"[A-Z]", text)),
        "upper_ratio":   len(re.findall(r"[A-Z]", text)) / (len(text) + 1),
        "has_html":      int(bool(re.search(r"<[^>]+>", text))),
        "trigger_count": sum(1 for w in spam_trigger_words if w in text.lower()),
    }

def predict(text, model_name):
    processed    = preprocess(text)
    tfidf_vec    = tfidf.transform([processed])
    feats        = extract_features(text)
    feat_df      = pd.DataFrame([feats])
    structural   = scaler.transform(feat_df)
    struct_sparse = csr_matrix(structural)
    X_combined   = hstack([tfidf_vec, struct_sparse])

    model = models[model_name]

    if model_name == "Naive Bayes":
        from sklearn.preprocessing import MinMaxScaler as MMS
        mms = MMS()
        X_combined = csr_matrix(mms.fit_transform(X_combined.toarray()))

    pred = model.predict(X_combined)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_combined)[0]
        confidence = prob[1] if pred == 1 else prob[0]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_combined)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
    else:
        confidence = 1.0

    return pred, confidence, feats

# ── UI ────────────────────────────────────────────────────────
st.title("AI-Based Spam Email Detection System")
st.markdown("**Project 03 | NLP / Machine Learning | Newton School of Technology**")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Email / SMS Message")
    user_input = st.text_area(
        "Paste your message here:",
        height=200,
        placeholder="e.g. Congratulations! You have won a free iPhone. Click here to claim now!"
    )
    model_name = st.selectbox("Select Model", list(models.keys()))

    if st.button("Classify Message", use_container_width=True):
        if user_input.strip():
            pred, confidence, feats = predict(user_input, model_name)

            st.markdown("---")
            if pred == 1:
                st.error(f"**SPAM DETECTED** — Confidence: {confidence:.1%}")
            else:
                st.success(f"**LEGITIMATE (HAM)** — Confidence: {confidence:.1%}")

            st.progress(float(confidence))

            st.markdown("### 🔍 Feature Analysis")
            feat_df = pd.DataFrame([feats]).T
            feat_df.columns = ["Value"]
            st.dataframe(feat_df, use_container_width=True)
        else:
            st.warning("Please enter a message first!")

with col2:
    st.subheader("📊 Model Info")
    st.info(f"**Selected:** {model_name}")

    st.markdown("### 🧪 Try These Examples")
    examples = {
        "Spam 1": "URGENT! You have won a 1000 prize! Call now to claim FREE reward!",
        "Spam 2": "Congratulations! Click here to get your FREE iPhone. Limited offer!",
        "Ham 1":  "Hey, are you coming to the meeting tomorrow at 10am?",
        "Ham 2":  "Can you pick up some groceries on the way home please?",
    }
    for label, example in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["example"] = example

    st.markdown("### All Models Comparison")
    try:
        df_results = pd.read_csv("reports/model_comparison.csv", index_col=0)
        st.dataframe(df_results, use_container_width=True)
    except:
        st.info("Run Phase 3 first to see model comparison")

st.markdown("---")
st.markdown("*Built by Hardik Hathwal | B.Tech AI | Newton School of Technology*")
