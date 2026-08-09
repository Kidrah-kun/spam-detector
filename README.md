# 📧 AI-Based Spam Email Detection System

> **Summer Internship Project — Newton School of Technology | B.Tech AI**

A machine learning system that classifies email and SMS messages as **spam** or **legitimate (ham)** using a hybrid feature pipeline combining TF-IDF text embeddings with 9 hand-engineered structural features.

---

## 🎯 Objectives

- Build an end-to-end NLP pipeline for binary text classification
- Engineer structural features that complement bag-of-words representations
- Train and compare four ML classifiers on realistic spam/ham data
- Deploy a real-time interactive dashboard using Streamlit

---

## ✨ Features

- **Text Preprocessing**: Lowercases, removes URLs/emails, punctuation, and applies Porter stemming via NLTK
- **Feature Extraction**: Combines TF-IDF (vocabulary-based) with 9 structural features (message length, uppercase ratio, link count, exclamation marks, spam trigger word count, etc.)
- **Multi-Model Comparison**: Naive Bayes, Logistic Regression, Random Forest, SVM (Linear)
- **Streamlit Dashboard**: Real-time predictions with confidence scores, feature breakdowns, and model comparison table

---

## 🛠️ Tech Stack

| Layer | Library |
|-------|---------|
| UI | Streamlit |
| ML Models | Scikit-Learn |
| NLP | NLTK (tokenization, stop words, stemming) |
| Features | Pandas, NumPy, SciPy (sparse matrices) |

---

## 📊 Dataset

- **Source**: Synthetic dataset of common spam and ham message templates (constructed during Phase 1)
- **Type**: Open, no personal data — generated using publicly known spam patterns and neutral communication templates
- **Classes**: Spam (promotional, phishing, prize-scam messages) | Ham (casual, professional messages)
- **Size**: ~2,000 labelled samples across 5 phases of iterative expansion

> No proprietary or personally identifiable data was used. See `notebooks/Phase1_Data_Exploration.ipynb` for full dataset construction details.

---

## 📁 Project Structure

```
spam-detector/
├── app/
│   └── app.py                  # Streamlit web app (Phase 5)
├── data/                       # Training data (generated in notebooks)
├── models/                     # Serialized .pkl model files
│   ├── tfidf_vectorizer.pkl
│   ├── feature_scaler.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── notebooks/
│   ├── Phase1_Data_Exploration.ipynb
│   ├── Phase2_Preprocessing.ipynb
│   ├── Phase3_Model_Training.ipynb
│   ├── Phase4_Evaluation.ipynb
│   └── Phase5_Streamlit_App.ipynb
├── reports/
│   ├── final_evaluation_report.csv
│   └── written_report.md       # Full project report (10–20 pages)
├── daily_log.md                # Day-by-day development journal
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and set up environment
```bash
git clone <your-repo-url>
cd spam-detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models (run notebooks in order)
```bash
jupyter notebook notebooks/
```
Run `Phase1` → `Phase2` → `Phase3` → `Phase4` → `Phase5` in sequence.

### 3. Launch the Streamlit app
```bash
cd app
streamlit run app.py
```

---

## ⚖️ Ethical Considerations

### Bias in Training Data
The synthetic dataset was constructed to be balanced and diverse, but it reflects English-language spam patterns. The system may perform less reliably on:
- Non-English messages or transliterated spam
- Novel spam campaigns not represented in the training patterns

Deployment in production systems should be accompanied by continuous monitoring and periodic retraining on real-world data.

### Data Privacy
- **No messages are stored** — the Streamlit app processes each input in-memory and discards it immediately
- No user identification or logging takes place
- The system is designed as a **demonstration prototype**, not a production email filter

### Responsible Use of AI
- The model has non-zero false positive rates — legitimate messages may occasionally be flagged as spam
- This system should **assist human review**, not replace it, in high-stakes scenarios (e.g., enterprise email filtering)
- Spam classifiers must not be weaponized to identify and selectively suppress communications

---

## 📚 References & Acknowledgements

1. **NLTK Library** — Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/
2. **Scikit-Learn** — Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830. https://scikit-learn.org/
3. **TF-IDF Spam Classification Survey** — Sriram et al. (2010). *Short text classification in twitter to improve information filtering*. ACM SIGIR.
4. **Porter Stemmer** — Porter, M. F. (1980). *An algorithm for suffix stripping*. Program, 14(3), 130–137.
5. **Streamlit** — Streamlit Inc. (2024). https://streamlit.io/

---

*Built by Hardik Hathwal | B.Tech AI | Newton School of Technology | Summer Internship 2026*
