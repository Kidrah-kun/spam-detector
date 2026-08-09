# Written Report: AI-Based Spam Email Detection System

**Course:** B.Tech Summer Internship — AI, ML & Cybersecurity Projects
**Author:** Hardik Hathwal
**Institution:** Newton School of Technology
**Date:** August 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Objectives](#2-objectives)
3. [Dataset Description](#3-dataset-description)
4. [System Architecture](#4-system-architecture)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training & Selection](#7-model-training--selection)
8. [Evaluation & Results](#8-evaluation--results)
9. [Deployment](#9-deployment)
10. [Ethical Considerations](#10-ethical-considerations)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [Conclusions](#12-conclusions)
13. [References](#13-references)

---

## 1. Introduction

Spam messages — unsolicited bulk communications sent for commercial, fraudulent, or malicious purposes — represent a persistent threat to digital communication systems. According to Statista (2024), approximately 45% of all email traffic globally is classified as spam. Beyond mere inconvenience, spam serves as a primary vector for phishing attacks, malware distribution, and financial fraud.

Traditional rule-based spam filters (e.g., keyword blacklists) are brittle: spammers rapidly adapt by introducing deliberate misspellings, encoding tricks, or rotating trigger words. Machine learning approaches offer a more robust and adaptive alternative, learning statistical patterns from labelled data that generalize to unseen spam variants.

This project implements a complete end-to-end spam classification pipeline using Natural Language Processing (NLP) and classical machine learning algorithms. The system is deployed as an interactive Streamlit web application, allowing real-time message classification with a confidence score and feature breakdown.

---

## 2. Objectives

| # | Objective |
|---|-----------|
| O1 | Design and implement a reproducible NLP preprocessing pipeline |
| O2 | Engineer structural features complementing bag-of-words representations |
| O3 | Train, compare, and evaluate four ML classifiers |
| O4 | Analyse and document false positive/negative characteristics |
| O5 | Deploy a functional prototype accessible via a web browser |
| O6 | Address ethical implications of automated message classification |

---

## 3. Dataset Description

### 3.1 Source and Construction

A synthetic dataset was constructed during Phase 1 of the project, consisting of message templates spanning common spam and ham categories:

| Category | Sub-types | Count |
|----------|-----------|-------|
| **Spam** | Prize scams, phishing lures, loan offers, urgency-based promotions | ~1,000 |
| **Ham** | Casual conversation, work communication, appointment scheduling | ~1,000 |

The dataset is entirely synthetic and contains **no personally identifiable information (PII)**. Templates were generated based on publicly documented spam patterns (e.g., from the 2006 Enron Email Dataset taxonomy and SpamAssassin corpus patterns), not copied from real user communications.

### 3.2 Class Distribution

The dataset was deliberately balanced (50/50) to avoid the class imbalance problem, which can artificially inflate accuracy metrics and bias models toward the majority class.

### 3.3 Key Observations from EDA

- Spam messages are on average **shorter** but contain more uppercase characters and exclamation marks
- Ham messages contain more pronouns ("I", "you", "we") and time references ("tomorrow", "meeting")
- Top spam tokens after preprocessing: `free`, `win`, `cash`, `urgnt`, `click`, `claim`
- Top ham tokens: `meet`, `call`, `tomorrow`, `home`, `pleas`, `thank`

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Input (Raw Text)                  │
└─────────────────────────┬────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
    ┌─────────▼──────┐     ┌──────────▼──────────┐
    │  Text Cleaning  │     │  Structural Features │
    │  + Stemming     │     │  Extraction (9 feat) │
    └─────────┬──────┘     └──────────┬──────────┘
              │                       │
    ┌─────────▼──────┐     ┌──────────▼──────────┐
    │ TF-IDF Vectorize│     │  StandardScaler      │
    │ (sparse matrix) │     │  → csr_matrix        │
    └─────────┬──────┘     └──────────┬──────────┘
              │                       │
              └───────────┬───────────┘
                          │  hstack (combine)
                ┌─────────▼────────┐
                │   ML Classifier   │
                │  (NB / LR / RF /  │
                │      SVM)         │
                └─────────┬────────┘
                          │
              ┌───────────▼───────────┐
              │   Prediction + Score  │
              │   Spam / Ham + Conf.  │
              └───────────────────────┘
```

---

## 5. Preprocessing Pipeline

### 5.1 Text Cleaning (`clean_text`)

Applied in order:
1. Convert to lowercase
2. Remove URLs (`http://`, `www.`) with regex `r"http\S+|www\S+"`
3. Remove email addresses with regex `r"\S+@\S+"`
4. Remove all non-alphabetic characters (digits, punctuation)
5. Collapse multiple spaces into single spaces

### 5.2 Tokenization & Stemming (`preprocess`)

1. Word tokenization using NLTK's `word_tokenize` (Punkt tokenizer)
2. Stop word removal using NLTK's English stop word list (179 words)
3. Porter Stemming — reduces inflected words to their root form (e.g., "winning" → "win", "claimed" → "claim")
4. Minimum token length filter: tokens shorter than 3 characters are discarded

### 5.3 TF-IDF Vectorization

- Fitted on training corpus; applied to test set without refitting (to prevent data leakage)
- Parameters: `max_features=5000`, `ngram_range=(1,1)`, `sublinear_tf=True`

---

## 6. Feature Engineering

### 6.1 Structural Features

Nine hand-engineered features capture surface-level statistical properties:

| Feature | Description | Spam Signal |
|---------|-------------|-------------|
| `msg_length` | Total character count | Spam often shorter |
| `word_count` | Whitespace-separated token count | Spam often fewer words |
| `num_links` | Count of http/https/www occurrences | Spam contains more links |
| `num_exclaim` | Count of `!` characters | Spam uses more exclamation marks |
| `num_special` | Count of non-alphanumeric chars | Spam uses special chars to evade filters |
| `num_digits` | Count of digit characters | Prize amounts, phone numbers |
| `upper_ratio` | Fraction of uppercase characters | Spam uses ALL CAPS for urgency |
| `has_html` | 1 if HTML tags detected | HTML emails used for phishing |
| `trigger_count` | Count of known spam trigger words | Direct spam signal |

### 6.2 Feature Scaling

Structural features were standardized using `sklearn.preprocessing.StandardScaler` fitted on the training set. For Naive Bayes (which requires non-negative inputs), a `MinMaxScaler` is applied at inference time to the combined sparse feature matrix.

### 6.3 Feature Combination

TF-IDF sparse matrix and scaled structural features (converted to sparse) are concatenated using `scipy.sparse.hstack`, creating a single combined feature matrix passed to each classifier.

---

## 7. Model Training & Selection

### 7.1 Models Evaluated

| Model | Rationale |
|-------|-----------|
| **Naive Bayes** | Fast, probabilistic baseline; natural fit for text classification |
| **Logistic Regression** | Linear, interpretable; strong baseline for high-dimensional text |
| **Random Forest** | Non-linear, ensemble; robust to feature noise |
| **SVM (Linear)** | Effective in high-dimensional spaces; good generalization |

### 7.2 Training Protocol

- **Split**: 80% training / 20% held-out test
- **Validation**: 5-fold stratified cross-validation on training set
- **Metric priority**: F1-score (harmonic mean of Precision and Recall) — more informative than accuracy on classification tasks

### 7.3 Hyperparameter Choices

- **Random Forest**: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2`
- **Logistic Regression**: `C=1.0`, `solver='lbfgs'`, `max_iter=1000`
- **SVM**: `kernel='linear'`, `C=1.0` (via `LinearSVC`)
- **Naive Bayes**: `alpha=1.0` (Laplace smoothing)

---

## 8. Evaluation & Results

### 8.1 Model Comparison (held-out test set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|:--------:|:---------:|:------:|:--------:|:---:|
| Naive Bayes | 0.891 | 0.874 | 0.912 | 0.893 | 0.947 |
| Logistic Regression | 0.934 | 0.929 | 0.941 | 0.935 | 0.978 |
| **Random Forest** | **0.961** | **0.958** | **0.964** | **0.961** | **0.992** |
| SVM Linear | 0.948 | 0.945 | 0.952 | 0.948 | 0.985 |

### 8.2 Key Findings

- **Random Forest** achieves the best overall F1 (0.961) and lowest false negative rate
- **Logistic Regression** offers the best trade-off between performance and interpretability
- **Naive Bayes** underperforms due to the feature independence assumption being violated by structural features
- All four models benefit substantially from combining TF-IDF with structural features — ablation showed ~8% F1 drop when using TF-IDF alone

### 8.3 Feature Importance (Random Forest)

Top 5 most predictive features:
1. `trigger_count` (spam trigger words)
2. `upper_ratio` (uppercase fraction)
3. TF-IDF token: `free`
4. `num_exclaim`
5. TF-IDF token: `win`

---

## 9. Deployment

The system is deployed as a **Streamlit web application** (`app/app.py`):

- **Input**: Text area accepting any message (email body, SMS, etc.)
- **Model Selector**: Dropdown to switch between all four trained classifiers
- **Output**: Spam/Ham label + confidence percentage + feature breakdown table
- **Model Comparison Panel**: Displays the evaluation report table from Phase 4

**Running the app:**
```bash
cd app
streamlit run app.py
```

The prototype is fully functional locally and can be deployed to Streamlit Community Cloud with no code changes.

---

## 10. Ethical Considerations

### 10.1 Bias in Training Data

The training dataset was constructed from synthetic English-language templates. Potential biases include:

- **Language bias**: The system is trained exclusively on English. Performance on Hinglish, regional language spam, or non-Latin scripts is untested and likely poor.
- **Template diversity**: Despite covering common spam patterns, novel attack vectors (e.g., context-aware AI-generated phishing) may evade detection.
- **False positive risk**: Legitimate promotional messages (e.g., sale announcements from known brands) may resemble spam. False positives disproportionately affect small businesses.

**Mitigation**: In a production deployment, the model should be continuously retrained on domain-specific data and monitored for demographic disparities in false positive rates.

### 10.2 Data Privacy

- This prototype processes all input **in-memory** — no messages are logged, stored, or transmitted
- No user authentication or session tracking is implemented
- The system is intended for demonstration purposes; integration with real email systems would require a thorough privacy impact assessment

### 10.3 Responsible Use

- Spam classifiers must not be used to selectively suppress legitimate communications based on content (e.g., political speech, whistleblowing)
- Model decisions should remain auditable — the feature breakdown panel in the app supports this principle
- Human review should remain in the loop for borderline confidence scores (<70%)

---

## 11. Limitations & Future Work

| Limitation | Proposed Solution |
|------------|------------------|
| Synthetic dataset may not generalize to real-world spam | Retrain on SpamAssassin or Enron corpus |
| No handling of image-based spam | Add OCR (Tesseract) preprocessing |
| No URL content analysis | Integrate URL reputation API |
| English-only | Multilingual BERT fine-tuning |
| No concept drift detection | Implement periodic model retraining pipeline |

---

## 12. Conclusions

This project successfully demonstrates a functional spam detection system combining TF-IDF NLP features with hand-engineered structural features. The hybrid approach outperforms TF-IDF-only baselines and provides interpretable feature breakdowns that support human review.

Key takeaways:
1. Structural features significantly complement vocabulary-based features
2. Random Forest achieves the best performance; Logistic Regression offers the best interpretability
3. A synthetic balanced dataset is sufficient for a prototype but must be replaced with real-world data for production
4. Ethical safeguards (no data persistence, human-in-the-loop for borderline cases) are essential for responsible deployment

---

## 13. References

1. Drucker, H., et al. (1999). *Support Vector Machines for Spam Categorization*. IEEE Transactions on Neural Networks, 10(5), 1048–1054.
2. Sahami, M., et al. (1998). *A Bayesian Approach to Filtering Junk E-Mail*. AAAI Workshop on Learning for Text Categorization.
3. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/
4. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830. https://scikit-learn.org/
5. SpamAssassin Public Corpus. Apache Software Foundation. https://spamassassin.apache.org/old/publiccorpus/
6. Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
7. Statista (2024). *Share of spam in global email traffic 2014–2023*. https://www.statista.com/statistics/420391/spam-email-traffic-share/
