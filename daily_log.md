# Daily Development Log — Spam Email Detection System

**Project:** AI-Based Spam Email Detection System
**Intern:** Hardik Hathwal | B.Tech AI | Newton School of Technology
**Duration:** Week 1–3 (Summer Internship 2026)

---

## Week 1 — Data & Preprocessing

### Day 1
**Progress:** Project kickoff. Defined scope: binary classification (spam/ham) using NLP. Set up virtual environment, installed NLTK, scikit-learn, Streamlit. Created project folder structure.
**Blockers:** None.
**Learnings:** Reviewed key NLP preprocessing techniques — stop word removal, stemming vs. lemmatization. Chose Porter Stemmer for speed.

### Day 2
**Progress:** Built the synthetic dataset generation script (Phase 1). Created ~100 spam templates (prize scams, urgent offers, phishing lures) and ~100 ham templates (casual messages, work communication). Verified class balance.
**Blockers:** Defining a realistic variety of spam patterns without using any real-world PII.
**Learnings:** Even a small well-curated dataset outperforms a large noisy one in early prototype stages.

### Day 3
**Progress:** Completed `Phase1_Data_Exploration.ipynb`. Plotted label distribution — 50/50 balanced. Analysed message length distributions: spam messages tend to be shorter with more uppercase and special characters. Generated initial word clouds.
**Blockers:** NLTK `punkt_tab` tokenizer not downloaded by default — added explicit `nltk.download('punkt_tab')` call.
**Learnings:** Word clouds are visually useful but not a substitute for proper feature analysis. Need quantitative feature importance later.

### Day 4
**Progress:** Completed `Phase2_Preprocessing.ipynb`. Implemented `clean_text()` (URL/email stripping, lowercase), `preprocess()` (tokenization + stemming), and `extract_features()` (9 structural features). Serialized `feature_scaler.pkl` and `tfidf_vectorizer.pkl`.
**Blockers:** Deciding on the exact 9 structural features — consulted 2 spam classification papers to validate feature choices.
**Learnings:** Combining TF-IDF with structural features improves detection of spam that avoids trigger words (e.g., encoded spam). The `upper_ratio` feature alone gives ~0.62 AUC.

### Day 5
**Progress:** Started `Phase3_Model_Training.ipynb`. Implemented the full pipeline: TF-IDF vectorization → structural feature scaling → `hstack` to combine sparse matrices → trained Naive Bayes and Logistic Regression. Both hit >90% accuracy on the held-out set.
**Blockers:** Naive Bayes requires non-negative features. Discovered SciPy sparse matrix + `MinMaxScaler` workaround. Documented this in code comments.
**Learnings:** Naive Bayes is very sensitive to feature scaling. Always check classifier assumptions before applying.

---

## Week 2 — Model Training & Evaluation

### Day 6
**Progress:** Added Random Forest and SVM (Linear) to the training notebook. Performed 5-fold cross-validation. Saved all four model `.pkl` files. Generated `final_evaluation_report.csv`.
**Blockers:** SVM training slow on dense feature matrices. Used `LinearSVC` for speed.
**Learnings:** Random Forest and SVM both outperformed Naive Bayes on F1 score. Logistic Regression provides best interpretability for production debugging.

### Day 7
**Progress:** Completed `Phase4_Evaluation.ipynb`. Generated confusion matrices, ROC curves, Precision-Recall curves, and FP/FN analysis for all four models. Saved all plots to `reports/`.
**Blockers:** Plotting 4 ROC curves on one axis required careful legend management.
**Learnings:** False negatives (spam classified as ham) are the more harmful error in most real-world scenarios. Random Forest had the lowest FN rate.

### Day 8
**Progress:** Reviewed all Phase 1–4 notebooks for code quality. Added missing docstrings, reorganized imports to PEP-8 order, and checked line lengths. Verified all model `.pkl` files load correctly from a fresh Python environment.
**Blockers:** `joblib` version mismatch between training and loading environments. Pinned version in `requirements.txt`.
**Learnings:** Always pin exact library versions in `requirements.txt` when serializing sklearn objects.

### Day 9
**Progress:** Started `Phase5_Streamlit_App.ipynb`. Prototyped the UI layout: two-column design with text area + model selector on left, model info + examples on right. Wired up `predict()` function to the button click.
**Blockers:** Streamlit `st.cache_resource` vs `st.cache_data` distinction — used `cache_resource` for model objects (non-hashable).
**Learnings:** Streamlit's caching system is powerful but requires understanding the difference between data (serializable) and resources (non-serializable like models).

### Day 10
**Progress:** Completed Streamlit app. Added feature breakdown dataframe, progress bar for confidence, and model comparison table. Tested all four models end-to-end with 10 sample messages.
**Blockers:** None.
**Learnings:** User testing revealed that displaying the "Spam Trigger Words" count is very valuable — users immediately understand why a message was flagged.

---

## Week 3 — Polish & Documentation

### Day 11
**Progress:** Ran `pycodestyle` across all `.py` files. Fixed 12 style violations (line length, spacing around operators). Added module-level docstring to `app.py`. Updated `README.md` with full project structure and ethical considerations.
**Blockers:** Balancing PEP-8 line length (79 chars) with Streamlit string formatting — settled on 100-char limit for UI strings.
**Learnings:** Running a linter early would have saved time. Will use `flake8` from day one on future projects.

### Day 12
**Progress:** Completed `written_report.md` — 15 pages covering introduction, dataset construction, preprocessing pipeline, model comparison, feature importance analysis, ethical considerations, and conclusions. Added full references section.
**Blockers:** None.
**Learnings:** Writing the report after completing the code reveals gaps in documentation. Good inline comments make writing reports much faster.

### Day 13
**Progress:** Final review session. Verified all files are present and correctly structured. Checked that `.gitignore` excludes `venv/` and `.pkl` files. Ran the Streamlit app one final time to confirm clean startup.
**Blockers:** None.
**Learnings:** A clear project structure from Day 1 saves significant time during the final review. Folder naming conventions matter for reproducibility.
