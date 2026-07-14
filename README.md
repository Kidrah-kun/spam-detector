# 📧 Spam Detector

A machine learning web application that classifies messages as spam or legitimate (ham). It uses a synthetic dataset of common spam and ham templates to demonstrate text classification.

## Features
- **Text Preprocessing**: Cleans input text using NLTK (removes URLs, punctuation, and applies stemming).
- **Feature Extraction**: Converts text to numerical format using TF-IDF and extracts 9 custom structural features (like message length and uppercase ratio).
- **Machine Learning**: Uses models (like Random Forest and Logistic Regression) to classify the combined features.
- **Web Interface**: A simple Streamlit dashboard for real-time predictions.

## Tech Stack
- **UI**: Streamlit
- **Machine Learning**: Scikit-Learn, NLTK, SciPy
- **Data Handling**: Pandas, NumPy

## Getting Started
1. Open a terminal and navigate to the project directory:
   ```bash
   cd app
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
