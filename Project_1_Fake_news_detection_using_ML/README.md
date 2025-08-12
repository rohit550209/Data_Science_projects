📄 Fake News Detection
📌 Overview
This project aims to build a machine learning model that can classify news articles as Fake or True based on their content.
It uses Natural Language Processing (NLP) techniques for text preprocessing and a classification model to detect fake news.

📂 Dataset
The dataset consists of two CSV files:

Fake.csv – Collection of fake news articles.

True.csv – Collection of true/real news articles.

Dataset Source: Fake News Dataset on Kaggle

Each dataset contains news title, text, and subject. A new binary column Fake is created for labeling:

1 → Fake News

0 → True News

🛠 Features
Text Preprocessing:

Lowercasing text

Removing punctuation & special characters

Removing stopwords

Tokenization

Model Training & Evaluation:

Train-test split

Vectorization (CountVectorizer / TF-IDF)

Classification model (e.g., Logistic Regression / Naive Bayes)

Accuracy & classification report

📊 Results
The trained model achieves strong accuracy in distinguishing between fake and true news, as demonstrated in the classification metrics.

📌 Future Improvements
Experiment with deep learning models (LSTM, BERT)

Deploy as a web app with Flask or Streamlit

Perform hyperparameter tuning for better accuracy
