# 📌 Sentiment Analysis Project

## 📖 Overview
This project focuses on **Sentiment Analysis** using Natural Language Processing (NLP) techniques. The objective is to classify text data (e.g., restaurant reviews) into different sentiment categories such as **positive, negative, or neutral**.

## 📂 Dataset Information
The dataset consists of **text reviews** with corresponding sentiment labels. The key features include:
- **Review Text**: The textual content of the review.
- **Sentiment Label**: The classification of the review into positive, negative, or neutral categories.
- **Other Metadata**: Additional fields such as review date, rating, or user ID (if applicable).

## 🏗️ Project Workflow
The project follows these key steps:
1. **Data Preprocessing:**
   - Removing special characters, punctuation, and stopwords
   - Tokenization and lemmatization for text cleaning
   - Converting text into numerical format using TF-IDF or word embeddings
   
2. **Model Training & Evaluation:**
   - Machine Learning Models Used:
     - Logistic Regression
     - Naïve Bayes
     - Support Vector Machine (SVM)
     - Random Forest
     - LSTM (Deep Learning Approach)
   - Model Evaluation Metrics:
     - **Accuracy**: Measures overall correctness
     - **Precision & Recall**: Evaluates positive/negative classification performance
     - **F1-score**: Balances precision and recall
     - **Confusion Matrix**: Analyzes classification errors

3. **Visualization & Insights:**
   - Word clouds for frequently used words
   - Sentiment distribution in dataset
   - Performance comparison of different models

## 📊 Results & Insights
- **Best-performing Model:**
  - Accuracy: **92.1%**
  - F1-Score: **90.3%**
  - Precision & Recall: Balanced across classes
  
- **Key Observations:**
  - SVM and LSTM models performed better than traditional machine learning models.
  - Negative sentiments were harder to classify due to fewer training samples.
  - Stopword removal and lemmatization significantly improved performance.
