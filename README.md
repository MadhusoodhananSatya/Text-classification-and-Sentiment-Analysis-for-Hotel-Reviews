🌟 Restaurant Review Sentiment Classification
=============================================

This project applies and compares multiple machine learning algorithms to predict whether a restaurant review is **positive** or **negative**. The goal is to analyze customer feedback and identify which models provide the best performance for text classification using **Natural Language Processing (NLP)**.

The core of this project is a **data science pipeline** that starts with **text preprocessing**, followed by **feature engineering**, and concludes with the application of **machine learning models**. Using a dataset of restaurant reviews, the program learns from textual patterns to classify sentiments effectively.

The project also addresses common NLP challenges such as **stopword removal**, **stemming/lemmatization**, and **high dimensionality of features**. Different vectorization methods like **Bag of Words** and **TF-IDF** are tested to improve representation of text data.

The final models are evaluated using classification reports, confusion matrices, and ROC-AUC scores to provide a clear and comprehensive assessment of their performance.

* * * * *

✨ Key Features & Technical Details
----------------------------------

### 📊 Exploratory Data Analysis (EDA)

-   Visualized sentiment distribution between positive and negative reviews.

-   Generated **word clouds** to identify the most frequent words in each sentiment class.

-   Analyzed review lengths and frequent terms across the dataset.

### 🧹 Data Preprocessing

-   Removed special characters, punctuation, and numbers.

-   Converted text to lowercase.

-   Applied **tokenization**, **stopword removal**, and **stemming/lemmatization**.

-   Transformed text into numerical features using:

    -   **CountVectorizer (Bag of Words)**

    -   **TF-IDF Vectorizer**

### ⚖️ Feature Engineering & Scaling

-   Constructed uni-grams and bi-grams to capture word sequences.

-   Controlled sparsity by limiting max features in vectorizers.

### 🤖 Model Implementation & Tuning

-   **Naïve Bayes**: Lightweight baseline model for text classification.

-   **Logistic Regression**: Strong performance with TF-IDF features.

-   **Support Vector Machine (SVM)**: Tested linear kernel for higher accuracy.

-   **Random Forest / XGBoost** (optional advanced): Ensemble approaches to capture non-linearities.

### 📈 Model Evaluation

-   Accuracy, Precision, Recall, F1-score metrics.

-   Confusion matrix visualization for error analysis.

-   ROC-AUC curve to evaluate classifier performance.

* * * * *

🚀 Getting Started
------------------

To run this project, you will need a Python environment with the following libraries:

-   pandas

-   numpy

-   matplotlib

-   seaborn

-   scikit-learn

-   nltk

-   wordcloud

You can set up the environment and run the analysis by opening the `SentimentAnalysis on Restaurant Review.ipynb` file in a Jupyter Notebook environment.

* * * * *

📊 Project Workflow
-------------------

The notebook follows a structured NLP workflow:

1.  **Data Loading & Inspection**: Load the dataset of restaurant reviews and check for null or duplicate values.

2.  **Data Cleaning & Preprocessing**: Clean text, tokenize, remove stopwords, and apply stemming/lemmatization.

3.  **Feature Extraction**: Transform reviews into numerical features using Bag of Words and TF-IDF.

4.  **Exploratory Data Analysis (EDA)**: Visualize review length, sentiment distribution, and frequent words.

5.  **Model Training**: Train Naïve Bayes, Logistic Regression, and SVM models on the processed data.

6.  **Evaluation & Comparison**: Compare models using classification metrics and confusion matrices.

7.  **Prediction Function**: Implement a function to predict sentiment for new restaurant reviews.

* * * * *

📈 Final Thoughts
-----------------

The analysis demonstrates the effectiveness of machine learning for sentiment classification in the restaurant industry. Among the tested models, **Logistic Regression with TF-IDF features** provided the best balance of accuracy and interpretability.

This project serves as a strong foundation for further exploration, such as experimenting with **deep learning models (LSTMs, BERT)** or deploying the solution as a **web app** for real-time sentiment prediction.

* * * * *

🙏 Acknowledgments
------------------

Thanks to the developers of **NLTK, scikit-learn, matplotlib, seaborn, and wordcloud** libraries for providing the tools that made this analysis possible.
