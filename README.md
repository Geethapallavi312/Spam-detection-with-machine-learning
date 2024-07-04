# Spam-detection-with-machine-learning and NLP

This project demonstrates the use of machine learning techniques to detect spam messages in a dataset of SMS messages. The goal is to classify messages as 'spam' or 'ham' (non-spam) using various classifiers and evaluate their performance.

## Project Overview
### Data Import and Preprocessing:

Load the dataset containing SMS messages.
Remove duplicates and handle missing values.
Encode the target variable ('ham' or 'spam').
### Text Processing using NLP:

Clean and preprocess the text data by removing non-alphabet characters, converting to lowercase, removing stopwords, and applying stemming.
Transform the text data into numerical vectors using TF-IDF vectorization.
### Model Training and Evaluation:

Train multiple machine learning classifiers including Naive Bayes, Logistic Regression, SVM, Decision Trees, KNN, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost.
Evaluate each model's performance using metrics such as accuracy, precision, recall, and F1-score.
Identify the best performing model based on accuracy.
### Results Visualization:

Visualize the performance of different models using bar plots.
### Prediction on New Data:

Preprocess and vectorize new input messages.
Predict the class (spam or ham) of new messages using the best performing model.
## Getting Started
## Prerequisites

Python 3.x
Required libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, xgboost
Ensure you have the dataset (spam.csv) in the project directory. You can download the dataset from kaggle directly or with the below link.
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

#### Project Structure
spam.csv: Dataset containing SMS messages labeled as 'ham' or 'spam'.
spam_detection.py: Main script containing the code for data preprocessing, model training, evaluation, and prediction.

## Results
The best performing model in this project was the Extra Trees Classifier, which achieved the highest accuracy on the test set.

## Conclusion
This project provides an end-to-end solution for detecting spam messages using machine learning. It includes data preprocessing, text processing using NLP, training and evaluating multiple classifiers, and making predictions on new data.

## Future Scope
Experiment with additional feature engineering techniques, such as word embeddings (Word2Vec, GloVe) or incorporating metadata features (e.g., message length, presence of special characters).
Develop a real-time spam detection system using a streaming platform like Apache Kafka or a web framework like Flask/Django for practical deployment.
