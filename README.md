# Natural Language Processing Projects Collection

A collection of various NLP projects.

## Projects

### 1. Spam Classifier
- This project is a simple Natural Language Processing (NLP) model that classifies text messages as spam or ham. The dataset was cleaned and preprocessed using techniques such as lemmatization and stopword removal. A Bag-of-Words model was built using CountVectorizer, and a Multinomial Naive Bayes classifier was trained for prediction. The model achieved high accuracy of 98.38% in identifying spam messages.


### 2. Stock Price Sentiment Analysis
- This project involves building a machine learning model to predict stock market sentiment (positive or negative movement) based on top financial news headlines. The data was preprocessed using text cleaning techniques and vectorized using TF-IDF with unigrams and bigrams. A Random Forest Classifier was tuned using time-series cross-validation to ensure reliable performance on unseen data. The final model was evaluated using accuracy, confusion matrix, and ROC AUC, and key influential words were identified using feature importance.


### 3. Fake News Classifier
- This project focuses on classifying news as fake or real using various machine learning models. After preprocessing and transforming the text data using TF-IDF, four models—Naive Bayes, Logistic Ridge Regression, Random Forest, and XGBoost—were trained and evaluated. The Random Forest Classifier achieved the best performance based on accuracy and ROC AUC score. Model evaluation also included feature importance analysis to identify the most influential words in predicting fake news.

### 4. Fake News Classifier with LSTM
- This extension of the Fake News Classifier project explores deep learning techniques for fake news detection using a Long Short-Term Memory (LSTM) neural network. After comprehensive text preprocessing—including tokenization, padding, and embedding—the LSTM model was trained to classify news articles as real or fake. The model architecture was designed to capture sequential dependencies in the text, leveraging word embeddings to improve semantic understanding. Performance was evaluated using metrics such as accuracy. Compared to traditional models, the LSTM classifier showed improved accuracy, making it especially effective in identifying fake news.

### 5. Consumer Complaints Classifier

This project develops and evaluates a machine learning model to classify consumer complaints based on their narrative descriptions. Leveraging the power of **pre-trained transformer models (specifically DistilBERT) and TensorFlow**, this system aims to automatically categorize complaints into predefined product categories, facilitating quicker analysis and routing.


#### Project Overview

The core objective of this project is to build a robust classification system for consumer complaints. By training a transformer model on historical complaint data, the system learns to identify patterns in complaint narratives and assign them to the most relevant product category. This automation can significantly reduce manual effort, improve efficiency in handling complaints, and provide insights into common complaint areas.

#### Features

* **Data Loading & Preprocessing:** Handling of CSV data, including dropping missing/empty complaint narratives.
* **Label Encoding:** Converts categorical product labels into numerical representations for model training.
* **Stratified Data Splitting:** Ensures balanced distribution of complaint categories across training, validation, and test sets, crucial for imbalanced datasets.
* **Transformer-based Tokenization:** Utilizes Hugging Face's `AutoTokenizer` for efficient and consistent text tokenization suitable for pre-trained models.
* **TF.data.Dataset Creation:** Optimizes data pipeline for TensorFlow training, leveraging prefetching for performance.
* **Fine-tuning Pre-trained Model:** Leverages `TFAutoModelForSequenceClassification` to fine-tune `distilbert-base-uncased` for text classification.
* **Comprehensive Evaluation:** Provides detailed classification reports, confusion matrices, accuracy, F1-score, precision, and recall.
* **Model Saving:** Saves the fine-tuned model and tokenizer for future inference without re-training.

#### Dataset

The project uses a dataset of consumer complaints. The default expected file is `complaints.csv`.

* **Source:** `https://www.kaggle.com/datasets/selener/consumer-complaint-database`

#### Model Architecture

The model is based on the **DistilBERT** architecture, a smaller, faster, and lighter version of BERT. It's chosen for its balance of performance and efficiency, making it suitable for fine-tuning on downstream tasks like text classification.

* **Base Model:** `distilbert-base-uncased` from Hugging Face Transformers.
* **Task:** Sequence Classification.
* **Framework:** TensorFlow 2.x and Keras.
