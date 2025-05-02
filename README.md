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
